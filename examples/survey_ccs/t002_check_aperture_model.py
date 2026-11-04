"""Check M2 aperture geometry, knob updates and generated tracking limits.

Run with ``python t002_check_aperture_model.py``. The existing t000 check covers
optics and survey equivalence; this check uses MAD-X as the aperture reference.
"""

from pathlib import Path

import numpy as np
from cpymad.madx import Madx
import xtrack as xt
from xtrack.aperture.profile_converters import profile_from_madx_aperture

from job_use_aperture import make_tracking_line


here = Path(__file__).resolve().parent
env = xt.load(here / 'm2_mtn3p8_v6_notilt.py')
line = env['m2']
assert len(env.aperture.profiles) == 31
assert len(line.aperture.pipe_positions) == 104
assert not line.aperture.is_built
assert not any(type(line.element_dict[name]).__name__.startswith('Limit')
               for name in line.element_names)
try:
    line.aperture.get_pipe_table()
except ValueError as error:
    assert 'Invalid aperture profile' in str(error)
else:
    raise AssertionError('Building with unset collimator gaps must fail')

env.vars.load(here / 'M2_MTN3p8_v5_notilt.str')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=160e9)
line.aperture.get_pipe_table()

mad = Madx(stdout=False)
mad.call(str(here / 'M2_MTN3p8_v6_notilt.seq'))
mad.call(str(here / 'M2_MTN3p8_v5_notilt.str'))
mad.input('beam, particle=proton, pc=160; use, sequence=m2;')
reference = {element.name: element for element in mad.sequence.m2.expanded_elements
             if any(getattr(element, 'aperture', []))}
assert set(reference) == set(line.aperture.pipe_positions.keys())

parameter_counts = dict(circle=1, ellipse=2, rectangle=2, racetrack=4, rectellipse=4)
samples, expected_x, expected_y = [], [], []
table = line.get_table()
for name, element in reference.items():
    params = element.aperture[:parameter_counts[element.apertype]]
    expected = profile_from_madx_aperture(element.apertype, params)
    actual = line.aperture.pipe_positions[name].pipe[0].profile.shape
    assert isinstance(actual, type(expected)), name
    for field, value in expected._to_dict().items():
        np.testing.assert_allclose(getattr(actual, field), value, rtol=0, atol=1e-14)

    # Compare axis intercepts inside each element, including rolled bends.
    half_x = params[0]
    half_y = params[0] if element.apertype == 'circle' else params[1]
    if element.apertype == 'rectellipse':
        half_x, half_y = min(params[0], params[2]), min(params[1], params[3])
    samples.extend(table['s', name] + line[name].length * np.array([1e-4, .5, 1-1e-4]))
    expected_x.extend([half_x] * 3)
    expected_y.extend([half_y] * 3)

extents = line.aperture.get_extents_at_s(samples)
for actual, expected in ((extents.max_x, expected_x), (-extents.min_x, expected_x),
                         (extents.max_y, expected_y), (-extents.min_y, expected_y)):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=5e-6)
print('All 104 installed apertures and sampled axis intercepts agree with MAD-X')

knob = 'kxcmh.x0610727_scr3h_coll1h'
scraper = 'xcmh.x0610727'
center = table['s', scraper] + line[scraper].length / 2
snapshots = []
for gap in (0.010, 0.028, 0.050):
    env[knob] = gap
    line.aperture.get_pipe_table()
    limit = line.aperture.get_limit_elements_at_s([center])[center]
    particles = xt.Particles(p0c=160e9, x=[0, .99*gap, 1.01*gap, -.99*gap, -1.01*gap])
    limit.track(particles)
    order = np.argsort(particles.particle_id)
    np.testing.assert_array_equal(particles.state[order] > 0, [True, True, False, True, False])
    snapshots.append((gap, limit))
    if gap == .028:
        tracking_line = make_tracking_line(line, max_step=5.)
        saved_k1 = line[scraper].k1

# Old limits and optics stay at their own setting after the source knob changes.
for gap, limit in snapshots:
    np.testing.assert_allclose([min(limit.x_vertices), max(limit.x_vertices)],
                               [-gap, gap], rtol=0, atol=1e-10)
assert tracking_line.env[knob] == .028
np.testing.assert_allclose(tracking_line[scraper].k1, saved_k1, rtol=0, atol=1e-14)
np.testing.assert_allclose(tracking_line.get_length(), line.get_length(), rtol=0, atol=1e-10)

limit_table = tracking_line.get_table().rows.match(element_type='LimitPolygon')
pipes = line.aperture.get_pipe_table()
for s in limit_table.s:
    assert np.any((pipes.s_start < s) & (s < pipes.s_end)), s
assert len(limit_table) >= 2 * len(pipes)
assert not any(type(line.element_dict[name]).__name__.startswith('Limit')
               for name in line.element_names)
print('Knob scan, particle survival, independent snapshots and aperture coverage checked')
