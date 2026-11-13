"""
Check that the xsuite python lattice (m2_mtn3p8_v6_notilt.py) is equivalent to
the MAD-X sequence (M2_MTN3p8_v6_notilt.seq): optics and survey at the entry of
all elements must be the same.
"""

import numpy as np
import xtrack as xt

env_py = xt.load('m2_mtn3p8_v6_notilt.py')
env_mad = xt.load('M2_MTN3p8_v6_notilt.seq')

line_py = env_py['m2']
line_mad = env_mad['m2']

for env in [env_py, env_mad]:
    env.vars.load('M2_MTN3p8_v5_notilt.str')

for line in [line_py, line_mad]:
    line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV,
                                     p0c=line.env['beam_momentum'] * 1e9)

# Elements to be compared (all elements of the python lattice except drifts)
tt_py = line_py.get_table()
names = tt_py.rows.match_not(element_type='Drift').rows.match_not(name='_end_point').name
assert len(names) == 184  # number of elements placed in the sequence
for nn in names:
    assert nn in line_mad.element_names, f'{nn} not found in MAD-X lattice'

# Optics
tw_py = line_py.twiss(betx=1, bety=1)
tw_mad = line_mad.twiss(betx=1, bety=1)

tw_py = tw_py.rows[names]
tw_mad = tw_mad.rows[names]
assert np.all(tw_py.name == tw_mad.name)

for col in ['s', 'betx', 'bety', 'alfx', 'alfy', 'mux', 'muy',
            'x', 'px', 'y', 'py', 'dx', 'dpx', 'dy', 'dpy']:
    vp = getattr(tw_py, col)
    vm = getattr(tw_mad, col)
    assert np.allclose(vp, vm, rtol=1e-10, atol=1e-12), (
        f'twiss {col}: max difference {np.max(np.abs(vp - vm))}')
print('Optics identical at the entry of all elements')

# Survey (T09 survey parameters)
X0, Y0, Z0 = -677.24488, 2441.56985, 4605.8619
theta0, phi0, psi0 = 1.406046 - np.pi / 2, -0.000358, 0

sv_py = line_py.survey(X0=X0, Y0=Y0, Z0=Z0, theta0=theta0, phi0=phi0, psi0=psi0)
sv_mad = line_mad.survey(X0=X0, Y0=Y0, Z0=Z0, theta0=theta0, phi0=phi0, psi0=psi0)

sv_py = sv_py.rows[names]
sv_mad = sv_mad.rows[names]
assert np.all(sv_py.name == sv_mad.name)

for col in ['s', 'X', 'Y', 'Z', 'theta', 'phi', 'psi']:
    vp = getattr(sv_py, col)
    vm = getattr(sv_mad, col)
    assert np.allclose(vp, vm, rtol=1e-10, atol=1e-12), (
        f'survey {col}: max difference {np.max(np.abs(vp - vm))}')
print('Survey identical at the entry of all elements')
