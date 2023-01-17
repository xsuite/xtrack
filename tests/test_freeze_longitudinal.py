import json
import pathlib
import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_freeze_longitudinal_explicit(test_context):

    fname_line = test_data_folder / 'lhc_no_bb/line_and_particle.json'

    # import a line and add reference particle
    with open(fname_line) as fid:
        line_dict = json.load(fid)

    print(f"Test {test_context.__class__}")

    line = xt.Line.from_dict(line_dict['line'])
    line.particle_ref = xp.Particles.from_dict(line_dict['particle'])
    # Build the tracker
    tracker = line.build_tracker(_context=test_context)

    # Freeze longitudinal coordinates
    tracker.freeze_longitudinal()

    # Track some particles with frozen longitudinal coordinates
    particles = tracker.build_particles(delta=1e-3, x=[-1e-3, 0, 1e-3])
    tracker.track(particles, num_turns=10)
    particles.move(_context=xo.context_default)
    assert np.allclose(particles.delta, 1e-3, rtol=0, atol=1e-12)
    assert np.allclose(particles.zeta, 0, rtol=9, atol=1e-12)

    # Twiss with frozen longitudinal coordinates (needs to be 4d)
    twiss = tracker.twiss(method='4d')
    assert twiss.slip_factor == 0

    # Unfreeze longitudinal coordinates
    tracker.freeze_longitudinal(False)

    # Track some particles with unfrozen longitudinal coordinates
    particles = tracker.build_particles(delta=1e-3, x=[-1e-3, 0, 1e-3])
    tracker.track(particles, num_turns=10)
    particles.move(_context=xo.context_default)
    assert np.allclose(particles.delta, 0.00099218, rtol=0, atol=1e-6)

    # Twiss with unfrozen longitudinal coordinates (can be 6d)
    twiss = tracker.twiss(method='6d')
    assert np.isclose(twiss.slip_factor, 0.00032151, rtol=0, atol=1e-6)


@for_all_test_contexts
def test_freeze_longitudinal_context_manager(test_context):

    fname_line = test_data_folder / 'lhc_no_bb/line_and_particle.json'

    # import a line and add reference particle
    with open(fname_line) as fid:
        line_dict = json.load(fid)

    print(f"Test {test_context.__class__}")

    line = xt.Line.from_dict(line_dict['line'])
    line.particle_ref = xp.Particles.from_dict(line_dict['particle'])

    # Build the tracker
    tracker = line.build_tracker(_context=test_context)

    with xt.freeze_longitudinal(tracker):
        # Track some particles with frozen longitudinal coordinates
        particles = tracker.build_particles(delta=1e-3, x=[-1e-3, 0, 1e-3])
        tracker.track(particles, num_turns=10)
        particles.move(_context=xo.context_default)
        assert np.allclose(particles.delta, 1e-3, rtol=0, atol=1e-12)
        assert np.allclose(particles.zeta, 0, rtol=9, atol=1e-12)

        # Twiss with frozen longitudinal coordinates (needs to be 4d)
        twiss = tracker.twiss(method='4d')
        assert twiss.slip_factor == 0

    # Track some particles with unfrozen longitudinal coordinates
    particles = tracker.build_particles(delta=1e-3, x=[-1e-3, 0, 1e-3])
    tracker.track(particles, num_turns=10)
    particles.move(_context=xo.context_default)
    assert np.allclose(particles.delta, 0.00099218, rtol=0, atol=1e-6)

    # Twiss with unfrozen longitudinal coordinates (can be 6d)
    twiss = tracker.twiss(method='6d')
    assert np.isclose(twiss.slip_factor, 0.00032151, rtol=0, atol=1e-6)


@for_all_test_contexts
def test_freeze_longitudinal_individual_methods(test_context):

    fname_line = test_data_folder / 'lhc_no_bb/line_and_particle.json'

    # import a line and add reference particle
    with open(fname_line) as fid:
        line_dict = json.load(fid)

    print(f"Test {test_context.__class__}")

    line = xt.Line.from_dict(line_dict['line'])
    line.particle_ref = xp.Particles.from_dict(line_dict['particle'])

    # Build the tracker
    tracker = line.build_tracker(_context=test_context)

    # Track some particles with frozen longitudinal coordinates
    particles = tracker.build_particles(delta=1e-3, x=[-1e-3, 0, 1e-3])
    tracker.track(particles, num_turns=10, freeze_longitudinal=True)
    particles.move(_context=xo.context_default)
    assert np.allclose(particles.delta, 1e-3, rtol=0, atol=1e-12)
    assert np.allclose(particles.zeta, 0, rtol=9, atol=1e-12)

    # Twiss with frozen longitudinal coordinates (needs to be 4d)
    twiss = tracker.twiss(method='4d', freeze_longitudinal=True)
    assert twiss.slip_factor == 0

    # Track some particles with unfrozen longitudinal coordinates
    particles = tracker.build_particles(delta=1e-3, x=[-1e-3, 0, 1e-3])
    tracker.track(particles, num_turns=10)
    particles.move(_context=xo.context_default)
    assert np.allclose(particles.delta, 0.00099218, rtol=0, atol=1e-6)

    # Twiss with unfrozen longitudinal coordinates (can be 6d)
    twiss = tracker.twiss(method='6d')
    assert np.isclose(twiss.slip_factor, 0.00032151, rtol=0, atol=1e-6)