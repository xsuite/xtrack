import json
import pathlib

import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo

#context = xo.ContextPyopencl()
context = xo.ContextCpu()

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

with open(test_data_folder.joinpath(
        'hllhc15_noerrors_nobb/line_and_particle.json')) as f:
    dct = json.load(f)
line = xt.Line.from_dict(dct['line'])
line.particle_ref = xp.Particles.from_dict(dct['particle'])

tracker0 = line.build_tracker()

num_particles = 50
particles0 = xp.generate_matched_gaussian_bunch(tracker=tracker0,
                                               num_particles=num_particles,
                                               nemitt_x=2.5e-6,
                                               nemitt_y=2.5e-6,
                                               sigma_z=9e-2)

def test_monitor():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        tracker = line.copy().build_tracker(_context=context)
        particles = particles0.copy(_context=context)

        # Test implicit monitor
        num_turns = 30
        tracker.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)
        mon = tracker.record_last_track
        assert np.all(mon.x.shape == np.array([50, 30]))
        assert np.all(mon.at_turn[3, :] == np.arange(0, num_turns))
        assert np.all(mon.particle_id[:, 3] == np.arange(0, num_particles))

        # Test explicit monitor passed to track
        monitor = xt.ParticlesMonitor(_context=context,
                                    start_at_turn=5, stop_at_turn=15,
                                    num_particles=num_particles)
        particles = particles0.copy(_context=context)
        num_turns = 30
        tracker.track(particles, num_turns=num_turns, turn_by_turn_monitor=monitor)
        assert np.all(monitor.x.shape == np.array([50, 10]))
        assert np.all(monitor.at_turn[3, :] == np.arange(5, 15))
        assert np.all(monitor.particle_id[:, 3] == np.arange(0, num_particles))

        # Test explicit monitor used in in stand-alone mode
        mon2 = xt.ParticlesMonitor(_context=context,
                                    start_at_turn=5, stop_at_turn=15,
                                    num_particles=num_particles)
        particles = particles0.copy(_context=context)
        num_turns = 30
        for ii in range(num_turns):
            mon2.track(particles)
            tracker.track(particles)
        assert np.all(mon2.x.shape == np.array([50, 10]))
        assert np.all(mon2.at_turn[3, :] == np.arange(5, 15))
        assert np.all(mon2.particle_id[:, 3] == np.arange(0, num_particles))

        # Test monitors with multiple frames
        monitor_multiframe = xt.ParticlesMonitor(_context=context,
                                            start_at_turn=5, stop_at_turn=10,
                                            n_repetitions=3,
                                            repetition_period=20,
                                            num_particles=num_particles)
        particles = particles0.copy(_context=context)
        num_turns = 100
        tracker.track(particles, num_turns=num_turns,
                    turn_by_turn_monitor=monitor_multiframe)
        assert np.all(monitor_multiframe.x.shape == np.array([3, 50, 5]))
        assert np.all(monitor_multiframe.at_turn[1, 3, :] == np.arange(25, 30))
        assert np.all(monitor_multiframe.particle_id[2, :, 3] == np.arange(0,
                                                                 num_particles))