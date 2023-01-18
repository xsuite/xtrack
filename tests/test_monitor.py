# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import pathlib

import numpy as np

import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts


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


@for_all_test_contexts
def test_monitor(test_context):
    tracker = line.copy().build_tracker(_context=test_context)
    particles = particles0.copy(_context=test_context)

    # Test implicit monitor
    num_turns = 30
    tracker.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)
    mon = tracker.record_last_track
    assert np.all(mon.x.shape == np.array([50, 30]))
    assert np.all(mon.at_turn[3, :] == np.arange(0, num_turns))
    assert np.all(mon.particle_id[:, 3] == np.arange(0, num_particles))
    assert np.all(mon.at_element[:, :] == 0)
    assert np.all(mon.pzeta[:, 0] == particles0.pzeta)

    # Test explicit monitor passed to track
    monitor = xt.ParticlesMonitor(_context=test_context,
                                  start_at_turn=5, stop_at_turn=15,
                                  num_particles=num_particles)
    particles = particles0.copy(_context=test_context)
    num_turns = 30
    tracker.track(particles, num_turns=num_turns, turn_by_turn_monitor=monitor)
    assert np.all(monitor.x.shape == np.array([50, 10]))
    assert np.all(monitor.at_turn[3, :] == np.arange(5, 15))
    assert np.all(monitor.particle_id[:, 3] == np.arange(0, num_particles))
    assert np.all(monitor.at_element[:, :] == 0)
    assert np.all(monitor.pzeta[:, 0] == mon.pzeta[:, 5]) #5 in mon because the 0th entry of monitor is the 5th turn (5th entry in mon)


    # Test explicit monitor used in in stand-alone mode
    mon2 = xt.ParticlesMonitor(_context=test_context,
                               start_at_turn=5, stop_at_turn=15,
                               num_particles=num_particles)
    particles = particles0.copy(_context=test_context)
    num_turns = 30
    for ii in range(num_turns):
        mon2.track(particles)
        tracker.track(particles)
    assert np.all(mon2.x.shape == np.array([50, 10]))
    assert np.all(mon2.at_turn[3, :] == np.arange(5, 15))
    assert np.all(mon2.particle_id[:, 3] == np.arange(0, num_particles))
    assert np.all(mon2.at_element[:, :] == 0)
    assert np.all(mon2.pzeta[:, 0] == mon.pzeta[:, 5]) #5 in mon because the 0th entry of monitor is the 5th turn (5th entry in mon)

    # Test monitors with multiple frames
    monitor_multiframe = xt.ParticlesMonitor(_context=test_context,
                                             start_at_turn=5, stop_at_turn=10,
                                             n_repetitions=3,
                                             repetition_period=20,
                                             num_particles=num_particles)
    particles = particles0.copy(_context=test_context)
    num_turns = 100
    tracker.track(particles,
                  num_turns=num_turns,
                  turn_by_turn_monitor=monitor_multiframe)
    assert np.all(monitor_multiframe.x.shape == np.array([3, 50, 5]))
    assert np.all(monitor_multiframe.at_turn[1, 3, :] == np.arange(25, 30))
    assert np.all(monitor_multiframe.particle_id[2, :, 3] == np.arange(0,
                                                             num_particles))
    assert np.all(monitor_multiframe.at_element[:, :, :] == 0)
    assert np.all(monitor_multiframe.pzeta[0, :, 0] == mon.pzeta[:, 5]) #5 in mon because the 0th entry of monitor is the 5th turn (5th entry in mon)

    # Test monitors installed in a line
    monitor_ip5 = xt.ParticlesMonitor(start_at_turn=5, stop_at_turn=15,
                                      num_particles=num_particles)
    monitor_ip8 = xt.ParticlesMonitor(start_at_turn=5, stop_at_turn=15,
                                      num_particles=num_particles)
    line_with_monitor = line.copy()
    line_with_monitor.insert_element(index='ip5', element=monitor_ip5, name='mymon5')
    line_with_monitor.insert_element(index='ip8', element=monitor_ip8, name='mymon8')

    tracker_w_monitor = line_with_monitor.build_tracker(_context=test_context)

    particles = particles0.copy(_context=test_context)
    num_turns = 50
    tracker_w_monitor.track(particles, num_turns=num_turns)

    assert np.all(monitor_ip5.x.shape == np.array([50, 10]))
    assert np.all(monitor_ip5.at_turn[3, :] == np.arange(5, 15))
    assert np.all(monitor_ip5.particle_id[:, 3] == np.arange(0, num_particles))
    assert np.all(monitor_ip5.at_element[:, :]
                        == line_with_monitor.element_names.index('ip5') - 1)

    assert np.all(monitor_ip8.x.shape == np.array([50, 10]))
    assert np.all(monitor_ip8.at_turn[3, :] == np.arange(5, 15))
    assert np.all(monitor_ip8.particle_id[:, 3] == np.arange(0, num_particles))
    assert np.all(monitor_ip8.at_element[:, :]
                        == line_with_monitor.element_names.index('ip8') - 1)
