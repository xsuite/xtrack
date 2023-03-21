# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import pathlib

import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo
from xobjects.test_helpers import for_all_test_contexts


test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

with open(test_data_folder.joinpath(
        'hllhc15_noerrors_nobb/line_and_particle.json')) as f:
    dct = json.load(f)
line0 = xt.Line.from_dict(dct['line'])
line0.particle_ref = xp.Particles.from_dict(dct['particle'])

line0.build_tracker()

num_particles = 50
particles0 = xp.generate_matched_gaussian_bunch(line=line0,
                                               num_particles=num_particles,
                                               nemitt_x=2.5e-6,
                                               nemitt_y=2.5e-6,
                                               sigma_z=9e-2)


@for_all_test_contexts
def test_monitor(test_context):
    line = line0.copy()
    line.build_tracker(_context=test_context)
    particles = particles0.copy(_context=test_context)

    # Test implicit monitor
    num_turns = 30
    line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)
    mon = line.record_last_track
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
    line.track(particles, num_turns=num_turns, turn_by_turn_monitor=monitor)
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
        line.track(particles)
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
    line.track(particles,
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
    line_w_monitor = line0.copy()
    line_w_monitor.insert_element(index='ip5', element=monitor_ip5, name='mymon5')
    line_w_monitor.insert_element(index='ip8', element=monitor_ip8, name='mymon8')

    line_w_monitor.build_tracker(_context=test_context)

    particles = particles0.copy(_context=test_context)
    num_turns = 50
    line_w_monitor.track(particles, num_turns=num_turns)

    assert np.all(monitor_ip5.x.shape == np.array([50, 10]))
    assert np.all(monitor_ip5.at_turn[3, :] == np.arange(5, 15))
    assert np.all(monitor_ip5.particle_id[:, 3] == np.arange(0, num_particles))
    assert np.all(monitor_ip5.at_element[:, :]
                        == line_w_monitor.element_names.index('ip5') - 1)

    assert np.all(monitor_ip8.x.shape == np.array([50, 10]))
    assert np.all(monitor_ip8.at_turn[3, :] == np.arange(5, 15))
    assert np.all(monitor_ip8.particle_id[:, 3] == np.arange(0, num_particles))
    assert np.all(monitor_ip8.at_element[:, :]
                        == line_w_monitor.element_names.index('ip8') - 1)



@for_all_test_contexts
def test_last_turns_monitor(test_context):

    particles = xp.Particles(p0c=6.5e12, x=[1,2,3,4,5,6], _context=test_context)
    num_particles = len(particles.x)

    monitor = xt.LastTurnsMonitor(n_last_turns=5, particle_id_range=(1, 5), _context=test_context)

    line = xt.Line([monitor])
    line.build_tracker(_context=test_context)

    for turn in range(10):

        line.track(particles, num_turns=1)

        # Note that indicees are re-ordered upon particle loss on CPU contexts,
        # so sort before manipulation
        if isinstance(test_context, xo.ContextCpu):
            particles.sort(interleave_lost_particles=True)

        particles.x[0] += 1
        particles.x[1] -= 1
        particles.x[2] += 2
        particles.x[3] -= 2
        particles.x[4] += 3
        particles.x[5] -= 3
        if turn == 2:
            particles.state[1] = 0
        if turn == 4:
            particles.state[2] = 0
        if turn == 6:
            particles.state[3] = 0

        if isinstance(test_context, xo.ContextCpu):
            particles.reorganize()


    assert np.all(monitor.particle_id == np.array([[0,0,1,1,1],[2]*5,[3]*5,[4]*5]))
    assert np.all(monitor.at_turn == np.array([np.clip(n-np.arange(4,-1,-1),0,None) for n in (2,4,6,9)]))
    assert np.all(monitor.x == np.array([[0,0,2,1,0],[3,5,7,9,11],[0,-2,-4,-6,-8],[20,23,26,29,32]]))


