# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts


@for_all_test_contexts
def test_aperture_turn_ele_and_monitor(test_context):
    x_aper_min = -0.1
    x_aper_max = 0.2
    y_aper_min = 0.2
    y_aper_max = 0.3

    part_gen_range = 0.35
    n_part=100

    particles = xp.Particles(_context=test_context,
                             p0c=6500e9,
                             x=np.zeros(n_part),
                             px=np.linspace(-1, 1, n_part),
                             y=np.zeros(n_part),
                             py=np.linspace(-2, 2, n_part),
                            )

    # Build a small test line
    tot_length = 2.
    n_slices = 10000
    line = xt.Line(elements=n_slices*[xt.Drift(
                                            length=tot_length/n_slices)],
                    element_names=['drift{ii}' for ii in range(n_slices)])

    line.build_tracker(_context=test_context)
    n_turns = 3
    line.track(particles, num_turns=n_turns, turn_by_turn_monitor=True)

    part_id = test_context.nparray_from_context_array(particles.particle_id)
    part_state = test_context.nparray_from_context_array(particles.state)
    part_x = test_context.nparray_from_context_array(particles.x)
    part_y = test_context.nparray_from_context_array(particles.y)
    part_px = test_context.nparray_from_context_array(particles.px)
    part_py = test_context.nparray_from_context_array(particles.py)
    part_s = test_context.nparray_from_context_array(particles.s)
    part_at_turn = test_context.nparray_from_context_array(particles.at_turn)
    part_at_element = test_context.nparray_from_context_array(particles.at_element)

    id_alive = part_id[part_state>0]

    #x = px*s
    s_expected = []
    s_tot = tot_length*n_turns
    for ii in range(n_part):
        if np.abs(part_px[ii]) * s_tot > line.global_xy_limit:
            s_expected_x = np.abs(line.global_xy_limit / part_px[ii])
        else:
            s_expected_x = s_tot

        if np.abs(part_py[ii] * s_tot) > line.global_xy_limit:
            s_expected_y = np.abs(line.global_xy_limit / part_py[ii])
        else:
            s_expected_y = s_tot

        if s_expected_x<s_expected_y:
            s_expected.append(s_expected_x)
        else:
            s_expected.append(s_expected_y)

    s_expected = np.array(s_expected)
    at_turn_expected = np.int_(np.clip(np.floor(s_expected/tot_length),
                                       0, n_turns))
    at_element_expected = np.floor((s_expected-tot_length*at_turn_expected)
                                         /(tot_length/n_slices)) + 1
    at_element_expected = np.int_(np.clip(at_element_expected, 0,
                                          n_slices-1))

    assert np.allclose(part_s + at_turn_expected*line.get_length(), s_expected,
                       atol=1e-3)
    assert np.allclose(at_turn_expected, part_at_turn)

    # I need to add a tolerance of one element as a mismatch is visible
    # on a few slices due to rounding
    assert np.allclose(at_element_expected, part_at_element, atol=1.1)

    # Test monitor
    mon = line.record_last_track
    for ii in range(n_part):
        iidd = part_id[ii]
        this_at_turn = part_at_turn[ii]
        this_px = part_px[ii]
        this_py = part_py[ii]

        for tt in range(n_turns):
            if tt<=this_at_turn:
                assert(mon.at_turn[iidd, tt] == tt)
                assert(np.isclose(mon.s[iidd, tt], 0., atol=1e-14))
                assert(np.isclose(mon.x[iidd, tt], tt*tot_length*this_px,
                                  atol=1e-14))
                assert(np.isclose(mon.y[iidd, tt], tt*tot_length*this_py,
                                  atol=1e-14))
                assert(np.isclose(mon.px[iidd, tt], this_px, atol=1e-14))
                assert(np.isclose(mon.py[iidd, tt], this_py, atol=1e-14))
            else:
                assert(mon.at_turn[iidd, tt] == 0)
                assert(mon.s[iidd, tt] == 0)
                assert(mon.x[iidd, tt] == 0)
                assert(mon.y[iidd, tt] == 0)
                assert(mon.px[iidd, tt] == 0)
                assert(mon.py[iidd, tt] == 0)


@for_all_test_contexts
def test_custom_monitor(test_context):
    x_aper_min = -0.1
    x_aper_max = 0.2
    y_aper_min = 0.2
    y_aper_max = 0.3

    part_gen_range = 0.35
    n_part=100

    particles = xp.Particles(_context=test_context,
                             p0c=6500e9,
                             x=np.zeros(n_part),
                             px=np.linspace(-1, 1, n_part),
                             y=np.zeros(n_part),
                             py=np.linspace(-2, 2, n_part),
                            )

    # Build a small test line
    tot_length = 2.
    n_slices = 10000
    line = xt.Line(elements=n_slices*[xt.Drift(
                                            length=tot_length/n_slices)],
                    element_names=['drift{ii}' for ii in range(n_slices)])

    line.build_tracker(_context=test_context)
    n_turns = 6
    start_monitor_at_turn = 2
    part_id_monitor_start = 5
    part_id_monitor_end = n_part - 7

    monitor = xt.ParticlesMonitor(_context=test_context,
            start_at_turn=start_monitor_at_turn,
            stop_at_turn=n_turns,
            particle_id_range=(part_id_monitor_start, part_id_monitor_end))
    for _ in range(n_turns):
        line.track(particles, turn_by_turn_monitor=monitor)

    part_id = test_context.nparray_from_context_array(particles.particle_id)
    part_state = test_context.nparray_from_context_array(particles.state)
    part_x = test_context.nparray_from_context_array(particles.x)
    part_y = test_context.nparray_from_context_array(particles.y)
    part_px = test_context.nparray_from_context_array(particles.px)
    part_py = test_context.nparray_from_context_array(particles.py)
    part_s = test_context.nparray_from_context_array(particles.s)
    part_at_turn = test_context.nparray_from_context_array(particles.at_turn)
    part_at_element = test_context.nparray_from_context_array(particles.at_element)

    id_alive = part_id[part_state>0]

    #x = px*s
    s_expected = []
    s_tot = tot_length*n_turns
    for ii in range(n_part):
        if np.abs(part_px[ii]) * s_tot > line.global_xy_limit:
            s_expected_x = np.abs(line.global_xy_limit / part_px[ii])
        else:
            s_expected_x = s_tot

        if np.abs(part_py[ii] * s_tot) > line.global_xy_limit:
            s_expected_y = np.abs(line.global_xy_limit / part_py[ii])
        else:
            s_expected_y = s_tot

        if s_expected_x<s_expected_y:
            s_expected.append(s_expected_x)
        else:
            s_expected.append(s_expected_y)

    s_expected = np.array(s_expected)
    at_turn_expected = np.int_(np.clip(np.floor(s_expected/tot_length),
                                       0, n_turns))
    at_element_expected = np.floor((s_expected-tot_length*at_turn_expected)
                                         /(tot_length/n_slices)) + 1
    at_element_expected = np.int_(np.clip(at_element_expected, 0,
                                          n_slices-1))

    assert np.allclose(part_s + at_turn_expected * line.line.get_length(),
                       s_expected, atol=1e-3)
    assert np.allclose(at_turn_expected, part_at_turn)

    # I need to add a tolerance of one element as a mismatch is visible
    # on a few slices due to rounding
    assert np.allclose(at_element_expected, part_at_element, atol=1.1)

    # Test monitor
    mon = monitor
    for ii in range(n_part):
        iidd = part_id[ii]
        this_at_turn = part_at_turn[ii]
        this_px = part_px[ii]
        this_py = part_py[ii]

        if iidd < part_id_monitor_start or iidd >= part_id_monitor_end:
            continue

        for tt in range(n_turns):
            if tt < start_monitor_at_turn:
                continue
            if tt<=this_at_turn:
                assert(mon.at_turn[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn] == tt)
                assert(mon.particle_id[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn] == iidd)
                assert(np.isclose(mon.s[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn],
                    0., atol=1e-14))
                assert(np.isclose(mon.x[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn],
                    tt*tot_length*this_px,
                    atol=1e-14))
                assert(np.isclose(mon.y[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn],
                    tt*tot_length*this_py,
                    atol=1e-14))
                assert(np.isclose(mon.px[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn],
                    this_px, atol=1e-14))
                assert(np.isclose(mon.py[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn],
                    this_py, atol=1e-14))
            else:
                assert(mon.at_turn[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn] == 0)
                assert(mon.particle_id[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn] == 0)
                assert(mon.s[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn] == 0)
                assert(mon.x[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn] == 0)
                assert(mon.y[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn] == 0)
                assert(mon.px[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn] == 0)
                assert(mon.py[iidd - part_id_monitor_start,
                    tt - start_monitor_at_turn] == 0)
