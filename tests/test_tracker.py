# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #
import json

import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp

from pathlib import Path


def test_ebe_monitor():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        line = xt.Line(elements=[xt.Multipole(knl=[0, 1.]),
                                xt.Drift(length=0.5),
                                xt.Multipole(knl=[0, -1]),
                                xt.Cavity(frequency=400e7, voltage=6e6),
                                xt.Drift(length=.5),
                                xt.Drift(length=0)])

        tracker = line.build_tracker(_context=context)

        particles = xp.Particles(x=[1e-3, -2e-3, 5e-3], y=[2e-3, -4e-3, 3e-3],
                                zeta=1e-2, p0c=7e12, mass0=xp.PROTON_MASS_EV,
                                _context=context)

        tracker.track(particles.copy(), turn_by_turn_monitor='ONE_TURN_EBE')

        mon = tracker.record_last_track

        for ii, ee in enumerate(line.elements):
            for tt, nn in particles._structure['per_particle_vars']:
                assert np.all(particles.to_dict()[nn] == getattr(mon, nn)[:, ii])
            ee.track(particles)
            particles.at_element += 1

def test_cycle():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        d0 = xt.Drift()
        c0 = xt.Cavity()
        d1 = xt.Drift()
        r0 = xt.SRotation()


        for collective in [True, False]:
            line = xt.Line(elements=[d0, c0, d1, r0])
            d1.iscollective = collective

            tracker = xt.Tracker(line=line, _context=context)

            ctracker_name = tracker.cycle(name_first_element='e2')
            ctracker_index = tracker.cycle(index_first_element=2)

            for ctracker in [ctracker_index, ctracker_name]:
                assert ctracker.line.element_names[0] == 'e2'
                assert ctracker.line.element_names[1] == 'e3'
                assert ctracker.line.element_names[2] == 'e0'
                assert ctracker.line.element_names[3] == 'e1'

                assert ctracker.line.elements[0] is d1
                assert ctracker.line.elements[1] is r0
                assert ctracker.line.elements[2] is d0
                assert ctracker.line.elements[3] is c0

def test_synrad_configuration():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")
        for collective in [False, True]:
            elements = [xt.Multipole(knl=[1]) for _ in range(10)]
            if collective:
                elements[5].iscollective = True
                elements[5].move(_context=context)

            tracker = xt.Tracker(line=xt.Line(elements=elements),
                                 _context=context)

            tracker.configure_radiation(mode='mean')
            for ee in tracker.line.elements:
                assert ee.radiation_flag == 1
            p = xp.Particles(x=[0.01, 0.02], _context=context)
            tracker.track(p)
            p.move(_context=xo.ContextCpu())
            assert np.all(p._rng_s1 + p._rng_s2 + p._rng_s3 + p._rng_s4 == 0)

            tracker.configure_radiation(mode='quantum')
            for ee in tracker.line.elements:
                assert ee.radiation_flag == 2
            p = xp.Particles(x=[0.01, 0.02], _context=context)
            tracker.track(p)
            p.move(_context=xo.ContextCpu())
            assert np.all(p._rng_s1 + p._rng_s2 + p._rng_s3 + p._rng_s4 > 0)

            tracker.configure_radiation(mode=None)
            for ee in tracker.line.elements:
                assert ee.radiation_flag == 0
            p = xp.Particles(x=[0.01, 0.02], _context=context)
            tracker.track(p)
            p.move(_context=xo.ContextCpu())
            assert np.all(p._rng_s1 + p._rng_s2 + p._rng_s3 + p._rng_s4 == 0)

def test_partial_tracking():
    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        n_elem = 9
        elements = [ xt.Drift(length=1.) for _ in range(n_elem) ]
        line = xt.Line(elements=elements)
        tracker = line.build_tracker(_context=context)
        assert not tracker.iscollective
        particles_init = xp.Particles(_context=context,
            x=[1e-3, -2e-3, 5e-3], y=[2e-3, -4e-3, 3e-3],
            zeta=1e-2, p0c=7e12, mass0=xp.PROTON_MASS_EV,
            at_turn=0, at_element=0)

        _default_track(tracker, particles_init)
        _ele_start_until_end(tracker, particles_init)
        _ele_start_with_shift(tracker, particles_init)
        _ele_start_with_shift_more_turns(tracker, particles_init)
        _ele_stop_from_start(tracker, particles_init)
        _ele_start_to_ele_stop(tracker, particles_init)
        _ele_start_to_ele_stop_with_overflow(tracker, particles_init)

def test_partial_tracking_with_collective():
     for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        n_elem = 9
        elements = [xt.Drift(length=1., _context=context) for _ in range(n_elem)]
        # Make some elements collective
        elements[3].iscollective = True
        elements[7].iscollective = True
        line = xt.Line(elements=elements)
        tracker = line.build_tracker(_context=context)
        assert tracker.iscollective
        assert len(tracker._parts) == 5
        particles_init = xp.Particles(
                _context=context,
                x=[1e-3, -2e-3, 5e-3], y=[2e-3, -4e-3, 3e-3],
                zeta=1e-2, p0c=7e12, mass0=xp.PROTON_MASS_EV,
                at_turn=0, at_element=0)

        _default_track(tracker, particles_init)
        _ele_start_until_end(tracker, particles_init)
        _ele_start_with_shift(tracker, particles_init)
        _ele_start_with_shift_more_turns(tracker, particles_init)
        _ele_stop_from_start(tracker, particles_init)
        _ele_start_to_ele_stop(tracker, particles_init)
        _ele_start_to_ele_stop_with_overflow(tracker, particles_init)


# Track from the start until the end of the first, second, and tenth turn
def _default_track(tracker, particles_init):
    n_elem = len(tracker.line.element_names)
    for turns in [1, 2, 10]:
        expected_end_turn = turns
        expected_end_element = 0
        expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

        particles = particles_init.copy()
        tracker.track(particles, num_turns=turns, turn_by_turn_monitor=True)
        check, end_turn, end_element, end_s = _get_at_turn_element(particles)
        assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                    and end_s==expected_end_element)
        assert tracker.record_last_track.x.shape == (len(particles.x), expected_num_monitor)

# Track, from any ele_start, until the end of the first, second, and tenth turn
def _ele_start_until_end(tracker, particles_init):
    n_elem = len(tracker.line.element_names)
    for turns in [1, 2, 10]:
        for start in range(n_elem):
            expected_end_turn = turns
            expected_end_element = 0
            expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

            particles = particles_init.copy()
            particles.at_element = start
            particles.s = start
            tracker.track(particles, num_turns=turns, ele_start=start, turn_by_turn_monitor=True)
            check, end_turn, end_element, end_s = _get_at_turn_element(particles)
            assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                        and end_s==expected_end_element)
            assert tracker.record_last_track.x.shape==(len(particles.x), expected_num_monitor)

# Track, from any ele_start, any shifts that stay within the first turn
def _ele_start_with_shift(tracker, particles_init):
    n_elem = len(tracker.line.element_names)
    for start in range(n_elem):
        for shift in range(1,n_elem-start):
            expected_end_turn = 0
            expected_end_element = start+shift
            expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

            particles = particles_init.copy()
            particles.at_element = start
            particles.s = start
            tracker.track(particles, ele_start=start, num_elements=shift, turn_by_turn_monitor=True)
            check, end_turn, end_element, end_s = _get_at_turn_element(particles)
            assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                        and end_s==expected_end_element)
            assert tracker.record_last_track.x.shape==(len(particles.x), expected_num_monitor)

# Track, from any ele_start, any shifts that are larger than one turn (up to 3 turns)
def _ele_start_with_shift_more_turns(tracker, particles_init):
    n_elem = len(tracker.line.element_names)
    for start in range(n_elem):
        for shift in range(n_elem-start, 3*n_elem+1):
            expected_end_turn = round(np.floor( (start+shift)/n_elem ))
            expected_end_element = start + shift - n_elem*expected_end_turn
            expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

            particles = particles_init.copy()
            particles.at_element = start
            particles.s = start
            tracker.track(particles, ele_start=start, num_elements=shift, turn_by_turn_monitor=True)
            check, end_turn, end_element, end_s = _get_at_turn_element(particles)
            assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                        and end_s==expected_end_element)
            assert tracker.record_last_track.x.shape==(len(particles.x), expected_num_monitor)

# Track from the start until any ele_stop in the first, second, and tenth turn
def _ele_stop_from_start(tracker, particles_init):
    n_elem = len(tracker.line.element_names)
    for turns in [1, 2, 10]:
        for stop in range(1, n_elem):
            expected_end_turn = turns-1
            expected_end_element = stop
            expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

            particles = particles_init.copy()
            tracker.track(particles, num_turns=turns, ele_stop=stop, turn_by_turn_monitor=True)
            check, end_turn, end_element, end_s = _get_at_turn_element(particles)
            assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                        and end_s==expected_end_element)
            assert tracker.record_last_track.x.shape==(len(particles.x), expected_num_monitor)

# Track from any ele_start until any ele_stop that is larger than ele_start
# for one, two, and ten turns
def _ele_start_to_ele_stop(tracker, particles_init):
    n_elem = len(tracker.line.element_names)
    for turns in [1, 2, 10]:
        for start in range(n_elem):
            for stop in range(start+1,n_elem):
                expected_end_turn = turns-1
                expected_end_element = stop
                expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

                particles = particles_init.copy()
                particles.at_element = start
                particles.s = start
                tracker.track(particles, num_turns=turns, ele_start=start, ele_stop=stop, turn_by_turn_monitor=True)
                check, end_turn, end_element, end_s = _get_at_turn_element(particles)
                assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                            and end_s==expected_end_element)
                assert tracker.record_last_track.x.shape==(len(particles.x), expected_num_monitor)

# Track from any ele_start until any ele_stop that is smaller than or equal to ele_start (turn increses by one)
# for one, two, and ten turns
def _ele_start_to_ele_stop_with_overflow(tracker, particles_init):
    n_elem = len(tracker.line.element_names)
    for turns in [1, 2, 10]:
        for start in range(n_elem):
            for stop in range(start+1):
                expected_end_turn = turns
                expected_end_element = stop
                expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

                particles = particles_init.copy()
                particles.at_element = start
                particles.s = start
                tracker.track(particles, num_turns=turns, ele_start=start, ele_stop=stop, turn_by_turn_monitor=True)
                check, end_turn, end_element, end_s = _get_at_turn_element(particles)
                assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                            and end_s==expected_end_element)
                assert tracker.record_last_track.x.shape==(len(particles.x), expected_num_monitor)


# Quick helper function to:
#   1) check that all survived particles are at the same element and turn
#   2) return that element and turn
def _get_at_turn_element(particles):
    part_cpu = particles.copy(_context=xo.ContextCpu())
    at_element = np.unique(part_cpu.at_element[part_cpu.state>0])
    at_turn = np.unique(part_cpu.at_turn[part_cpu.state>0])
    at_s = np.unique(part_cpu.s[part_cpu.state>0])
    all_together = len(at_turn)==1 and len(at_element)==1 and len(at_s)==1
    return all_together, at_turn[0], at_element[0], at_s[0]


def test_tracker_binary_serialization(tmp_path):
    tmp_file = tmp_path / 'test_tracker_binary_serialization.npy'
    file_path = tmp_file.resolve()

    line = xt.Line(
        elements={
            'mn': xt.Multipole(knl=[1, 2]),
            'ms': xt.Multipole(ksl=[3]),
            'd': xt.Drift(length=4),
        },
        element_names=['mn', 'd', 'ms', 'd', 'mn'],
    )

    tracker = line.build_tracker(_context=xo.context_default)

    tracker.to_binary_file(file_path)
    new_tracker = xt.Tracker.from_binary_file(file_path)

    assert new_tracker._buffer is not tracker._buffer

    new_line = new_tracker.line

    assert line.element_names == new_line.element_names

    assert [elem.__class__.__name__ for elem in line.elements] == \
           ['Multipole', 'Drift', 'Multipole', 'Drift', 'Multipole']
    assert line.elements[0]._xobject._offset == \
           new_line.elements[4]._xobject._offset
    assert line.elements[1]._xobject._offset == \
           new_line.elements[3]._xobject._offset

    assert len(set(elem._xobject._buffer for elem in new_line.elements)) == 1

    assert (new_line.elements[0].knl == [1, 2]).all()
    assert new_line.elements[1].length == 4
    assert (new_line.elements[2].ksl == [3]).all()


def test_tracker_binary_serialisation_with_knobs(tmp_path):
    tmp_file = tmp_path / 'test_tracker_binary_serialization.npy'
    tmp_file_path = tmp_file.resolve()

    line_with_knobs_path = (Path(__file__).parent /
                            '../test_data/hllhc15_noerrors_nobb' /
                            'line_w_knobs_and_particle.json')
    with open(line_with_knobs_path.resolve(), 'r') as line_file:
        line_dict = json.load(line_file)
    line_with_knobs = xt.Line.from_dict(line_dict['line'])

    tracker = line_with_knobs.build_tracker(_context=xo.context_default)
    tracker.to_binary_file(tmp_file_path)
    new_tracker = xt.Tracker.from_binary_file(tmp_file_path)
    new_tracker.line.particle_ref = xp.Particles(
        mass0=xp.PROTON_MASS_EV, q0=1, gamma0=line_dict['particle']['gamma0'])

    assert tracker.line._var_management.keys() == new_tracker.line._var_management.keys()

    new_tracker.vars['on_x1'] = 250
    assert np.isclose(new_tracker.twiss(at_elements=['ip1'])['px'][0], 250e-6,
                      atol=1e-6, rtol=0)
    new_tracker.vars['on_x1'] = -300
    assert np.isclose(new_tracker.twiss(at_elements=['ip1'])['px'][0], -300e-6,
                      atol=1e-6, rtol=0)

    new_tracker.vars['on_x5'] = 130
    assert np.isclose(new_tracker.twiss(at_elements=['ip5'])['py'][0], 130e-6,
                      atol=1e-6, rtol=0)
    new_tracker.vars['on_x5'] = -270
    assert np.isclose(new_tracker.twiss(at_elements=['ip5'])['py'][0], -270e-6,
                      atol=1e-6, rtol=0)
