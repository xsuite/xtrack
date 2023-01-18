# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #
import json
import pathlib

import numpy as np
import xobjects as xo
import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts

from pathlib import Path

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_ebe_monitor(test_context):
    line = xt.Line(elements=[xt.Multipole(knl=[0, 1.]),
                            xt.Drift(length=0.5),
                            xt.Multipole(knl=[0, -1]),
                            xt.Cavity(frequency=400e7, voltage=6e6),
                            xt.Drift(length=.5),
                            xt.Drift(length=0)])

    tracker = line.build_tracker(_context=test_context)

    particles = xp.Particles(x=[1e-3, -2e-3, 5e-3], y=[2e-3, -4e-3, 3e-3],
                            zeta=1e-2, p0c=7e12, mass0=xp.PROTON_MASS_EV,
                            _context=test_context)

    tracker.track(particles.copy(), turn_by_turn_monitor='ONE_TURN_EBE')

    mon = tracker.record_last_track

    for ii, ee in enumerate(line.elements):
        for tt, nn in particles._structure['per_particle_vars']:
            assert np.all(particles.to_dict()[nn] == getattr(mon, nn)[:, ii])
        ee.track(particles)
        particles.at_element += 1


@for_all_test_contexts
def test_cycle(test_context):
    d0 = xt.Drift()
    c0 = xt.Cavity()
    d1 = xt.Drift()
    r0 = xt.SRotation()
    particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, gamma0=1.05)

    for collective in [True, False]:
        line = xt.Line(elements=[d0, c0, d1, r0])
        d1.iscollective = collective

        tracker = xt.Tracker(line=line, _context=test_context)
        tracker.particle_ref = particle_ref

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

            assert ctracker.particle_ref.mass0 == xp.PROTON_MASS_EV
            assert ctracker.particle_ref.gamma0 == 1.05


@for_all_test_contexts
def test_synrad_configuration(test_context):
    for collective in [False, True]:
        elements = [xt.Multipole(knl=[1]) for _ in range(10)]
        if collective:
            elements[5].iscollective = True
            elements[5].move(_context=test_context)

        tracker = xt.Tracker(line=xt.Line(elements=elements),
                             _context=test_context)

        tracker.configure_radiation(model='mean')
        for ee in tracker.line.elements:
            assert ee.radiation_flag == 1
        p = xp.Particles(x=[0.01, 0.02], _context=test_context)
        tracker.track(p)
        p.move(_context=xo.ContextCpu())
        assert np.all(p._rng_s1 + p._rng_s2 + p._rng_s3 + p._rng_s4 == 0)

        tracker.configure_radiation(model='quantum')
        for ee in tracker.line.elements:
            assert ee.radiation_flag == 2
        p = xp.Particles(x=[0.01, 0.02], _context=test_context)
        tracker.track(p)
        p.move(_context=xo.ContextCpu())
        assert np.all(p._rng_s1 + p._rng_s2 + p._rng_s3 + p._rng_s4 > 0)

        tracker.configure_radiation(model=None)
        for ee in tracker.line.elements:
            assert ee.radiation_flag == 0
        p = xp.Particles(x=[0.01, 0.02], _context=test_context)
        tracker.track(p)
        p.move(_context=xo.ContextCpu())
        assert np.all(p._rng_s1 + p._rng_s2 + p._rng_s3 + p._rng_s4 > 0)


@for_all_test_contexts
def test_partial_tracking(test_context):
    n_elem = 9
    elements = [ xt.Drift(length=1.) for _ in range(n_elem) ]
    line = xt.Line(elements=elements)
    tracker = line.build_tracker(_context=test_context)
    assert not tracker.iscollective
    particles_init = xp.Particles(_context=test_context,
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


@for_all_test_contexts
def test_partial_tracking_with_collective(test_context):
    n_elem = 9
    elements = [xt.Drift(length=1., _context=test_context) for _ in range(n_elem)]
    # Make some elements collective
    elements[3].iscollective = True
    elements[7].iscollective = True
    line = xt.Line(elements=elements)
    tracker = line.build_tracker(_context=test_context)
    assert tracker.iscollective
    assert len(tracker._parts) == 5
    particles_init = xp.Particles(
            _context=test_context,
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
    tracker.particle_ref = xp.Particles.from_dict(line_dict['particle'])
    tracker.to_binary_file(tmp_file_path)
    new_tracker = xt.Tracker.from_binary_file(tmp_file_path)

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


def test_tracker_hashable_config():
    tracker = xt.Tracker(line=xt.Line([]))
    tracker.config.TEST_FLAG_BOOL = True
    tracker.config.TEST_FLAG_INT = 42
    tracker.config.TEST_FLAG_FALSE = False
    tracker.config.ZZZ = 'lorem'
    tracker.config.AAA = 'ipsum'

    expected = (
        ('AAA', 'ipsum'),
        ('TEST_FLAG_BOOL', True),
        ('TEST_FLAG_INT', 42),
        ('XFIELDS_BB3D_NO_BEAMSTR', True), # active by default
        ('XTRACK_MULTIPOLE_NO_SYNRAD', True), # active by default
        ('ZZZ', 'lorem'),
    )

    assert tracker._hashable_config() == expected


def test_tracker_config_to_headers():
    tracker = xt.Tracker(line=xt.Line([]))

    tracker.config.clear()
    tracker.config.TEST_FLAG_BOOL = True
    tracker.config.TEST_FLAG_INT = 42
    tracker.config.TEST_FLAG_FALSE = False
    tracker.config.ZZZ = 'lorem'
    tracker.config.AAA = 'ipsum'

    expected = [
        '#define TEST_FLAG_BOOL',
        '#define TEST_FLAG_INT 42',
        '#define ZZZ lorem',
        '#define AAA ipsum',
    ]

    assert set(tracker._config_to_headers()) == set(expected)


@for_all_test_contexts
def test_tracker_config(test_context):
    class TestElement(xt.BeamElement):
        _xofields = {
            'dummy': xo.Float64,
        }
        _extra_c_sources = ["""
            /*gpufun*/
            void TestElement_track_local_particle(
                    TestElementData el,
                    LocalParticle* part0)
            {
                //start_per_particle_block (part0->part)

                    #if TEST_FLAG == 2
                    LocalParticle_set_x(part, 7);
                    #endif

                    #ifdef TEST_FLAG_BOOL
                    LocalParticle_set_y(part, 42);
                    #endif

                //end_per_particle_block
            }
            """]

    test_element = TestElement(_context=test_context)
    line = xt.Line([test_element])
    tracker = xt.Tracker(line=line)

    particles = xp.Particles(p0c=1e9, x=[0], y=[0], _context=test_context)

    p = particles.copy()
    tracker.config.TEST_FLAG = 2
    tracker.track(p)
    assert p.x[0] == 7.0
    assert p.y[0] == 0.0
    first_kernel = tracker._current_track_kernel

    p = particles.copy()
    tracker.config.TEST_FLAG = False
    tracker.config.TEST_FLAG_BOOL = True
    tracker.track(p)
    assert p.x[0] == 0.0
    assert p.y[0] == 42.0
    assert tracker._current_track_kernel is not first_kernel

    tracker.config.TEST_FLAG = 2
    tracker.config.TEST_FLAG_BOOL = False
    assert len(tracker.track_kernel) == 3 # As tracker.track_kernel.keys() =
                                          # dict_keys([(), (('TEST_FLAG', 2),), (('TEST_FLAG_BOOL', True),)])
    assert tracker._current_track_kernel is first_kernel


@for_all_test_contexts
def test_optimize_for_tracking(test_context):
    fname_line_particles = test_data_folder / 'hllhc15_noerrors_nobb/line_and_particle.json'

    with open(fname_line_particles, 'r') as fid:
        input_data = json.load(fid)

    line = xt.Line.from_dict(input_data['line'])
    line.particle_ref = xp.Particles.from_dict(input_data['particle'])

    tracker = line.build_tracker(_context=test_context)

    particles = tracker.build_particles(
        x_norm=np.linspace(-2, 2, 1000), y_norm=0.1, delta=3e-4,
        nemitt_x=2.5e-6, nemitt_y=2.5e-6)

    p_no_optimized = particles.copy()
    p_optimized = particles.copy()

    num_turns = 10

    tracker.track(p_no_optimized, num_turns=num_turns, time=True)
    df_before_optimize = tracker.line.to_pandas()
    n_markers_before_optimize = (df_before_optimize.element_type == 'Marker').sum()
    assert n_markers_before_optimize > 4 # There are at least the IPs

    tracker.optimize_for_tracking(keep_markers=True)
    df_optimize_keep_markers = tracker.line.to_pandas()
    n_markers_optimize_keep = (df_optimize_keep_markers.element_type == 'Marker').sum()
    assert n_markers_optimize_keep == n_markers_before_optimize

    tracker.optimize_for_tracking(keep_markers=['ip1', 'ip5'])
    df_optimize_ip15 = tracker.line.to_pandas()
    n_markers_optimize_ip15 = (df_optimize_ip15.element_type == 'Marker').sum()
    assert n_markers_optimize_ip15 == 2

    tracker.optimize_for_tracking()
    df_optimize = tracker.line.to_pandas()
    n_markers_optimize = (df_optimize.element_type == 'Marker').sum()
    assert n_markers_optimize == 0

    n_multipoles_before_optimize = (df_before_optimize.element_type == 'Multipole').sum()
    n_multipoles_optimize = (df_optimize.element_type == 'Multipole').sum()
    assert n_multipoles_before_optimize > n_multipoles_optimize

    n_drifts_before_optimize = (df_before_optimize.element_type == 'Drift').sum()
    n_drifts_optimize = (df_optimize.element_type == 'Drift').sum()
    assert n_drifts_before_optimize > n_drifts_optimize

    tracker.track(p_optimized, num_turns=num_turns, time=True)

    p_no_optimized.move(xo.context_default)
    p_optimized.move(xo.context_default)

    assert np.all(p_no_optimized.state == 1)
    assert np.all(p_optimized.state == 1)

    assert np.allclose(p_no_optimized.x, p_optimized.x, rtol=0, atol=1e-14)
    assert np.allclose(p_no_optimized.y, p_optimized.y, rtol=0, atol=1e-14)
    assert np.allclose(p_no_optimized.px, p_optimized.px, rtol=0, atol=1e-14)
    assert np.allclose(p_no_optimized.py, p_optimized.py, rtol=0, atol=1e-14)
    assert np.allclose(p_no_optimized.zeta, p_optimized.zeta, rtol=0, atol=1e-11)
    assert np.allclose(p_no_optimized.delta, p_optimized.delta, rtol=0, atol=1e-14)
