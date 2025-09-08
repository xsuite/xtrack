# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #
import json
import pathlib

import numpy as np
import pytest
from cpymad.madx import Madx

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts, fix_random_seed

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()


@for_all_test_contexts
def test_simple_collective_line(test_context):
    num_turns = 100
    elements = [xt.Drift(length=2., _context=test_context) for _ in range(5)]
    elements[3].iscollective = True
    line = xt.Line(elements=elements)
    line.reset_s_at_end_turn = False

    particles = xp.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12,
            _context=test_context)
    line.build_tracker(_context=test_context)
    line.track(particles, num_turns=num_turns)

    particles.move(_context=xo.ContextCpu())

    assert np.all(particles.at_turn == num_turns)
    xo.assert_allclose(particles.s, 10 * num_turns, rtol=0, atol=1e-14)



@for_all_test_contexts
def test_ebe_monitor(test_context):
    line = xt.Line(elements=[xt.Multipole(knl=[0, 1.]),
                            xt.Drift(length=0.5),
                            xt.Multipole(knl=[0, -1]),
                            xt.Cavity(frequency=400e7, voltage=6e6),
                            xt.Drift(length=.5),
                            xt.Drift(length=0)])

    line.build_tracker(_context=test_context)

    particles = xp.Particles(x=[1e-3, -2e-3, 5e-3], y=[2e-3, -4e-3, 3e-3],
                            zeta=1e-2, p0c=7e12, mass0=xp.PROTON_MASS_EV,
                            _context=test_context)

    line.track(particles.copy(), turn_by_turn_monitor='ONE_TURN_EBE')

    mon = line.record_last_track

    for ii, ee in enumerate(line.elements):
        for tt, nn in particles.per_particle_vars:
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

        line.build_tracker(_context=test_context)
        line.particle_ref = particle_ref

        cline_name = line.copy()
        cline_index = line.copy()
        cline_name.build_tracker(_context=test_context)
        cline_index.build_tracker(_context=test_context)

        cline_name.cycle(name_first_element='e2')
        cline_index.cycle(index_first_element=2)

        assert cline_name.tracker is not None
        assert cline_index.tracker is not None

        for cline in [cline_index, cline_name]:
            assert cline.element_names[0] == 'e2'
            assert cline.element_names[1] == 'e3'
            assert cline.element_names[2] == 'e0'
            assert cline.element_names[3] == 'e1'

            assert isinstance(cline.elements[0], xt.Drift)
            assert isinstance(cline.elements[1], xt.SRotation)
            assert isinstance(cline.elements[2], xt.Drift)
            assert isinstance(cline.elements[3], xt.Cavity)

            assert cline.particle_ref.mass0 == xp.PROTON_MASS_EV
            assert cline.particle_ref.gamma0 == 1.05

@for_all_test_contexts
def test_synrad_configuration(test_context):
    for collective in [False, True]:
        elements = [xt.Multipole(knl=[1]) for _ in range(10)]
        if collective:
            elements[5].iscollective = True
            elements[5].move(_context=test_context)

        line = xt.Line(elements=elements)
        line.build_tracker(_context=test_context)

        line.configure_radiation(model='mean')
        for ee in line.elements:
            assert ee.radiation_flag == 1
        p = xp.Particles(x=[0.01, 0.02], _context=test_context)
        line.track(p)
        p.move(_context=xo.ContextCpu())
        assert np.all(p._rng_s1 + p._rng_s2 + p._rng_s3 + p._rng_s4 == 0)

        line.configure_radiation(model='quantum')
        for ee in line.elements:
            assert ee.radiation_flag == 2
        p = xp.Particles(x=[0.01, 0.02], _context=test_context)
        line.track(p)
        p.move(_context=xo.ContextCpu())
        assert np.all(p._rng_s1 + p._rng_s2 + p._rng_s3 + p._rng_s4 > 0)

        line.configure_radiation(model=None)
        for ee in line.elements:
            assert ee.radiation_flag == 0
        p = xp.Particles(x=[0.01, 0.02], _context=test_context)
        line.track(p)
        p.move(_context=xo.ContextCpu())
        assert np.all(p._rng_s1 + p._rng_s2 + p._rng_s3 + p._rng_s4 > 0)


@for_all_test_contexts
def test_partial_tracking(test_context):
    n_elem = 9
    elements = [ xt.Drift(length=1.) for _ in range(n_elem) ]
    line = xt.Line(elements=elements)
    line.build_tracker(_context=test_context)
    assert not line.iscollective
    particles_init = xp.Particles(_context=test_context,
        x=[1e-3, -2e-3, 5e-3], y=[2e-3, -4e-3, 3e-3],
        zeta=1e-2, p0c=7e12, mass0=xp.PROTON_MASS_EV,
        at_turn=0, at_element=0)

    _default_track(line, particles_init)
    _ele_start_until_end(line, particles_init)
    _ele_start_with_shift(line, particles_init)
    _ele_start_with_shift_more_turns(line, particles_init)
    _ele_stop_from_start(line, particles_init)
    _ele_start_to_ele_stop(line, particles_init)
    _ele_start_to_ele_stop_with_overflow(line, particles_init)


@for_all_test_contexts
def test_partial_tracking_with_collective(test_context):
    n_elem = 9
    elements = [xt.Drift(length=1., _context=test_context) for _ in range(n_elem)]
    # Make some elements collective
    elements[3].iscollective = True
    elements[7].iscollective = True
    line = xt.Line(elements=elements)
    line.build_tracker(_context=test_context)
    assert line.iscollective
    assert len(line.tracker._parts) == 5
    particles_init = xp.Particles(
            _context=test_context,
            x=[1e-3, -2e-3, 5e-3], y=[2e-3, -4e-3, 3e-3],
            zeta=1e-2, p0c=7e12, mass0=xp.PROTON_MASS_EV,
            at_turn=0, at_element=0)

    _default_track(line, particles_init)
    _ele_start_until_end(line, particles_init)
    _ele_start_with_shift(line, particles_init)
    _ele_start_with_shift_more_turns(line, particles_init)
    _ele_stop_from_start(line, particles_init)
    _ele_start_to_ele_stop(line, particles_init)
    _ele_start_to_ele_stop_with_overflow(line, particles_init)


# Track from the start until the end of the first, second, and tenth turn
def _default_track(line, particles_init):
    n_elem = len(line.element_names)
    for turns in [1, 2, 10]:
        expected_end_turn = turns
        expected_end_element = 0
        expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

        particles = particles_init.copy()
        line.track(particles, num_turns=turns, turn_by_turn_monitor=True)
        check, end_turn, end_element, end_s = _get_at_turn_element(particles)
        assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                    and end_s==expected_end_element)
        assert line.record_last_track.x.shape == (len(particles.x), expected_num_monitor)


# Track, from any ele_start, until the end of the first, second, and tenth turn
def _ele_start_until_end(line, particles_init):
    n_elem = len(line.element_names)
    for turns in [1, 2, 10]:
        for start in range(n_elem):
            expected_end_turn = turns
            expected_end_element = 0
            expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

            particles = particles_init.copy()
            particles.at_element = start
            particles.s = start
            line.track(particles, num_turns=turns, ele_start=start, turn_by_turn_monitor=True)
            check, end_turn, end_element, end_s = _get_at_turn_element(particles)
            assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                        and end_s==expected_end_element)
            assert line.record_last_track.x.shape==(len(particles.x), expected_num_monitor)


# Track, from any ele_start, any shifts that stay within the first turn
def _ele_start_with_shift(line, particles_init):
    n_elem = len(line.element_names)
    for start in range(n_elem):
        for shift in range(1,n_elem-start):
            expected_end_turn = 0
            expected_end_element = start+shift
            expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

            particles = particles_init.copy()
            particles.at_element = start
            particles.s = start
            line.track(particles, ele_start=start, num_elements=shift, turn_by_turn_monitor=True)
            check, end_turn, end_element, end_s = _get_at_turn_element(particles)
            assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                        and end_s==expected_end_element)
            assert line.record_last_track.x.shape==(len(particles.x), expected_num_monitor)

# Track, from any ele_start, any shifts that are larger than one turn (up to 3 turns)
def _ele_start_with_shift_more_turns(line, particles_init):
    n_elem = len(line.element_names)
    for start in range(n_elem):
        for shift in range(n_elem-start, 3*n_elem+1):
            expected_end_turn = round(np.floor( (start+shift)/n_elem ))
            expected_end_element = start + shift - n_elem*expected_end_turn
            expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

            particles = particles_init.copy()
            particles.at_element = start
            particles.s = start
            line.track(particles, ele_start=start, num_elements=shift, turn_by_turn_monitor=True)
            check, end_turn, end_element, end_s = _get_at_turn_element(particles)
            assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                        and end_s==expected_end_element)
            assert line.record_last_track.x.shape==(len(particles.x), expected_num_monitor)


# Track from the start until any ele_stop in the first, second, and tenth turn
def _ele_stop_from_start(line, particles_init):
    n_elem = len(line.element_names)
    for turns in [1, 2, 10]:
        for stop in range(1, n_elem):
            expected_end_turn = turns-1
            expected_end_element = stop
            expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

            particles = particles_init.copy()
            line.track(particles, num_turns=turns, ele_stop=stop, turn_by_turn_monitor=True)
            check, end_turn, end_element, end_s = _get_at_turn_element(particles)
            assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                        and end_s==expected_end_element)
            assert line.record_last_track.x.shape==(len(particles.x), expected_num_monitor)


# Track from any ele_start until any ele_stop that is larger than ele_start
# for one, two, and ten turns
def _ele_start_to_ele_stop(line, particles_init):
    n_elem = len(line.element_names)
    for turns in [1, 2, 10]:
        for start in range(n_elem):
            for stop in range(start+1,n_elem):
                expected_end_turn = turns-1
                expected_end_element = stop
                expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

                particles = particles_init.copy()
                particles.at_element = start
                particles.s = start
                line.track(particles, num_turns=turns, ele_start=start, ele_stop=stop, turn_by_turn_monitor=True)
                check, end_turn, end_element, end_s = _get_at_turn_element(particles)
                assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                            and end_s==expected_end_element)
                assert line.record_last_track.x.shape==(len(particles.x), expected_num_monitor)


# Track from any ele_start until any ele_stop that is smaller than or equal to ele_start (turn increses by one)
# for one, two, and ten turns
def _ele_start_to_ele_stop_with_overflow(line, particles_init):
    n_elem = len(line.element_names)
    for turns in [1, 2, 10]:
        for start in range(n_elem):
            for stop in range(start+1):
                expected_end_turn = turns
                expected_end_element = stop
                expected_num_monitor = expected_end_turn if expected_end_element==0 else expected_end_turn+1

                particles = particles_init.copy()
                particles.at_element = start
                particles.s = start
                line.track(particles, num_turns=turns, ele_start=start, ele_stop=stop, turn_by_turn_monitor=True)
                check, end_turn, end_element, end_s = _get_at_turn_element(particles)
                assert (check and end_turn==expected_end_turn and end_element==expected_end_element
                            and end_s==expected_end_element)
                assert line.record_last_track.x.shape==(len(particles.x), expected_num_monitor)


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


def test_tracker_hashable_config():
    line = xt.Line([])
    line.build_tracker()
    line.config.TEST_FLAG_BOOL = True
    line.config.TEST_FLAG_INT = 42
    line.config.TEST_FLAG_FALSE = False
    line.config.ZZZ = 'lorem'
    line.config.AAA = 'ipsum'

    expected = (
        ('AAA', 'ipsum'),
        ('TEST_FLAG_BOOL', True),
        ('TEST_FLAG_INT', 42),
        ('XFIELDS_BB3D_NO_BEAMSTR', True), # active by default
        ('XFIELDS_BB3D_NO_BHABHA', True), # active by default
        ('XTRACK_GLOBAL_XY_LIMIT', 1.0), # active by default
        ('XTRACK_MULTIPOLE_NO_SYNRAD', True), # active by default
        ('ZZZ', 'lorem'),
    )

    assert line.tracker._hashable_config() == expected


def test_tracker_config_to_headers():
    line = xt.Line([])
    line.build_tracker()

    line.config.clear()
    line.config.TEST_FLAG_BOOL = True
    line.config.TEST_FLAG_INT = 42
    line.config.TEST_FLAG_FALSE = False
    line.config.ZZZ = 'lorem'
    line.config.AAA = 'ipsum'

    expected = [
        '#define TEST_FLAG_BOOL',
        '#define TEST_FLAG_INT 42',
        '#define ZZZ lorem',
        '#define AAA ipsum',
    ]

    assert set(line.tracker._config_to_headers()) == set(expected)


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
    line.build_tracker(_context=test_context)

    particles = xp.Particles(p0c=1e9, x=[0], y=[0], _context=test_context)

    p = particles.copy()
    line.config.TEST_FLAG = 2
    line.track(p)
    assert p.x[0] == 7.0
    assert p.y[0] == 0.0
    first_kernel, first_data = line.tracker.get_track_kernel_and_data_for_present_config()

    p = particles.copy()
    line.config.TEST_FLAG = False
    line.config.TEST_FLAG_BOOL = True
    line.track(p)
    assert p.x[0] == 0.0
    assert p.y[0] == 42.0
    current_kernel, current_data = line.tracker.get_track_kernel_and_data_for_present_config()
    assert current_kernel is not first_kernel
    assert current_data is not first_data

    line.config.TEST_FLAG = 2
    line.config.TEST_FLAG_BOOL = False
    assert len(line.tracker.track_kernel) == 3 # As line.track_kernel.keys() =
                                          # dict_keys([(), (('TEST_FLAG', 2),), (('TEST_FLAG_BOOL', True),)])
    current_kernel, current_data = line.tracker.get_track_kernel_and_data_for_present_config()
    assert current_kernel is first_kernel
    assert current_data is first_data

@for_all_test_contexts
@pytest.mark.parametrize('multiline', [False, True])
def test_optimize_for_tracking(test_context, multiline):

    if multiline:
        mline = xt.load(test_data_folder /
                         "hllhc15_collider/collider_00_from_mad.json")
        line = mline['lhcb1']
        line.particle_ref = xp.Particles(p0c=7000e9)
        line.vars['vrf400'] = 16
    else:
        fname_line_particles = test_data_folder / 'hllhc15_noerrors_nobb/line_and_particle.json'
        with open(fname_line_particles, 'r') as fid:
            input_data = json.load(fid)
        line = xt.Line.from_dict(input_data['line'])
        line.particle_ref = xp.Particles.from_dict(input_data['particle'])

    line.build_tracker(_context=test_context)

    particles = line.build_particles(
        x_norm=np.linspace(-2, 2, 1000), y_norm=0.1, delta=3e-4,
        nemitt_x=2.5e-6, nemitt_y=2.5e-6)

    p_no_optimized = particles.copy()
    p_optimized = particles.copy()

    num_turns = 10

    line.track(p_no_optimized, num_turns=num_turns, time=True)
    df_before_optimize = line.to_pandas()
    n_markers_before_optimize = (df_before_optimize.element_type == 'Marker').sum()
    assert n_markers_before_optimize > 4 # There are at least the IPs

    line.optimize_for_tracking(keep_markers=True)
    df_optimize_keep_markers = line.to_pandas()
    n_markers_optimize_keep = (df_optimize_keep_markers.element_type == 'Marker').sum()
    assert n_markers_optimize_keep == n_markers_before_optimize

    line.optimize_for_tracking(keep_markers=['ip1', 'ip5'])
    df_optimize_ip15 = line.to_pandas()
    n_markers_optimize_ip15 = (df_optimize_ip15.element_type == 'Marker').sum()
    assert n_markers_optimize_ip15 == 2

    line.optimize_for_tracking()

    assert type(line['mb.b10l3.b1..1']) is xt.SimpleThinBend
    assert type(line['mq.10l3.b1..1']) is xt.SimpleThinQuadrupole

    df_optimize = line.to_pandas()
    n_markers_optimize = (df_optimize.element_type == 'Marker').sum()
    assert n_markers_optimize == 0

    n_multipoles_before_optimize = (df_before_optimize.element_type == 'Multipole').sum()
    n_multipoles_optimize = (df_optimize.element_type == 'Multipole').sum()
    assert n_multipoles_before_optimize > n_multipoles_optimize

    n_drifts_before_optimize = (df_before_optimize.element_type == 'Drift').sum()
    n_drifts_optimize = (df_optimize.element_type == 'Drift').sum()
    assert n_drifts_before_optimize > n_drifts_optimize

    line.track(p_optimized, num_turns=num_turns, time=True)

    p_no_optimized.move(xo.context_default)
    p_optimized.move(xo.context_default)

    assert np.all(p_no_optimized.state == 1)
    assert np.all(p_optimized.state == 1)

    xo.assert_allclose(p_no_optimized.x, p_optimized.x, rtol=0, atol=1e-14)
    xo.assert_allclose(p_no_optimized.y, p_optimized.y, rtol=0, atol=1e-14)
    xo.assert_allclose(p_no_optimized.px, p_optimized.px, rtol=0, atol=1e-14)
    xo.assert_allclose(p_no_optimized.py, p_optimized.py, rtol=0, atol=1e-14)
    xo.assert_allclose(p_no_optimized.zeta, p_optimized.zeta, rtol=0, atol=1e-10)
    xo.assert_allclose(p_no_optimized.delta, p_optimized.delta, rtol=0, atol=1e-14)


@for_all_test_contexts
def test_backtrack_with_flag(test_context):

    line = xt.load(test_data_folder /
                'hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
    line.build_tracker(_context=test_context)

    line.vars['on_crab1'] = -190
    line.vars['on_crab5'] = -190
    line.vars['on_x1'] = 130
    line.vars['on_x5'] = 130

    p = xp.Particles(_context=test_context,
        p0c=7000e9, x=1e-4, px=1e-6, y=2e-4, py=3e-6, zeta=0.01, delta=1e-4)

    line.track(p, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_forward = line.record_last_track

    line.track(p, backtrack=True, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_backtrack = line.record_last_track

    xo.assert_allclose(mon_forward.x, mon_backtrack.x, rtol=0, atol=1e-10)
    xo.assert_allclose(mon_forward.y, mon_backtrack.y, rtol=0, atol=1e-10)
    xo.assert_allclose(mon_forward.px, mon_backtrack.px, rtol=0, atol=1e-10)
    xo.assert_allclose(mon_forward.py, mon_backtrack.py, rtol=0, atol=1e-10)
    xo.assert_allclose(mon_forward.zeta, mon_backtrack.zeta, rtol=0, atol=1e-10)
    xo.assert_allclose(mon_forward.delta, mon_backtrack.delta, rtol=0, atol=1e-10)


@for_all_test_contexts
@pytest.mark.parametrize(
    'with_progress,turns',
    [(True, 300), (True, 317), (7, 523), (1, 21), (10, 10)]
)
@pytest.mark.parametrize('collective', [True, False], ids=['collective', 'non-collective'])
def test_tracking_with_progress(test_context, with_progress, turns, collective):
    elements = [xt.Drift(length=2, _context=test_context) for _ in range(5)]
    elements[3].iscollective = collective
    line = xt.Line(elements=elements)
    line.reset_s_at_end_turn = False

    particles = xp.Particles(x=[1e-3, 2e-3, 3e-3], p0c=7e12, _context=test_context)
    line.build_tracker(_context=test_context)
    line.track(particles, num_turns=turns, with_progress=with_progress)
    particles.move(xo.ContextCpu())

    assert np.all(particles.at_turn == turns)
    xo.assert_allclose(particles.s, 10 * turns, rtol=0, atol=1e-14)


@for_all_test_contexts
@pytest.mark.parametrize(
    'ele_start,ele_stop,expected_x',
    [
        (None, None, [0, 0.005, 0.010, 0.015, 0.020, 0.025]),
        (None, 3, [0, 0.005, 0.010, 0.015, 0.020, 0.023]),
        (2, None, [0, 0.003, 0.008, 0.013, 0.018, 0.023]),
        (2, 3, [0, 0.003, 0.008, 0.013, 0.018, 0.021]),
        (3, 2, [0, 0.002, 0.007, 0.012, 0.017, 0.022, 0.024]),
    ],
)
@pytest.mark.parametrize('with_progress', [False, True, 1, 2, 3])
def test_tbt_monitor_with_progress(test_context, ele_start, ele_stop, expected_x, with_progress):
    line = xt.Line(elements=[xt.Drift(length=1, _context=test_context)] * 5)
    line.build_tracker(_context=test_context)

    p = xt.Particles(px=0.001, _context=test_context)
    line.track(p, num_turns=5, turn_by_turn_monitor=True, with_progress=with_progress, ele_start=ele_start, ele_stop=ele_stop)
    p.move(_context=xo.context_default)

    monitor_recorded_x = line.record_last_track.x
    assert monitor_recorded_x.shape == (1, len(expected_x) - 1)

    recorded_x = np.concatenate([monitor_recorded_x[0], p.x])
    xo.assert_allclose(recorded_x, expected_x, atol=1e-16)


@pytest.fixture
def pimms_mad():
    pimms_path = test_data_folder / 'pimms/PIMMS.seq'
    mad = Madx(stdout=False)
    mad.option(echo=False)
    mad.call(str(pimms_path))
    mad.beam()
    mad.use('pimms')
    return mad


@for_all_test_contexts
@fix_random_seed(784239)
def test_track_log_and_merit_function(pimms_mad, test_context):
    line = xt.Line.from_madx_sequence(
        pimms_mad.sequence.pimms,
        deferred_expressions=True,
    )
    line.particle_ref = xt.Particles(
        q0=1,
        mass0=xt.PROTON_MASS_EV,
        kinetic_energy0=200e6, # eV
    )
    line.insert_element(
        name='septum_aperture',
        element=xt.LimitRect(min_x=-0.1, max_x=0.1, min_y=-0.1, max_y=0.1),
        at='extr_septum',
    )
    line['septum_aperture'].max_x = 0.035
    line.configure_bend_model(edge='full', core='adaptive', num_multipole_kicks=1)

    line.vars['kqf_common'] = 0
    line.vars['kqfa'] = line.vars['kqf_common']
    line.vars['kqfb'] = line.vars['kqf_common']

    line.vars['kqf_common'] = 2e-2
    line.vars['kqd'] = -2e-2

    line.build_tracker(_context=test_context)

    # Prepare a match on the tunes, dispersion at electrostatic septum, and
    # correct the chromaticity. Don't run it, instead we will simulate an
    # external optimizer.
    opt = line.match(
        solve=False,
        method='4d',
        vary=[
            xt.VaryList(['ksf', 'ksd'], step=1e-3),
            xt.VaryList(['kqfa', 'kqfb'], limits=(0, 1), step=1e-3, tag='qf'),
            xt.Vary('kqd', limits=(-1, 0), step=1e-3, tag='qd', weight=10),
        ],
        targets=[
            xt.TargetSet(dqx=-0.1, dqy=-0.1, tol=1e-3, tag="chrom"),
            xt.Target(dx=0, at='extr_septum', tol=1e-6),
            xt.TargetSet(qx=1.663, qy=1.72, tol=1e-6, tag="tunes"),
        ]
    )

    # Check that the merit function works correctly
    merit_function = opt.get_merit_function(return_scalar=True, check_limits=False)

    x_expected = [line.vars[kk]._value for kk in ['ksf', 'ksd', 'kqfa', 'kqfb', 'kqd']]
    x_expected[4] /= 10  # include the weight
    x_start = merit_function.get_x()
    xo.assert_allclose(x_start, x_expected, atol=1e-14)

    expected_limits = [
        [-1e200, 1e200],  # default
        [-1e200, 1e200],  # default
        [0, 1],
        [0, 1],
        [-0.1, 0],
    ]
    xo.assert_allclose(merit_function.get_x_limits(), expected_limits, atol=1e-14)

    # Below numbers obtained by first only matching the tunes, then the above
    x_optimized = [-1.40251213, 0.81823393, 0.31196667, 0.52478984, -0.052393429]
    merit_function.set_x(x_optimized)
    assert np.all(opt.target_status(ret=True)['tol_met'])

    # Now prepare to track and to log intensity and sextupole strength
    num_particles = 1000
    beam_intensity = 1e10  # p+

    # Generate particles with a gaussian distribution in normalized phase space,
    # and a gaussian momentum distribution (rms spread 5e-4)
    particles = line.build_particles(
        x_norm=np.random.normal(size=num_particles),
        px_norm=np.random.normal(size=num_particles),
        y_norm=np.random.normal(size=num_particles),
        py_norm=np.random.normal(size=num_particles),
        delta=5e-4 * np.random.normal(size=num_particles),
        method='4d',
        weight=beam_intensity / num_particles,
        nemitt_x=1.5e-6, nemitt_y=1e-6,
    )

    # Define time-dependent behaviour of the quadrupoles
    line.functions['fun_kqfa'] = xt.FunctionPieceWiseLinear(
        x=[0, 0.5e-3],
        y=[line.vv['kqfa'], 0.313],
    )
    line.vars['kqfa'] = line.functions['fun_kqfa'](line.vars['t_turn_s'])
    line.vars['kse2'] = 9

    kqfa_before = line.vv['kqfa']

    def measure_intensity(_, particles):
        ctx2np = particles._context.nparray_from_context_array
        state = ctx2np(particles.state)
        weight = ctx2np(particles.weight)
        mask_alive = state > 0
        return np.sum(weight[mask_alive])

    intensity_before = measure_intensity(None, particles)

    log = xt.Log('kqfa',  # vars to be logged
                 intensity=measure_intensity)  # user-defined function to be logged

    line.discard_tracker()
    line.build_tracker(_context=test_context)
    line.enable_time_dependent_vars = True
    num_turns = 1000
    line.track(particles=particles, num_turns=num_turns, log=log)

    intensity_after = measure_intensity(None, particles)

    # Check that kqfa is increasing linearly
    fit = np.polyfit(np.arange(num_turns), line.log_last_track['kqfa'], 1, full=True)
    (slope, _), residual, _, _, _ = fit
    assert slope > 0
    assert residual < 1e-28
    xo.assert_allclose(line.log_last_track['kqfa'][0], kqfa_before, atol=1e-14, rtol=0)
    xo.assert_allclose(line.log_last_track['kqfa'][-1], line.vv['kqfa'], atol=1e-14, rtol=0)

    # Check that intensity is decreasing
    intensity = np.array(line.log_last_track['intensity'])
    assert np.all(intensity[:-1] - intensity[1:] >= 0)
    xo.assert_allclose(intensity[0], intensity_before, atol=1e-14, rtol=0)
    # The last log point is from the beginning of the last turn:
    xo.assert_allclose(intensity[-1], intensity_after, atol=0, rtol=1e-2)


@for_all_test_contexts
def test_init_io_buffer(test_context):
    class TestElementRecord(xo.HybridClass):
        _xofields = {
            '_index': xt.RecordIndex,
            'record_field': xo.Int64[:],
            'record_at_element': xo.Int64[:],
        }

    class TestElement(xt.BeamElement):
        _xofields={
            'element_field': xo.Int64,
        }

        _internal_record_class = TestElementRecord

        _extra_c_sources = [
            r'''
            /*gpufun*/
            void TestElement_track_local_particle(TestElementData el, LocalParticle* part0){
                // Extract the record and record_index
                TestElementRecordData record = TestElementData_getp_internal_record(el, part0);
                RecordIndex record_index = NULL;
                if (record){
                    record_index = TestElementRecordData_getp__index(record);
                }

                int64_t elem_field = TestElementData_get_element_field(el);

                //start_per_particle_block (part0->part)
                    if (record) {  // Record exists
                        // Get a slot in the record (this is thread safe)
                        int64_t i_slot = RecordIndex_get_slot(record_index);

                        if (i_slot>=0) {  // Slot available
                            TestElementRecordData_set_record_field(
                                record,
                                i_slot,
                                elem_field
                            );
                            TestElementRecordData_set_record_at_element(
                                record,
                                i_slot,
                                LocalParticle_get_at_element(part)
                            );
                        }
                    }
                //end_per_particle_block
            }
            '''
        ]

    line = xt.Line(elements=[
        TestElement(element_field=3),
        TestElement(element_field=4),
    ])
    line.build_tracker(_context=test_context)

    record = line.start_internal_logging_for_elements_of_type(
        TestElement,
        capacity=1000,
    )

    num_turns = 100

    part = xp.Particles(_context=test_context, x=[1e-3, 2e-3, 3e-3])
    line.track(part, num_turns=num_turns)

    num_recorded = record._index.num_recorded
    num_particles = len(part.x)

    record.move(_context=xo.ContextCpu())

    assert num_recorded == (2 * num_particles * num_turns)
    assert np.sum(record.record_field == 3) == num_particles * num_turns
    assert np.sum(record.record_field == 4) == num_particles * num_turns
    assert np.all(record.record_field[:num_recorded][
                    record.record_at_element[:num_recorded] == 0] == 3)
    assert np.all(record.record_field[:num_recorded][
                    record.record_at_element[:num_recorded] == 1] == 4)
    assert np.all(record.record_field[num_recorded:] == 0)

    # Now we stop logging and manually reset to mimic the situation where the
    # record is manually flushed.
    line.stop_internal_logging_for_elements_of_type(TestElement)
    line.tracker._init_io_buffer()

    assert line.tracker.io_buffer is not None
