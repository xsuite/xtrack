# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #
import numpy as np
import pytest
import xobjects as xo
import xpart as xp
from cpymad.madx import Madx
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt
from xtrack.mad_loader import MadLoader
from xtrack.slicing import Strategy, Uniform


@pytest.mark.parametrize(
    'k0, k1, length',
    [
        (-0.1, 0, 0.9),
        (0, 0, 0.9),
        (-0.1, 0.12, 0.9),
        (0, 0.12, 0.8),
        (0.15, -0.23, 0.9),
        (0, 0.13, 1.7),
    ]
)
@for_all_test_contexts
def test_combined_function_dipole_against_madx(test_context, k0, k1, length):
    """
    Test the combined function dipole against madx. We import bends from madx
    using use_true_thick_bend=False, and the true bend is not in madx.
    """

    p0 = xp.Particles(
        mass0=xp.PROTON_MASS_EV,
        beta0=0.5,
        x=0.01,
        px=0.1,
        y=-0.03,
        py=0.001,
        zeta=0.1,
        delta=[-0.8, -0.5, -0.1, 0, 0.1, 0.5, 0.8],
        _context=test_context,
    )
    mad = Madx()
    mad.input(f"""
    ss: sequence, l={length};
        b: sbend, at={length / 2}, angle={k0 * length}, k1={k1}, l={length};
    endsequence;
    beam;
    use, sequence=ss;
    """)

    ml = MadLoader(mad.sequence.ss, allow_thick=True)
    line_thick = ml.make_line()
    line_thick.build_tracker(_context=test_context)
    line_thick.config.XTRACK_USE_EXACT_DRIFTS = True # to be consistent with madx for large angle and k0 = 0

    for ii in range(len(p0.x)):
        mad.input(f"""
        beam, particle=proton, pc={p0.p0c[ii] / 1e9}, sequence=ss, radiate=FALSE;

        track, onepass, onetable;
        start, x={p0.x[ii]}, px={p0.px[ii]}, y={p0.y[ii]}, py={p0.py[ii]}, \
            t={p0.zeta[ii]/p0.beta0[ii]}, pt={p0.ptau[ii]};
        run, turns=1;
        endtrack;
        """)

        mad_results = mad.table.mytracksumm[-1]

        part = p0.copy(_context=test_context)
        line_thick.track(part, _force_no_end_turn_actions=True)
        part.move(_context=xo.context_default)

        xt_tau = part.zeta/part.beta0
        assert np.allclose(part.x[ii], mad_results.x, atol=1e-11, rtol=0)
        assert np.allclose(part.px[ii], mad_results.px, atol=1e-11, rtol=0)
        assert np.allclose(part.y[ii], mad_results.y, atol=1e-11, rtol=0)
        assert np.allclose(part.py[ii], mad_results.py, atol=1e-11, rtol=0)
        assert np.allclose(xt_tau[ii], mad_results.t, atol=1e-10, rtol=0)
        assert np.allclose(part.ptau[ii], mad_results.pt, atol=1e-11, rtol=0)


def test_thick_bend_survey():
    circumference = 10
    rho = circumference / (2 * np.math.pi)
    h = 1 / rho
    k = 1 / rho

    p0 = xp.Particles(p0c=7e12, mass0=xp.PROTON_MASS_EV, x=0.7, px=-0.4, delta=0.0)

    el = xt.Bend(k0=k, h=h, length=circumference, num_multipole_kicks=0, model='full')
    line = xt.Line(elements=[el])
    line.reset_s_at_end_turn = False
    line.build_tracker()

    s_array = np.linspace(0, circumference, 1000)

    X0_array = np.zeros_like(s_array)
    Z0_array = np.zeros_like(s_array)

    X_array = np.zeros_like(s_array)
    Z_array = np.zeros_like(s_array)

    for ii, s in enumerate(s_array):
        p = p0.copy()

        el.length = s
        el.knl = np.array([3e-4, 4e-4, 0, 0, 0]) * s / circumference
        line.track(p)

        theta = s / rho

        X0 = -rho * (1 - np.cos(theta))
        Z0 = rho * np.sin(theta)

        ex_X = np.cos(theta)
        ex_Z = np.sin(theta)

        X0_array[ii] = X0
        Z0_array[ii] = Z0

        X_array[ii] = X0 + p.x[0] * ex_X
        Z_array[ii] = Z0 + p.x[0] * ex_Z

    Xmid = (np.min(X_array) + np.max(X_array)) / 2
    Zmid = (np.min(Z_array) + np.max(Z_array)) / 2
    Xc = X_array - Xmid
    Zc = Z_array - Zmid
    rhos = np.sqrt(Xc ** 2 + Zc ** 2)
    errors = np.max(np.abs(rhos - 10 / (2 * np.math.pi)))
    assert errors < 2e-6

@for_all_test_contexts
@pytest.mark.parametrize('element_type', [xt.Bend, xt.CombinedFunctionMagnet])
@pytest.mark.parametrize('h', [0.0, 0.1])
def test_thick_multipolar_component(test_context, element_type, h):
    bend_length = 1.0
    k0 = h
    knl = np.array([0.0, 0.01, -0.02, 0.03])
    ksl = np.array([0.0, -0.03, 0.02, -0.01])
    num_kicks = 2

    # Bend with a multipolar component
    bend_with_mult = element_type(
        k0=k0,
        h=h,
        length=bend_length,
        knl=knl,
        ksl=ksl,
        num_multipole_kicks=num_kicks,
    )

    # Separate bend and a corresponding multipole
    bend_no_mult = element_type(
        k0=k0,
        h=h,
        length=bend_length / (num_kicks + 1),
        num_multipole_kicks=0,
    )
    multipole = xt.Multipole(
        knl=knl / num_kicks,
        ksl=ksl / num_kicks,
    )

    # Two lines that should be equivalent
    line_no_slices = xt.Line(
        elements=[bend_with_mult],
        element_names=['bend_with_mult'],
    )
    line_with_slices = xt.Line(
        elements={'bend_no_mult': bend_no_mult, 'multipole': multipole},
        element_names=(['bend_no_mult', 'multipole'] * num_kicks) + ['bend_no_mult'],
    )

    # Track some particles
    p0 = xp.Particles(x=0.1, px=0.2, y=0.3, py=0.4, zeta=0.5, delta=0.6,
                      _context=test_context)

    p_no_slices = p0.copy(_context=test_context)
    line_no_slices.build_tracker(_context=test_context)
    line_no_slices.track(p_no_slices)

    p_with_slices = p0.copy()
    line_with_slices.build_tracker(_context=test_context)
    line_with_slices.track(p_with_slices, turn_by_turn_monitor='ONE_TURN_EBE')

    p_no_slices.move(_context=xo.context_default)
    p_with_slices.move(_context=xo.context_default)

    # Check that the results are the same
    for attr in ['x', 'px', 'y', 'py', 'zeta', 'delta']:
        assert np.allclose(
            getattr(p_no_slices, attr),
            getattr(p_with_slices, attr),
            atol=1e-14,
            rtol=0,
        )

@pytest.mark.parametrize(
    'with_knobs',
    [True, False],
    ids=['with knobs', 'no knobs'],
)
@pytest.mark.parametrize(
    'use_true_thick_bends',
    [True, False],
    ids=['true bend', 'combined function magnet'],
)
@pytest.mark.parametrize('bend_type', ['rbend', 'sbend'])
def test_import_thick_bend_from_madx(use_true_thick_bends, with_knobs, bend_type):
    mad = Madx()
    mad.options.rbarc = False

    mad.input(f"""
    knob_a := 1.0;
    knob_b := 2.0;
    ! Make the sequence a bit longer to accommodate rbends
    ss: sequence, l:=2 * knob_b, refer=entry;
        elem: {bend_type}, at=0, angle:=0.1 * knob_a, l:=knob_b,
            k0:=0.2 * knob_a, k1=0, k2:=0.4 * knob_a,
            fint:=0.5 * knob_a, hgap:=0.6 * knob_a,
            e1:=0.7 * knob_a, e2:=0.8 * knob_a;
    endsequence;
    """)
    mad.beam()
    mad.use(sequence='ss')

    line = xt.Line.from_madx_sequence(
        sequence=mad.sequence.ss,
        deferred_expressions=with_knobs,
        allow_thick=True,
    )
    line.configure_bend_model(core={False: 'expanded', True: 'full'}[
                              use_true_thick_bends])

    elem_den = line['elem_den']
    elem = line['elem']
    elem_dex = line['elem_dex']

    # Check that the line has correct values to start with
    assert elem.model == {False: 'expanded', True: 'full'}[use_true_thick_bends]
    assert isinstance(elem_den, xt.DipoleEdge)
    assert isinstance(elem_dex, xt.DipoleEdge)

    # Element:
    assert np.isclose(elem.length, 2.0, atol=1e-16)
    assert np.isclose(elem.k0, 0.2, atol=1e-16)
    assert np.isclose(elem.h, 0.05, atol=1e-16)  # h = angle / L
    assert np.allclose(elem.ksl, 0.0, atol=1e-16)

    assert np.allclose(
        elem.knl,
        np.array([0, 0, 0.8, 0, 0]),  # knl = [0, 0, k2 * L, 0, 0]
        atol=1e-16,
    )

    # Edges:
    assert np.isclose(elem_den.fint, 0.5, atol=1e-16)
    assert np.isclose(elem_den.hgap, 0.6, atol=1e-16)
    assert np.isclose(elem_den.e1,
                      {'rbend': 0.7 + 0.1 / 2, 'sbend': 0.7}[bend_type],
                      atol=1e-16)
    assert np.isclose(elem_den.k, 0.2, atol=1e-16)

    assert np.isclose(elem_dex.fint, 0.5, atol=1e-16)
    assert np.isclose(elem_dex.hgap, 0.6, atol=1e-16)
    assert np.isclose(elem_dex.e1,
                     {'rbend': 0.8 + 0.1 / 2, 'sbend': 0.8}[bend_type],
                      atol=1e-16)
    assert np.isclose(elem_dex.k, 0.2, atol=1e-16)

    # Finish the test here if we are not using knobs
    if not with_knobs:
        assert line.vars is None
        return

    # Change the knob values
    line.vars['knob_a'] = 2.0
    line.vars['knob_b'] = 3.0

    # Verify that the line has been adjusted correctly
    # Element:
    assert np.isclose(elem.length, 3.0, atol=1e-16)
    assert np.isclose(elem.k0, 0.4, atol=1e-16)
    assert np.isclose(elem.h, 0.2 / 3.0, atol=1e-16)  # h = angle / length
    assert np.allclose(elem.ksl, 0.0, atol=1e-16)

    assert np.allclose(
        elem.knl,
        np.array([0, 0, 2.4, 0, 0]),  # knl = [0, 0, k2 * L, 0, 0]
        atol=1e-16,
    )

    # Edges:
    assert np.isclose(elem_den.fint, 1.0, atol=1e-16)
    assert np.isclose(elem_den.hgap, 1.2, atol=1e-16)
    assert np.isclose(elem_den.e1,
        {'rbend': 1.4 + 0.2 / 2, 'sbend': 1.4}[bend_type],
        atol=1e-16)
    assert np.isclose(elem_den.k, 0.4, atol=1e-16)

    assert np.isclose(elem_dex.fint, 1.0, atol=1e-16)
    assert np.isclose(elem_dex.hgap, 1.2, atol=1e-16)
    assert np.isclose(elem_dex.e1,
        {'rbend': 1.6 + 0.2 / 2, 'sbend': 1.6}[bend_type],
        atol=1e-16)
    assert np.isclose(elem_dex.k, 0.4, atol=1e-16)

@pytest.mark.parametrize('with_knobs', [False, True])
def test_import_thick_quad_from_madx(with_knobs):
    mad = Madx()

    mad.input(f"""
    knob_a := 0.0;
    knob_b := 2.0;
    ss: sequence, l:=knob_b, refer=entry;
        elem: quadrupole, at=0, k1:=0.1 + knob_a, k1s:=0.2 + knob_a, l:=knob_b;
    endsequence;
    """)
    mad.beam()
    mad.use(sequence='ss')

    line = xt.Line.from_madx_sequence(
        sequence=mad.sequence.ss,
        allow_thick=True,
        deferred_expressions=with_knobs,
    )

    elem_tilt_entry = line['elem_tilt_entry']
    elem = line['elem']
    elem_tilt_exit = line['elem_tilt_exit']

    # Verify that the line has been imported correctly
    assert np.isclose(elem.length, 2.0, atol=1e-16)
    assert np.isclose(elem.k1, 0.5 * np.sqrt(0.01 + 0.04), atol=1e-16)

    expected_tilt_before = -np.arctan2(0.2, 0.1) / 2
    tilt_entry = elem_tilt_entry.angle / 180 * np.math.pi  # rotation takes degrees
    assert np.isclose(tilt_entry, expected_tilt_before, atol=1e-16)
    tilt_exit = elem_tilt_exit.angle / 180 * np.math.pi  # ditto
    assert np.isclose(-expected_tilt_before, tilt_exit, atol=1e-16)

    # Finish the test here if we are not using knobs
    if not with_knobs:
        assert line.vars is None
        return

    # Change the knob values
    line.vars['knob_a'] = 1.0
    line.vars['knob_b'] = 3.0

    # Verify that the line has been adjusted correctly
    assert np.isclose(elem.length, 3.0, atol=1e-16)
    assert np.isclose(elem.k1, 0.5 * np.sqrt(1.21 + 1.44), atol=1e-16)

    expected_tilt_after = -np.arctan2(1.2, 1.1) / 2
    changed_tilt_entry = elem_tilt_entry.angle / 180 * np.math.pi  # rotation takes degrees
    assert np.isclose(changed_tilt_entry, expected_tilt_after, atol=1e-16)
    changed_tilt_exit = elem_tilt_exit.angle / 180 * np.math.pi  # ditto
    assert np.isclose(-expected_tilt_after, changed_tilt_exit, atol=1e-16)


@pytest.mark.parametrize(
    'with_knobs',
    [True, False],
    ids=['with knobs', 'no knobs'],
)
@pytest.mark.parametrize('bend_type', ['rbend', 'sbend'])
def test_import_thick_bend_from_madx_and_slice(
        with_knobs,
        bend_type,
):
    mad = Madx()
    mad.options.rbarc = False
    mad.input(f"""
    knob_a := 1.0;
    knob_b := 2.0;
    ! Make the sequence a bit longer to accommodate rbends
    ss: sequence, l:=2 * knob_b, refer=entry;
        elem: {bend_type}, at=0, angle:=0.1 * knob_a, l:=knob_b,
            k0:=0.2 * knob_a, k1=0, k2:=0.4 * knob_a,
            fint:=0.5 * knob_a, hgap:=0.6 * knob_a,
            e1:=0.7 * knob_a, e2:=0.8 * knob_a;
    endsequence;
    """)
    mad.beam()
    mad.use(sequence='ss')

    line = xt.Line.from_madx_sequence(
        sequence=mad.sequence.ss,
        deferred_expressions=with_knobs,
        allow_thick=True,
    )

    line.slice_thick_elements(slicing_strategies=[Strategy(Uniform(2))])

    elems = [line[f'elem..{ii}'] for ii in range(2)]
    drifts = [line[f'drift_elem..{ii}'] for ii in range(2)]

    # Verify that the slices are correct
    for elem in elems:
        assert np.isclose(elem.length, 1.0, atol=1e-16)
        assert np.allclose(elem.knl, [0.2, 0, 0.4, 0, 0], atol=1e-16)
        assert np.allclose(elem.ksl, 0, atol=1e-16)
        assert np.isclose(elem.hxl, 0.05, atol=1e-16)
        assert np.isclose(elem.hyl, 0, atol=1e-16)

    for drift in drifts:
        assert np.isclose(drift.length, 2/3, atol=1e-16)

    # Finish the test here if we are not using knobs
    if not with_knobs:
        assert line.vars is None
        return

    # Change the knob values
    line.vars['knob_a'] = 2.0
    line.vars['knob_b'] = 3.0

    # Verify that the line has been adjusted correctly
    for elem in elems:
        assert np.isclose(elem.length, 1.5, atol=1e-16)
        assert np.allclose(elem.knl, [0.6, 0., 1.2, 0, 0], atol=1e-16)
        assert np.allclose(elem.ksl, 0, atol=1e-16)
        assert np.isclose(elem.hxl, 0.1, atol=1e-16)  # hl = angle / slice_count
        assert np.isclose(elem.hyl, 0, atol=1e-16)

    for drift in drifts:
        assert np.isclose(drift.length, 1, atol=1e-16)


@pytest.mark.parametrize(
    'with_knobs',
    [True, False],
    ids=['with knobs', 'no knobs'],
)
def test_import_thick_quad_from_madx_and_slice(with_knobs):
    mad = Madx()
    mad.input(f"""
    knob_a := 0.0;
    knob_b := 2.0;
    ss: sequence, l:=knob_b, refer=entry;
        elem: quadrupole, at=0, k1:=0.1 + knob_a, k1s:=0.2 + knob_a, l:=knob_b;
    endsequence;
    """)
    mad.beam()
    mad.use(sequence='ss')

    line = xt.Line.from_madx_sequence(
        sequence=mad.sequence.ss,
        deferred_expressions=with_knobs,
        allow_thick=True,
    )

    line.slice_thick_elements(slicing_strategies=[Strategy(Uniform(2))])

    elems = [line[f'elem..{ii}'] for ii in range(2)]
    drifts = [line[f'drift_elem..{ii}'] for ii in range(2)]

    # Verify that the slices are correct
    for elem in elems:
        assert np.isclose(elem.length, 1.0, atol=1e-16)
        expected_k1l = 0.5 * np.sqrt(0.01 + 0.04) * 2
        assert np.allclose(elem.knl, [0, expected_k1l / 2, 0, 0, 0], atol=1e-16)
        assert np.allclose(elem.ksl, 0, atol=1e-16)
        assert np.isclose(elem.hxl, 0, atol=1e-16)
        assert np.isclose(elem.hyl, 0, atol=1e-16)

    for drift in drifts:
        assert np.isclose(drift.length, 2/3, atol=1e-16)

    # Finish the test here if we are not using knobs
    if not with_knobs:
        assert line.vars is None
        return

    # Change the knob values
    line.vars['knob_a'] = 2.0
    line.vars['knob_b'] = 3.0

    # Verify that the line has been adjusted correctly
    for elem in elems:
        assert np.isclose(elem.length, 1.5, atol=1e-16)
        expected_k1l = 0.5 * np.sqrt(2.2 ** 2 + 2.1 ** 2) * 3
        assert np.allclose(elem.knl, [0, expected_k1l / 2, 0, 0, 0], atol=1e-16)
        assert np.allclose(elem.ksl, 0, atol=1e-16)
        assert np.isclose(elem.hxl, 0, atol=1e-16)
        assert np.isclose(elem.hyl, 0, atol=1e-16)

    for drift in drifts:
        assert np.isclose(drift.length, 1, atol=1e-16)

@for_all_test_contexts
def test_fringe_implementations(test_context):

    fringe = xt.DipoleEdge(k=0.12, fint=100, hgap=0.035, model='full')

    line = xt.Line(elements=[fringe])
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, beta0=0.5)
    line.build_tracker(_context=test_context)

    p0 = line.build_particles(px=0.5, py=0.001, y=0.01, delta=0.1)

    p_ng = p0.copy()
    p_ptc = p0.copy()

    R_ng = line.compute_one_turn_matrix_finite_differences(particle_on_co=p0.copy())
    line.track(p_ng)
    line.config.XTRACK_FRINGE_FROM_PTC = True
    R_ptc = line.compute_one_turn_matrix_finite_differences(particle_on_co=p0.copy())
    line.track(p_ptc)

    assert np.isclose(p_ng.x, p_ptc.x, rtol=0, atol=1e-10)
    assert np.isclose(p_ng.px, p_ptc.px, rtol=0, atol=1e-12)
    assert np.isclose(p_ng.y, p_ptc.y, rtol=0, atol=1e-12)
    assert np.isclose(p_ng.py, p_ptc.py, rtol=0, atol=1e-12)
    assert np.isclose(p_ng.delta, p_ptc.delta, rtol=0, atol=1e-12)
    assert np.isclose(p_ng.s, p_ptc.s, rtol=0, atol=1e-12)
    assert np.isclose(p_ng.zeta, p_ptc.zeta, rtol=0, atol=1e-10)

    assert np.isclose(np.linalg.det(R_ng), 1, rtol=0, atol=1e-8) # Symplecticity check
    assert np.isclose(np.linalg.det(R_ptc), 1, rtol=0, atol=1e-8) # Symplecticity check


@for_all_test_contexts
def test_backtrack_with_bend_and_quadrupole(test_context):

    # Check bend
    b = xt.Bend(k0=0.2, h=0.1, length=1.0)
    line = xt.Line(elements=[b])
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, beta0=0.5)
    line.reset_s_at_end_turn = False
    line.build_tracker(_context=test_context)

    p0 = line.build_particles(x=0.01, px=0.02, y=0.03, py=0.04,
                            zeta=0.05, delta=0.01)
    p1 = p0.copy(_context=test_context)
    line.track(p1)
    p2 = p1.copy(_context=test_context)
    line.track(p2, backtrack=True)
    p3 = p1.copy(_context=test_context)
    line.configure_bend_model(core='full')
    line.track(p3, backtrack=True)
    p3.move(_context=xo.context_default)
    assert np.all(p3.state == -30)
    p4 = p1.copy(_context=test_context)
    line.configure_bend_model(core='expanded')
    b.num_multipole_kicks = 3
    line.track(p4, backtrack=True)
    p4.move(_context=xo.context_default)
    assert np.all(p4.state == -31)

    # Same for quadrupole
    q = xt.Quadrupole(k1=0.2, length=1.0)
    line = xt.Line(elements=[q])
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, beta0=0.5)
    line.reset_s_at_end_turn = False
    line.build_tracker(_context=test_context)
    p0 = line.build_particles(x=0.01, px=0.02, y=0.03, py=0.04,
                                zeta=0.05, delta=0.01)
    p1 = p0.copy(_context=test_context)
    line.track(p1)
    p2 = p1.copy(_context=test_context)
    line.track(p2, backtrack=True)
    p4 = p1.copy(_context=test_context)
    q.num_multipole_kicks = 4
    line.track(p4, backtrack=True)
    assert np.all(p4.state == -31)
    de = xt.DipoleEdge(e1=0.1, k=3, fint=0.3)
    line = xt.Line(elements=[de])
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, beta0=0.5)
    line.reset_s_at_end_turn = False
    line.build_tracker(_context=test_context)
    p0 = line.build_particles(x=0.01, px=0.02, y=0.03, py=0.04,
                                zeta=0.05, delta=0.01)
    p1 = p0.copy(_context=test_context)
    line.track(p1)
    p1.move(_context=xo.context_default)
    assert np.all(p1.state == 1)
    line.configure_bend_model(edge='full')
    p2 = p1.copy(_context=test_context)
    line.track(p2, backtrack=True)
    p2.move(_context=xo.context_default)
    assert np.all(p2.state == -32)