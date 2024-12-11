# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #
import itertools
import json
import pathlib

import numpy as np
import pytest
from cpymad.madx import Madx

import xobjects as xo
import xpart as xp
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts
from xtrack.mad_loader import MadLoader
from xtrack.slicing import Strategy, Uniform


@pytest.mark.parametrize(
    'k0, k1, length, k2, use_multipole',
    [
        (-0.1, 0, 0.9, 0, False),
        (0, 0, 0.9, 0, False),
        (-0.1, 0.012, 0.9, 0, False),
        (0, 0.012, 0.8, 0, False),
        (0.15, -0.023, 0.9, 0, False),
        (0, 0.013, 1.7, 0, False),
        (0, 0, 0.9, 0.1, False),
        (-0.1, 0.003, 0.9, 0.02, False),
        (-0.1, 0.003, 0.9, 0.02, True),
    ]
)
@pytest.mark.parametrize('model', ['adaptive', 'full', 'bend-kick-bend', 'rot-kick-rot'])
@for_all_test_contexts
def test_combined_function_dipole_against_ptc(test_context, k0, k1, k2, length,
                                              use_multipole,  model):

    p0 = xp.Particles(
        mass0=xp.PROTON_MASS_EV,
        beta0=0.5,
        x=0.01,
        px=0.01,
        y=-0.005,
        py=0.001,
        zeta=0.1,
        delta=[-0.1, -0.05, 0, 0.05, 0.1],
        _context=test_context,
    )
    mad = Madx(stdout=False)
    mad.input(f"""
    ss: sequence, l={length};
        b: sbend, at={length / 2}, angle={k0 * length}, k1={k1}, k2={k2}, l={length};
    endsequence;
    beam;
    use, sequence=ss;
    """)

    ml = MadLoader(mad.sequence.ss, allow_thick=True)
    line_thick = ml.make_line()
    line_thick.config.XTRACK_USE_EXACT_DRIFTS = True # to be consistent with mad
    line_thick.build_tracker(_context=test_context)
    line_thick.configure_bend_model(core=model, edge='full')

    if use_multipole:
        line_thick['b'].knl[1] = k1 * length
        line_thick['b'].knl[2] = k2 * length
        line_thick['b'].k1 = 0
        line_thick['b'].k2 = 0

    line_core_only = xt.Line(elements=[line_thick['b'].copy()])
    line_core_only.build_tracker(_context=test_context)
    line_core_only.configure_bend_model(edge='suppressed')

    for ii in range(len(p0.x)):
        mad.input(f"""
        beam, particle=proton, pc={p0.p0c[ii] / 1e9}, sequence=ss, radiate=FALSE;

        ptc_create_universe;
        ptc_create_layout, time=true, model=1, exact=true, method=6, nst=10000;

        ptc_start, x={p0.x[ii]}, px={p0.px[ii]}, y={p0.y[ii]}, py={p0.py[ii]},
                   pt={p0.ptau[ii]}, t={p0.zeta[ii]/p0.beta0[ii]};
        ptc_track, icase=6, turns=1, onetable;
        ptc_track_end;
        ptc_end;

        """)

        mad_results = mad.table.tracksumm[-1]  # coming from PTC

        part = p0.copy(_context=test_context)
        line_thick.track(part)
        part.move(_context=xo.context_default)

        xt_tau = part.zeta/part.beta0
        xo.assert_allclose(part.x[ii], mad_results.x, rtol=0,
                           atol=(1e-11 if k1 == 0 and k2 == 0 else 5e-9))
        xo.assert_allclose(part.px[ii], mad_results.px, rtol=0,
                           atol=(1e-11 if k1 == 0 and k2 == 0 else 5e-9))
        xo.assert_allclose(part.y[ii], mad_results.y, rtol=0,
                           atol=(1e-11 if k1 == 0 and k2 == 0 else 5e-9))
        xo.assert_allclose(part.py[ii], mad_results.py, rtol=0,
                           atol=(1e-11 if k1 == 0 and k2 == 0 else 5e-9))
        xo.assert_allclose(xt_tau[ii], mad_results.t, rtol=0,
                           atol=(1e-10 if k1 == 0 and k2 == 0 else 5e-9))
        xo.assert_allclose(part.ptau[ii], mad_results.pt, atol=1e-11, rtol=0)

        part = p0.copy(_context=test_context)
        line_core_only.track(part)
        line_core_only.track(part, backtrack=True)
        part.move(_context=xo.context_default)
        p0.move(_context=xo.context_default)
        assert np.all(part.state == 1)
        xo.assert_allclose(part.x[ii], p0.x[ii], atol=1e-11, rtol=0)
        xo.assert_allclose(part.px[ii], p0.px[ii], atol=1e-11, rtol=0)
        xo.assert_allclose(part.y[ii], p0.y[ii], atol=1e-11, rtol=0)
        xo.assert_allclose(part.py[ii], p0.py[ii], atol=1e-11, rtol=0)
        xo.assert_allclose(part.zeta[ii], p0.zeta[ii], atol=1e-11, rtol=0)
        xo.assert_allclose(part.ptau[ii], p0.ptau[ii], atol=1e-11, rtol=0)

@for_all_test_contexts
def test_combined_function_dipole_expanded(test_context):

    p0 = xp.Particles(
        mass0=xp.PROTON_MASS_EV,
        beta0=0.5,
        x=0.003,
        px=0.001,
        y=-0.005,
        py=0.001,
        zeta=0.1,
        delta=[-1e-3, -0.05e-3, 0, 0.05e-3, 1e-3],
        _context=test_context,
    )

    bend = xt.Bend(k0=1e-3, h=0.9e-3, length=1, k1=0.001, knl=[0, 0, 0.02])
    line_thick = xt.Line(elements=[bend], element_names=['b'])
    line_thick.build_tracker(_context=test_context)

    line_thick.configure_bend_model(core='expanded', num_multipole_kicks=100)
    assert line_thick['b'].model == 'expanded'
    p_test = p0.copy(_context=test_context)
    line_thick.track(p_test)
    p_test.move(_context=xo.context_default)

    line_thick.configure_bend_model(core='rot-kick-rot', num_multipole_kicks=100)
    assert line_thick['b'].model == 'rot-kick-rot'
    p_ref = p0.copy(_context=test_context)
    line_thick.track(p_ref)
    p_ref.move(_context=xo.context_default)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=5e-9)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=2e-9)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=5e-9)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=2e-9)
    xo.assert_allclose(p_test.zeta, p_ref.zeta, rtol=0, atol=1e-11)
    xo.assert_allclose(p_test.ptau, p_ref.ptau, atol=1e-11, rtol=0)

    # Check backtrack
    line_thick.configure_bend_model(core='expanded')
    p_test.move(_context=test_context)
    line_thick.track(p_test, backtrack=True)
    p_test.move(_context=xo.context_default)
    p0.move(_context=xo.context_default)

    xo.assert_allclose(p_test.x, p0.x, atol=1e-11, rtol=0)
    xo.assert_allclose(p_test.px, p0.px, atol=1e-11, rtol=0)
    xo.assert_allclose(p_test.y, p0.y, atol=1e-11, rtol=0)
    xo.assert_allclose(p_test.py, p0.py, atol=1e-11, rtol=0)
    xo.assert_allclose(p_test.zeta, p0.zeta, atol=1e-11, rtol=0)
    xo.assert_allclose(p_test.ptau, p0.ptau, atol=1e-11, rtol=0)

def test_thick_bend_survey():
    circumference = 10
    rho = circumference / (2 * np.pi)
    h = 1 / rho
    k = 1 / rho

    p0 = xp.Particles(p0c=7e12, mass0=xp.PROTON_MASS_EV, x=0.7, px=-0.4, delta=0.0)

    el = xt.Bend(k0=k, h=h, length=circumference)
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
    errors = np.max(np.abs(rhos - 10 / (2 * np.pi)))
    assert errors < 2e-6


@for_all_test_contexts
@pytest.mark.parametrize('element_type', [xt.Bend])
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
    line_no_slices.configure_bend_model(core='expanded')
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
    line_with_slices.configure_bend_model(core='expanded')
    line_with_slices.track(p_with_slices, turn_by_turn_monitor='ONE_TURN_EBE')

    p_no_slices.move(_context=xo.context_default)
    p_with_slices.move(_context=xo.context_default)

    # Check that the results are the same
    for attr in ['x', 'px', 'y', 'py', 'zeta', 'delta']:
        xo.assert_allclose(
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
    mad = Madx(stdout=False)
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


    assert np.all(line.get_table().name == np.array(
        ['ss$start', 'elem', 'drift_0', 'ss$end',
       '_end_point']))
    elem = line['elem']

    # Check that the line has correct values to start with
    assert elem.model == {False: 'expanded', True: 'full'}[use_true_thick_bends]

    # Element:
    xo.assert_allclose(elem.length, 2.0, atol=1e-16)
    xo.assert_allclose(elem.k0, 0.2, atol=1e-16)
    xo.assert_allclose(elem.h, 0.05, atol=1e-16)  # h = angle / L
    xo.assert_allclose(elem.ksl, 0.0, atol=1e-16)

    xo.assert_allclose(
        elem.knl,
        np.array([0, 0, 0.8, 0, 0, 0]),  # knl = [0, 0, k2 * L, 0, 0]
        atol=1e-16,
    )

    # Edges:
    xo.assert_allclose(elem.edge_entry_fint, 0.5, atol=1e-16)
    xo.assert_allclose(elem.edge_entry_hgap, 0.6, atol=1e-16)
    xo.assert_allclose(elem.edge_entry_angle,
                      {'rbend': 0.7 + 0.1 / 2, 'sbend': 0.7}[bend_type],
                      atol=1e-16)

    xo.assert_allclose(elem.edge_exit_fint, 0.5, atol=1e-16)
    xo.assert_allclose(elem.edge_exit_hgap, 0.6, atol=1e-16)
    xo.assert_allclose(elem.edge_exit_angle,
                     {'rbend': 0.8 + 0.1 / 2, 'sbend': 0.8}[bend_type],
                      atol=1e-16)

    # Finish the test here if we are not using knobs
    if not with_knobs:
        assert 'knob_a' not in line.vars
        return
    else:
        assert 'knob_a' in line.vars

    # Change the knob values
    line.vars['knob_a'] = 2.0
    line.vars['knob_b'] = 3.0

    # Verify that the line has been adjusted correctly
    # Element:
    xo.assert_allclose(elem.length, 3.0, atol=1e-16)
    xo.assert_allclose(elem.k0, 0.4, atol=1e-16)
    xo.assert_allclose(elem.h, 0.2 / 3.0, atol=1e-16)  # h = angle / length
    xo.assert_allclose(elem.ksl, 0.0, atol=1e-16)

    xo.assert_allclose(
        elem.knl,
        np.array([0, 0, 2.4, 0, 0, 0]),  # knl = [0, 0, k2 * L, 0, 0]
        atol=1e-16,
    )

    # Edges:
    xo.assert_allclose(elem.edge_entry_fint, 1.0, atol=1e-16)
    xo.assert_allclose(elem.edge_entry_hgap, 1.2, atol=1e-16)
    xo.assert_allclose(elem.edge_entry_angle,
        {'rbend': 1.4 + 0.2 / 2, 'sbend': 1.4}[bend_type],
        atol=1e-16)
    xo.assert_allclose(elem.k0, 0.4, atol=1e-16)

    xo.assert_allclose(elem.edge_exit_fint, 1.0, atol=1e-16)
    xo.assert_allclose(elem.edge_exit_hgap, 1.2, atol=1e-16)
    xo.assert_allclose(elem.edge_exit_angle,
        {'rbend': 1.6 + 0.2 / 2, 'sbend': 1.6}[bend_type],
        atol=1e-16)


@pytest.mark.parametrize('with_knobs', [False, True])
def test_import_thick_quad_from_madx(with_knobs):
    mad = Madx(stdout=False)

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

    elem = line['elem']

    # Verify that the line has been imported correctly
    xo.assert_allclose(elem.length, 2.0, atol=1e-16)
    xo.assert_allclose(elem.k1, 0.1, atol=1e-16)
    xo.assert_allclose(elem.k1s, 0.2, atol=1e-16)

    # Finish the test here if we are not using knobs
    if not with_knobs:
        assert 'knob_a' not in line.vars
        return
    else:
        assert 'knob_a' in line.vars

    # Change the knob values
    line.vars['knob_a'] = 1.0
    line.vars['knob_b'] = 3.0

    # Verify that the line has been adjusted correctly
    xo.assert_allclose(elem.length, 3.0, atol=1e-16)
    xo.assert_allclose(elem.k1, 1.1, atol=1e-16)
    xo.assert_allclose(elem.k1s, 1.2, atol=1e-16)


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
    mad = Madx(stdout=False)
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
    line.build_tracker(compile=False)

    elems = [line[f'elem..{ii}'] for ii in range(2)]
    drifts = [line[f'drift_elem..{ii}'] for ii in range(2)]

    # Verify that the slices are correct
    for elem in elems:
        assert isinstance(elem, xt.ThinSliceBend)
        xo.assert_allclose(elem.weight, 0.5, atol=1e-16)
        xo.assert_allclose(elem._parent.length, 2.0, atol=1e-16)
        xo.assert_allclose(elem._parent.k0, 0.2, atol=1e-16)
        xo.assert_allclose(elem._parent.knl, [0., 0, 0.8, 0, 0, 0], atol=1e-16)
        xo.assert_allclose(elem._parent.ksl, 0, atol=1e-16)
        xo.assert_allclose(elem._parent.h, 0.05, atol=1e-16)

    for drift in drifts:
        xo.assert_allclose(drift._parent.length, 2., atol=1e-16)
        xo.assert_allclose(drift.weight, 1./3., atol=1e-16)

    # Finish the test here if we are not using knobs
    if not with_knobs:
        'knob_a' not in line.vars
        return
    else:
        assert 'knob_a' in line.vars

    # Change the knob values
    line.vars['knob_a'] = 2.0
    line.vars['knob_b'] = 3.0

    # Verify that the line has been adjusted correctly
    for elem in elems:
        xo.assert_allclose(elem.weight, 0.5, atol=1e-16)
        xo.assert_allclose(elem._parent.length, 3.0, atol=1e-16)
        xo.assert_allclose(elem._parent.k0, 0.4, atol=1e-16)
        xo.assert_allclose(elem._parent.knl, [0., 0, 2.4, 0, 0, 0], atol=1e-16)
        xo.assert_allclose(elem._parent.ksl, 0, atol=1e-16)
        xo.assert_allclose(elem._parent.h, 0.2/3, atol=1e-16)

        xo.assert_allclose(elem._xobject.weight, 0.5, atol=1e-16)
        xo.assert_allclose(elem._xobject._parent.length, 3.0, atol=1e-16)
        xo.assert_allclose(elem._xobject._parent.k0, 0.4, atol=1e-16)
        xo.assert_allclose(elem._xobject._parent.knl, [0., 0, 2.4, 0, 0, 0], atol=1e-16)
        xo.assert_allclose(elem._xobject._parent.ksl, 0, atol=1e-16)
        xo.assert_allclose(elem._xobject._parent.h, 0.2/3, atol=1e-16)

        assert elem._parent._buffer is line._buffer
        assert elem._xobject._parent._buffer is line._buffer

    for drift in drifts:
        xo.assert_allclose(drift._parent.length, 3, atol=1e-16)
        xo.assert_allclose(drift.weight, 1./3., atol=1e-16)

        assert drift._parent._buffer is line._buffer
        assert drift._xobject._parent._buffer is line._buffer


@pytest.mark.parametrize(
    'with_knobs',
    [True, False],
    ids=['with knobs', 'no knobs'],
)
def test_import_thick_quad_from_madx_and_slice(with_knobs):
    mad = Madx(stdout=False)
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
    line.build_tracker(compile=False)

    elems = [line[f'elem..{ii}'] for ii in range(2)]
    drifts = [line[f'drift_elem..{ii}'] for ii in range(2)]

    # Verify that the slices are correct
    for elem in elems:
        xo.assert_allclose(elem.weight, 0.5, atol=1e-16)
        xo.assert_allclose(elem._parent.length, 2.0, atol=1e-16)
        xo.assert_allclose(elem._parent.k1, 0.1, atol=1e-16)
        xo.assert_allclose(elem._parent.k1s, 0.2, atol=1e-16)

    for drift in drifts:
        xo.assert_allclose(drift._parent.length, 2., atol=1e-16)
        xo.assert_allclose(drift.weight, 1./3., atol=1e-16)

    # Finish the test here if we are not using knobs
    if not with_knobs:
        assert 'knob_a' not in line.vars
        return
    else:
        assert 'knob_a' in line.vars

    # Change the knob values
    line.vars['knob_a'] = 2.0
    line.vars['knob_b'] = 3.0

    # Verify that the line has been adjusted correctly
    for elem in elems:
        xo.assert_allclose(elem.weight, 0.5, atol=1e-16)
        xo.assert_allclose(elem._parent.length, 3.0, atol=1e-16)
        xo.assert_allclose(elem._parent.k1, 2.1, atol=1e-16)
        xo.assert_allclose(elem._parent.k1s, 2.2, atol=1e-16)

        xo.assert_allclose(elem._xobject.weight, 0.5, atol=1e-16)
        xo.assert_allclose(elem._xobject._parent.length, 3.0, atol=1e-16)
        xo.assert_allclose(elem._xobject._parent.k1, 2.1, atol=1e-16)
        xo.assert_allclose(elem._xobject._parent.k1s, 2.2, atol=1e-16)

        assert elem._parent._buffer is line._buffer
        assert elem._xobject._parent._buffer is line._buffer

    for drift in drifts:
        xo.assert_allclose(drift._parent.length, 3., atol=1e-16)
        xo.assert_allclose(drift.weight, 1./3., atol=1e-16)

        xo.assert_allclose(drift._xobject._parent.length, 3., atol=1e-16)
        xo.assert_allclose(drift._xobject.weight, 1./3., atol=1e-16)

        assert drift._parent._buffer is line._buffer
        assert drift._xobject._parent._buffer is line._buffer


@for_all_test_contexts
def test_fringe_implementations(test_context):

    fringe = xt.DipoleEdge(k=0.12, fint=100, hgap=0.035, model='full')

    line = xt.Line(elements=[fringe])
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, beta0=0.5)
    line.build_tracker(_context=test_context)

    p0 = line.build_particles(px=0.5, py=0.001, y=0.01, delta=0.1)

    p_ng = p0.copy()
    p_ptc = p0.copy()

    R_ng = line.compute_one_turn_matrix_finite_differences(particle_on_co=p0.copy())['R_matrix']
    line.track(p_ng)
    line.config.XTRACK_FRINGE_FROM_PTC = True
    R_ptc = line.compute_one_turn_matrix_finite_differences(particle_on_co=p0.copy())['R_matrix']
    line.track(p_ptc)

    p_ng.move(_context=xo.context_default)
    p_ptc.move(_context=xo.context_default)

    xo.assert_allclose(p_ng.x, p_ptc.x, rtol=0, atol=1e-10)
    xo.assert_allclose(p_ng.px, p_ptc.px, rtol=0, atol=1e-12)
    xo.assert_allclose(p_ng.y, p_ptc.y, rtol=0, atol=1e-12)
    xo.assert_allclose(p_ng.py, p_ptc.py, rtol=0, atol=1e-12)
    xo.assert_allclose(p_ng.delta, p_ptc.delta, rtol=0, atol=1e-12)
    xo.assert_allclose(p_ng.s, p_ptc.s, rtol=0, atol=1e-12)
    xo.assert_allclose(p_ng.zeta, p_ptc.zeta, rtol=0, atol=1e-10)

    xo.assert_allclose(np.linalg.det(R_ng), 1, rtol=0, atol=1e-8) # Symplecticity check
    xo.assert_allclose(np.linalg.det(R_ptc), 1, rtol=0, atol=1e-8) # Symplecticity check


@for_all_test_contexts
def test_backtrack_with_bend_quadrupole_and_cfm(test_context):

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

    p0.move(_context=xo.context_default)
    p2.move(_context=xo.context_default)
    xo.assert_allclose(p2.s, p0.s, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.x, p0.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.px, p0.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.y, p0.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.py, p0.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.zeta, p0.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.delta, p0.delta, atol=1e-15, rtol=0)

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

    p0.move(_context=xo.context_default)
    p2.move(_context=xo.context_default)
    xo.assert_allclose(p2.s, p0.s, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.x, p0.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.px, p0.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.y, p0.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.py, p0.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.zeta, p0.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.delta, p0.delta, atol=1e-15, rtol=0)

    # Same for dipole edge
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

    # Same for combined function magnet
    cfm = xt.Bend(length=1.0, k1=0.2, h=0.1)
    line = xt.Line(elements=[cfm])
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, beta0=0.5)
    line.reset_s_at_end_turn = False
    line.build_tracker(_context=test_context)

    p0 = line.build_particles(x=0.01, px=0.02, y=0.03, py=0.04,
                            zeta=0.05, delta=0.01)
    p1 = p0.copy(_context=test_context)
    line.track(p1)
    p2 = p1.copy(_context=test_context)
    line.track(p2, backtrack=True)

    p0.move(_context=xo.context_default)
    p2.move(_context=xo.context_default)
    xo.assert_allclose(p2.s, p0.s, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.x, p0.x, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.px, p0.px, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.y, p0.y, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.py, p0.py, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.zeta, p0.zeta, atol=1e-15, rtol=0)
    xo.assert_allclose(p2.delta, p0.delta, atol=1e-15, rtol=0)

def test_import_thick_with_apertures_and_slice():
    mad = Madx(stdout=False)

    mad.input("""
    k1=0.2;
    tilt=0.1;

    elm: sbend,
        k1:=k1,
        l=1,
        angle=0.1,
        tilt=0.2,
        apertype="rectellipse",
        aperture={0.1,0.2,0.11,0.22},
        aper_tol={0.1,0.2,0.3},
        aper_tilt:=tilt,
        aper_offset={0.2, 0.3};

    seq: sequence, l=1;
    elm: elm, at=0.5;
    endsequence;

    beam;
    use, sequence=seq;
    """)

    line = xt.Line.from_madx_sequence(
        sequence=mad.sequence.seq,
        allow_thick=True,
        install_apertures=True,
        deferred_expressions=True,
    )


    def _assert_eq(a, b):
        xo.assert_allclose(a, b, atol=1e-16)

    _assert_eq(line[f'elm_aper'].rot_s_rad, 0.1)
    _assert_eq(line[f'elm_aper'].shift_x, 0.2)
    _assert_eq(line[f'elm_aper'].shift_y, 0.3)
    _assert_eq(line[f'elm_aper'].max_x, 0.1)
    _assert_eq(line[f'elm_aper'].max_y, 0.2)
    _assert_eq(line[f'elm_aper'].a_squ, 0.11 ** 2)
    _assert_eq(line[f'elm_aper'].b_squ, 0.22 ** 2)

    _assert_eq(line[f'elm'].rot_s_rad, 0.2)

    line.slice_thick_elements(slicing_strategies=[Strategy(Uniform(2))])

    assert np.all(line.get_table().rows['elm_entry':'elm_exit'].name == [
        'elm_entry',                    # entry marker
        'elm_aper..0',                  # entry edge aperture
        'elm..entry_map',               # entry edge (+transform)
        'drift_elm..0',                 # drift 0
        'elm_aper..1',                  # slice 1 aperture
        'elm..0',                       # slice 0 (+transform)
        'drift_elm..1',                 # drift 1
        'elm_aper..2',                  # slice 2 aperture
        'elm..1',                       # slice 2 (+transform)
        'drift_elm..2',                 # drift 2
        'elm_aper..3',                  # exit edge aperture
        'elm..exit_map',                # exit edge (+transform)
        'elm_exit',                     # exit marker
    ])

    line.build_tracker(compile=False) # To resolve parents

    for i in range(4):
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).rot_s_rad, 0.1)
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).shift_x, 0.2)
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).shift_y, 0.3)
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).max_x, 0.1)
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).max_y, 0.2)
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).a_squ, 0.11 ** 2)
        _assert_eq(line[f'elm_aper..{i}'].resolve(line).b_squ, 0.22 ** 2)

    for i in range(2):
        _assert_eq(line[f'elm..{i}']._parent.rot_s_rad, 0.2)

@for_all_test_contexts
def test_sextupole(test_context):
    k2 = 3.
    k2s = 5.
    length = 0.4

    line_thin = xt.Line(elements=[
        xt.Drift(length=length/2),
        xt.Multipole(knl=[0., 0., k2 * length],
                    ksl=[0., 0., k2s * length],
                    length=length),
        xt.Drift(length=length/2),
    ])
    line_thin.build_tracker(_context=test_context)

    line_thick = xt.Line(elements=[
        xt.Sextupole(k2=k2, k2s=k2s, length=length),
    ])
    line_thick.build_tracker(_context=test_context)

    p = xt.Particles(
        p0c=6500e9,
        x=[-3e-2, -2e-3, 0, 1e-3, 2e-3, 3e-2],
        px=[1e-6, 2e-6,  0, 2e-6, 1e-6, 1e-6],
        y=[-2e-2, -5e-3, 0, 5e-3, -4e-3, 2e-2],
        py=[2e-6, 4e-6,  0, 2e-6, 1e-6, 1e-6],
        delta=[1e-3, 2e-3, 0, -2e-3, -1e-3, -1e-3],
        zeta=[-5e-2, -6e-3, 0, 6e-3, 5e-3, 5e-2],
    )

    p_thin = p.copy(_context=test_context)
    p_thick = p.copy(_context=test_context)

    line_thin.track(p_thin)
    line_thick.track(p_thick)

    p_thin.move(_context=xo.context_default)
    p_thick.move(_context=xo.context_default)
    xo.assert_allclose(p_thin.x, p_thick.x, rtol=0, atol=1e-14)
    xo.assert_allclose(p_thin.px, p_thick.px, rtol=0, atol=1e-14)
    xo.assert_allclose(p_thin.y, p_thick.y, rtol=0, atol=1e-14)
    xo.assert_allclose(p_thin.py, p_thick.py, rtol=0, atol=1e-14)
    xo.assert_allclose(p_thin.delta, p_thick.delta, rtol=0, atol=1e-14)
    xo.assert_allclose(p_thin.zeta, p_thick.zeta, rtol=0, atol=1e-14)

    # slicing
    Teapot = xt.slicing.Teapot
    Strategy = xt.slicing.Strategy

    line_sliced = line_thick.copy()
    line_sliced.slice_thick_elements(
        slicing_strategies=[Strategy(slicing=Teapot(5))])
    line_sliced.build_tracker(_context=test_context)

    p_sliced = p.copy(_context=test_context)
    line_sliced.track(p_sliced)

    p_sliced.move(_context=xo.context_default)
    xo.assert_allclose(p_sliced.x, p_thick.x, rtol=0, atol=5e-6)
    xo.assert_allclose(p_sliced.px, p_thick.px, rtol=0.01, atol=1e-10)
    xo.assert_allclose(p_sliced.y, p_thick.y, rtol=0, atol=5e-6)
    xo.assert_allclose(p_sliced.py, p_thick.py, rtol=0.01, atol=1e-10)
    xo.assert_allclose(p_sliced.delta, p_thick.delta, rtol=0, atol=1e-14)
    xo.assert_allclose(p_sliced.zeta, p_thick.zeta, rtol=0, atol=2e-7)

    p_thick.move(_context=test_context)
    p_thin.move(_context=test_context)
    p_sliced.move(_context=test_context)

    line_thin.track(p_thin, backtrack=True)
    line_thick.track(p_thick, backtrack=True)
    line_sliced.track(p_sliced, backtrack=True)

    p_thick.move(_context=xo.context_default)
    p_thin.move(_context=xo.context_default)
    p_sliced.move(_context=xo.context_default)

    xo.assert_allclose(p_thin.x, p.x, rtol=0, atol=1e-14)
    xo.assert_allclose(p_thin.px, p.px, rtol=0, atol=1e-14)
    xo.assert_allclose(p_thin.y, p.y, rtol=0, atol=1e-14)
    xo.assert_allclose(p_thin.py, p.py, rtol=0, atol=1e-14)
    xo.assert_allclose(p_thin.delta, p.delta, rtol=0, atol=1e-14)

    xo.assert_allclose(p_thick.x, p.x, rtol=0, atol=1e-14)
    xo.assert_allclose(p_thick.px, p.px, rtol=0, atol=1e-14)
    xo.assert_allclose(p_thick.y, p.y, rtol=0, atol=1e-14)
    xo.assert_allclose(p_thick.py, p.py, rtol=0, atol=1e-14)
    xo.assert_allclose(p_thick.delta, p.delta, rtol=0, atol=1e-14)

    xo.assert_allclose(p_sliced.x, p.x, rtol=0, atol=1e-14)
    xo.assert_allclose(p_sliced.px, p.px, rtol=0, atol=1e-14)
    xo.assert_allclose(p_sliced.y, p.y, rtol=0, atol=1e-14)
    xo.assert_allclose(p_sliced.py, p.py, rtol=0, atol=1e-14)
    xo.assert_allclose(p_sliced.delta, p.delta, rtol=0, atol=1e-14)
    xo.assert_allclose(p_sliced.zeta, p.zeta, rtol=0, atol=1e-14)

    from cpymad.madx import Madx
    mad = Madx(stdout=False)
    mad.input(f"""
        knob_a := 1.0;
        knob_b := 2.0;
        knob_l := 0.4;
        ss: sequence, l:=2 * knob_b, refer=entry;
            elem: sextupole, at=0, l:=knob_l, k2:=3*knob_a, k2s:=5*knob_b;
        endsequence;
        """)
    mad.beam()
    mad.use(sequence='ss')

    line_mad = xt.Line.from_madx_sequence(mad.sequence.ss, allow_thick=True,
                                        deferred_expressions=True)
    line_mad.build_tracker()

    elem = line_mad['elem']
    assert isinstance(elem, xt.Sextupole)
    xo.assert_allclose(elem.length, 0.4, rtol=0, atol=1e-14)
    xo.assert_allclose(elem.k2, 3, rtol=0, atol=1e-14)
    xo.assert_allclose(elem.k2s, 10, rtol=0, atol=1e-14)

    line_mad.vv['knob_a'] = 0.5
    line_mad.vv['knob_b'] = 0.6
    line_mad.vv['knob_l'] = 0.7

    xo.assert_allclose(elem.length, 0.7, rtol=0, atol=1e-14)
    xo.assert_allclose(elem.k2, 1.5, rtol=0, atol=1e-14)
    xo.assert_allclose(elem.k2s, 3.0, rtol=0, atol=1e-14)

@pytest.mark.parametrize(
    'ks, ksi, length',
    [
        # thick:
        (-0.1, 0, 0.9),
        (0, 0, 0.9),
        (0.13, 0, 1.6),
    ]
)
@for_all_test_contexts
def test_solenoid_against_madx(test_context, ks, ksi, length):
    p0 = xp.Particles(
        mass0=xp.PROTON_MASS_EV,
        beta0=[0.15, 0.5, 0.85, 0.15, 0.5, 0.85, 0.5],
        x=-0.03,
        y=0.01,
        px=-0.1,
        py=0.1,
        zeta=0.1,
        delta=[-0.8, -0.5, -0.1, 0, 0.1, 0.5, 0.8],
        _context=test_context,
    )

    mad = Madx(stdout=False)
    if length == 0:
        dr_len = 1e-11
        mad.input(f"""
        ss: sequence, l={dr_len};
            sol: solenoid, at=0, ks={ks}, ksi={ksi}, l=0;
            ! since in MAD-X we can't track a zero-length line, we put in
            ! this tiny drift here at the end of the sequence:
            dr: drift, at={dr_len / 2}, l={dr_len};
        endsequence;
        beam;
        use, sequence=ss;
        """)
    else:
        mad.input(f"""
        ss: sequence, l={length};
            sol: solenoid, at={length / 2}, ks={ks}, ksi={ksi}, l={length};
        endsequence;
        beam;
        use, sequence=ss;
        """)

    ml = MadLoader(mad.sequence.ss, allow_thick=True)
    line_thick = ml.make_line()
    line_thick.build_tracker(_context=test_context)
    line_thick.config.XTRACK_USE_EXACT_DRIFTS = True  # to be consistent with madx

    for ii in range(len(p0.x)):
        mad.input(f"""
        beam, particle=proton, pc={p0.p0c[ii] / 1e9}, sequence=ss, radiate=FALSE;

        track, onepass, onetable;
        start, x={p0.x[ii]}, px={p0.px[ii]}, y={p0.y[ii]}, py={p0.py[ii]}, \
            t={p0.zeta[ii]/p0.beta0[ii]}, pt={p0.ptau[ii]};
        run,
            turns=1,
            track_harmon=1e-15;  ! since in this test we don't care about
              ! losing particles due to t difference, we set track_harmon to
              ! something very small, to make t_max large.
        endtrack;
        """)

        mad_results = mad.table.mytracksumm[-1]

        part = p0.copy(_context=test_context)
        line_thick.track(part, _force_no_end_turn_actions=True)
        part.move(_context=xo.context_default)

        xt_tau = part.zeta/part.beta0
        xo.assert_allclose(part.x[ii], mad_results.x, atol=1e-10, rtol=0), 'x'
        xo.assert_allclose(part.px[ii], mad_results.px, atol=1e-11, rtol=0), 'px'
        xo.assert_allclose(part.y[ii], mad_results.y, atol=1e-10, rtol=0), 'y'
        xo.assert_allclose(part.py[ii], mad_results.py, atol=1e-11, rtol=0), 'py'
        xo.assert_allclose(xt_tau[ii], mad_results.t, atol=1e-9, rtol=0), 't'
        xo.assert_allclose(part.ptau[ii], mad_results.pt, atol=1e-11, rtol=0), 'pt'
        xo.assert_allclose(part.s[ii], mad_results.s, atol=1e-11, rtol=0), 's'


@for_all_test_contexts
def test_solenoid_thick_drift_like(test_context):
    solenoid = xt.Solenoid(ks=1.001e-9, length=1, _context=test_context)
    l_drift = xt.Line(elements=[xt.Drift(length=1)])
    l_drift.config.XTRACK_USE_EXACT_DRIFTS = True
    l_drift.build_tracker(_context=test_context)

    p0 = xp.Particles(
        x=0.1, px=0.2, y=0.3, py=0.4, zeta=0.5, delta=0.6,
        _context=test_context,
    )

    p_sol = p0.copy()
    solenoid.track(p_sol)

    p_drift = p0.copy()
    l_drift.track(p_drift)

    p_sol.move(_context=xo.context_default)
    p_drift.move(_context=xo.context_default)

    xo.assert_allclose(p_sol.x, p_drift.x, atol=1e-9)
    xo.assert_allclose(p_sol.px, p_drift.px, atol=1e-9)
    xo.assert_allclose(p_sol.y, p_drift.y, atol=1e-9)
    xo.assert_allclose(p_sol.py, p_drift.py, atol=1e-9)
    xo.assert_allclose(p_sol.zeta, p_drift.zeta, atol=1e-9)
    xo.assert_allclose(p_sol.delta, p_drift.delta, atol=1e-9)


@for_all_test_contexts
@pytest.mark.parametrize(
    'length, expected', [
        (2 * np.pi / np.sqrt(2), [1, 0, -1, 0, 2*np.pi, 0]),
        (np.pi / np.sqrt(2), [0, 0.5, 0, 0.5, np.pi, 0]),
    ],
)
def test_solenoid_thick_analytic(test_context, length, expected):
    solenoid = xt.Solenoid(
        ks=1,
        length=length,
        _context=test_context,
    )

    p0 = xp.Particles(
        x=1,
        px=0,
        y=-1,
        py=0,
        _context=test_context,
    )

    p_sol = p0.copy()
    solenoid.track(p_sol)

    p_sol.move(_context=xo.context_default)

    xo.assert_allclose(p_sol.x, expected[0], atol=1e-9)
    xo.assert_allclose(p_sol.px, expected[1], atol=1e-9)
    xo.assert_allclose(p_sol.y, expected[2], atol=1e-9)
    xo.assert_allclose(p_sol.py, expected[3], atol=1e-9)
    delta_ell = (p_sol.s - p_sol.zeta) * p_sol.rvv
    xo.assert_allclose(delta_ell, expected[4], atol=1e-9)
    xo.assert_allclose(p_sol.delta, expected[5], atol=1e-9)
    xo.assert_allclose(p_sol.s, length, atol=1e-9)

@for_all_test_contexts
@pytest.mark.parametrize(
    'backtrack', [False, True],
)
def test_solenoid_with_mult_kicks(test_context, backtrack):
    length = 2
    ks = 1.5
    num_kicks = 2
    knl = np.array([0.1, 0.4, 0.5])
    ksl = np.array([0.2, 0.3, 0.6])

    solenoid_with_kicks = xt.Solenoid(
        length=length,
        ks=ks,
        num_multipole_kicks=num_kicks,
        knl=knl,
        ksl=ksl,
    )

    line_ref = xt.Line(
        elements=[
            xt.Solenoid(length=length / (num_kicks + 1), ks=ks),
            xt.Multipole(knl=knl / num_kicks, ksl=ksl / num_kicks),
            xt.Solenoid(length=length / (num_kicks + 1), ks=ks),
            xt.Multipole(knl=knl / num_kicks, ksl=ksl / num_kicks),
            xt.Solenoid(length=length / (num_kicks + 1), ks=ks),
        ],
        element_names=[
            'sol_0', 'kick_0', 'sol_1', 'kick_1', 'sol_2',
        ]
    )

    line_test = xt.Line(elements=[
            solenoid_with_kicks,
        ],
        element_names=['sol'],
    )

    coords = np.linspace(-0.05, 0.05, 10)
    coords_6d = np.array(list(itertools.product(*(coords,) * 6))).T

    p0 = xt.Particles(
        _context=test_context,
        x=coords_6d[0],
        px=coords_6d[1],
        y=coords_6d[2],
        py=coords_6d[3],
        zeta=coords_6d[4],
        delta=coords_6d[5],
    )
    p_ref = p0.copy()
    p_test = p0.copy()

    line_ref.build_tracker(_context=test_context)
    line_ref.track(p_ref, num_turns=1, backtrack=backtrack)

    line_test.build_tracker(_context=test_context)
    line_test.track(p_test, num_turns=1, backtrack=backtrack)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.delta, p_ref.delta, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.pzeta, p_ref.pzeta, rtol=0, atol=1e-13)


@for_all_test_contexts
def test_solenoid_shifted_and_rotated_multipolar_kick(test_context):
    ks = 0.9
    length = 1
    knl = [0.1, 0.4, 0.5]
    ksl = [0.2, 0.3, 0.6]
    mult_rot_y_rad = 0.2
    mult_shift_x = 0.3

    solenoid = xt.Solenoid(
        ks=ks,
        length=length,
        knl=knl,
        ksl=ksl,
        num_multipole_kicks=3,
        mult_rot_y_rad=mult_rot_y_rad,
        mult_shift_x=mult_shift_x,
    )

    solenoid_no_kick = xt.Solenoid(ks=0.9, length=0.25)
    kick = xt.Multipole(knl=np.array(knl) / 3, ksl=np.array(ksl) / 3)

    line_test = xt.Line(elements=[solenoid])
    line_test.build_tracker(_context=test_context)

    elements_ref = [solenoid_no_kick] + 3 * [
        xt.XYShift(dx=mult_shift_x),
        xt.YRotation(angle=np.rad2deg(-mult_rot_y_rad)),
        kick,
        xt.YRotation(angle=np.rad2deg(mult_rot_y_rad)),
        xt.XYShift(dx=-mult_shift_x),
        solenoid_no_kick
    ]
    line_ref = xt.Line(elements=elements_ref)
    line_ref.build_tracker(_context=test_context)

    p0 = xt.Particles(x=1e-2, px=-2e-4, y=-2e-2, py=3e-4, zeta=1e-2, delta=1e-3)
    p_test = p0.copy(_context=test_context)
    p_ref = p0.copy(_context=test_context)

    line_test.track(p_test)
    line_ref.track(p_ref)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-16)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-16)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-16)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-16)
    xo.assert_allclose(p_test.zeta, p_ref.zeta, rtol=0, atol=1e-16)
    xo.assert_allclose(p_test.delta, p_ref.delta, rtol=0, atol=1e-16)


@pytest.mark.parametrize('shift_x', (0, 1e-3))
@pytest.mark.parametrize('shift_y', (0, 1e-3))
@pytest.mark.parametrize('test_element_name', ('Bend', 'Quadrupole', 'Sextupole'))
def test_solenoid_multipole_shifts(shift_x, shift_y, test_element_name):
    ################################################################################
    # User Parameters
    ################################################################################
    N_SLICES = int(1E3)

    BETX = 100E-3
    BETY = 1E-3
    PX0 = 0

    KS = 0.00
    K0 = 1E-3
    K1 = 1E-3
    K2 = 1E-3

    ################################################################################
    # Build Test Elements
    ################################################################################
    drift0 = xt.Drift(length=1)
    drift1 = xt.Drift(length=1)

    bend = xt.Bend(length=1, k0=K0)
    quad = xt.Quadrupole(length=1, k1=K1)
    sext = xt.Sextupole(length=1, k2=K2)

    bend_sol = xt.Solenoid(length=1 / N_SLICES, ks=KS,
                           knl=[K0 * (1 / N_SLICES), 0, 0], num_multipole_kicks=1)
    quad_sol = xt.Solenoid(length=1 / N_SLICES, ks=KS,
                           knl=[0, K1 * (1 / N_SLICES), 0], num_multipole_kicks=1)
    sext_sol = xt.Solenoid(length=1 / N_SLICES, ks=KS,
                           knl=[0, 0, K2 * (1 / N_SLICES)], num_multipole_kicks=1)

    ################################################################################
    # Comparisons
    ################################################################################
    test_element, test_sol = {
        'Bend': (bend, bend_sol),
        'Quadrupole': (quad, quad_sol),
        'Sextupole': (sext, sext_sol),
    }[test_element_name]

    ########################################
    # Build Lines
    ########################################
    line = xt.Line(
        elements=[drift0] + [test_element] + [drift0],
        particle_ref=xt.Particles(p0c=1E9, mass0=xt.ELECTRON_MASS_EV))
    line.configure_bend_model(edge='suppressed')

    sol_line = xt.Line(
        elements=[drift1] + [test_sol] * N_SLICES + [drift1],
        particle_ref=xt.Particles(p0c=1E9, mass0=xt.ELECTRON_MASS_EV))

    # Slice test line
    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(slicing=xt.Uniform(N_SLICES, mode='thin'), element_type=xt.Bend),
            xt.Strategy(slicing=xt.Uniform(N_SLICES, mode='thin'), element_type=xt.Quadrupole),
            xt.Strategy(slicing=xt.Uniform(N_SLICES, mode='thin'), element_type=xt.Sextupole)])

    ########################################
    # Test and plot with shifts
    ########################################
    test_element.shift_x = shift_x
    test_element.shift_y = shift_y
    test_sol.mult_shift_x = shift_x
    test_sol.mult_shift_y = shift_y

    tw = line.twiss(
        _continue_if_lost=True,
        start=xt.START,
        end=xt.END,
        betx=BETX,
        bety=BETY,
        px=PX0)
    tw_sol = sol_line.twiss(
        _continue_if_lost=True,
        start=xt.START,
        end=xt.END,
        betx=BETX,
        bety=BETY,
        px=PX0)

    ########################################
    # Assertions
    ########################################
    assert np.isclose(tw.x[-1], tw_sol.x[-1], rtol=1E-6)
    assert np.isclose(tw.y[-1], tw_sol.y[-1], rtol=1E-6)


def test_solenoid_multipole_rotations():
    N_SLICES = int(1E2)
    K0 = 1E-3
    L_SOL = 1
    XING_RAD = 1E-3

    BETX = 100E-3
    BETY = 1E-3

    ########################################
    # Build Environment
    ########################################
    env = xt.Environment(particle_ref=xt.Particles(p0c=1E9))

    ########################################
    # Line (beamline frame)
    ########################################
    bl_components_in = [env.new('bl_drift0', xt.Drift, length=1)]
    bl_components_out = [env.new('bl_drift1', xt.Drift, length=1)]

    bl_components_sol = [
        env.new(f'bl_sol.{i}', xt.Solenoid,
                length=(L_SOL / N_SLICES),
                ks=0,
                knl=[K0 * (L_SOL / N_SLICES), 0, 0],
                num_multipole_kicks=1)
        for i in range(N_SLICES)]

    bl_line = env.new_line(
        components=bl_components_in + bl_components_sol + bl_components_out)

    ########################################
    # Line (horizontal rotated frame)
    ########################################
    hrot_components_in = [
        env.new('hrot_drift0', xt.Drift, length=1),
        env.new('hshift_in', xt.XYShift, dx=np.sin(XING_RAD) * L_SOL / 2),
        env.new('hrot_in', xt.YRotation, angle=-np.rad2deg(XING_RAD))]

    hrot_components_out = [
        env.new('hrot_out', xt.YRotation, angle=np.rad2deg(XING_RAD)),
        env.new('hshift_out', xt.XYShift, dx=np.sin(XING_RAD) * L_SOL / 2),
        env.new('hrot_drift1', xt.Drift, length=1)]

    hrot_components_sol = [
        env.new(f'hrot_sol.{i}', xt.Solenoid,
                length=(L_SOL / N_SLICES) * np.cos(XING_RAD),
                ks=0,
                knl=[K0 * (L_SOL / N_SLICES), 0, 0],
                num_multipole_kicks=1,
                mult_rot_y_rad=XING_RAD,
                mult_shift_x=np.sin(XING_RAD) * L_SOL * (i / N_SLICES - 1 / 2))
        for i in range(N_SLICES)]

    hrot_line = env.new_line(
        components=hrot_components_in + hrot_components_sol + hrot_components_out)

    ########################################
    # Line (vertical rotated frame)
    ########################################
    vrot_components_in = [
        env.new('vrot_drift0', xt.Drift, length=1),
        env.new('vshift_in', xt.XYShift, dy=np.sin(XING_RAD) * L_SOL / 2),
        env.new('vrot_in', xt.XRotation, angle=np.rad2deg(XING_RAD))]
    # TODO: Minus sign difference here as still inconsistent definition with XRotation and YRotation
    vrot_components_out = [
        env.new('vrot_out', xt.XRotation, angle=-np.rad2deg(XING_RAD)),
        env.new('vshift_out', xt.XYShift, dy=np.sin(XING_RAD) * L_SOL / 2),
        env.new('vrot_drift1', xt.Drift, length=1)]

    vrot_components_sol = [
        env.new(f'vrot_sol.{i}', xt.Solenoid,
                length=(L_SOL / N_SLICES) * np.cos(XING_RAD),
                ks=0,
                knl=[K0 * (L_SOL / N_SLICES), 0, 0],
                num_multipole_kicks=1,
                mult_rot_x_rad=XING_RAD,
                mult_shift_y=np.sin(XING_RAD) * L_SOL * (i / N_SLICES - 1 / 2))
        for i in range(N_SLICES)]

    vrot_line = env.new_line(
        components=vrot_components_in + vrot_components_sol + vrot_components_out)

    ################################################################################
    # Comparisons
    ################################################################################
    bl_twiss = bl_line.twiss(
        method='4d',
        start=xt.START,
        end=xt.END,
        betx=BETX,
        bety=BETY)

    hrot_twiss = hrot_line.twiss(
        method='4d',
        start=xt.START,
        end=xt.END,
        betx=BETX,
        bety=BETY)

    vrot_twiss = vrot_line.twiss(
        method='4d',
        start=xt.START,
        end=xt.END,
        betx=BETX,
        bety=BETY)

    ################################################################################
    # Test Assertions
    ################################################################################
    # Tolerances lower for derivative quantities (alfx, alfy, dpx, dpy)
    assert np.isclose(bl_twiss['x'][-1], hrot_twiss['x'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['y'][-1], hrot_twiss['y'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['betx'][-1], hrot_twiss['betx'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['bety'][-1], hrot_twiss['bety'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['alfx'][-1], hrot_twiss['alfx'][-1], rtol=1E-4)
    assert np.isclose(bl_twiss['alfy'][-1], hrot_twiss['alfy'][-1], rtol=1E-4)
    assert np.isclose(bl_twiss['dx'][-1], hrot_twiss['dx'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['dy'][-1], hrot_twiss['dy'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['dpx'][-1], hrot_twiss['dpx'][-1], rtol=1E-4)
    assert np.isclose(bl_twiss['dpy'][-1], hrot_twiss['dpy'][-1], rtol=1E-4)
    assert np.isclose(bl_twiss['mux'][-1], hrot_twiss['mux'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['muy'][-1], hrot_twiss['muy'][-1], rtol=1E-6)

    assert np.isclose(bl_twiss['x'][-1], vrot_twiss['x'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['y'][-1], vrot_twiss['y'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['betx'][-1], vrot_twiss['betx'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['bety'][-1], vrot_twiss['bety'][-1], rtol=1E-4)
    assert np.isclose(bl_twiss['alfx'][-1], vrot_twiss['alfx'][-1], rtol=1E-4)
    assert np.isclose(bl_twiss['alfy'][-1], vrot_twiss['alfy'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['dx'][-1], vrot_twiss['dx'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['dy'][-1], vrot_twiss['dy'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['dpx'][-1], vrot_twiss['dpx'][-1], rtol=1E-4)
    assert np.isclose(bl_twiss['dpy'][-1], vrot_twiss['dpy'][-1], rtol=1E-4)
    assert np.isclose(bl_twiss['mux'][-1], vrot_twiss['mux'][-1], rtol=1E-6)
    assert np.isclose(bl_twiss['muy'][-1], vrot_twiss['muy'][-1], rtol=1E-6)


@pytest.mark.parametrize(
    'radiation_mode,config',
    [
        ('mean', {}),
        ('mean', {'XTRACK_SYNRAD_KICK_SAME_AS_FIRST': True}),
        ('mean', {'XTRACK_SYNRAD_SCALE_SAME_AS_FIRST': True}),
        ('quantum', {}),
    ],
)
def test_drift_like_solenoid_with_kicks_radiation(radiation_mode, config):
    test_context = xo.ContextCpu()

    config['XTRACK_USE_EXACT_DRIFTS'] = True
    knl = [0.1, 0.4, 0.5]
    ksl = [0.2, 0.3, 0.6]

    line_test = xt.Line(elements=[
        xt.Drift(length=0.5),
        xt.Multipole(knl=knl, ksl=ksl),
        xt.Drift(length=0.5),
    ])

    line_ref = xt.Line(elements=[
        xt.Solenoid(ks=0, length=1, knl=knl, ksl=ksl, num_multipole_kicks=1)
    ])

    coords = np.linspace(-0.05, 0.05, 10)
    coords_6d = np.array(list(itertools.product(*(coords,) * 6))).T

    p0 = xt.Particles(
        _context=test_context,
        x=coords_6d[0],
        px=coords_6d[1],
        y=coords_6d[2],
        py=coords_6d[3],
        zeta=coords_6d[4],
        delta=coords_6d[5],
    )
    p0._init_random_number_generator(seeds=np.arange(len(coords_6d[0]), dtype=int))
    p_ref = p0.copy()
    p_test = p0.copy()

    line_ref.build_tracker(_context=test_context)
    line_ref.configure_radiation(model=radiation_mode)
    line_ref.config.update(config)
    line_ref.track(p_ref, num_turns=1)

    line_test.build_tracker(_context=test_context)
    line_test.configure_radiation(model=radiation_mode)
    line_test.config.update(config)
    line_test.track(p_test, num_turns=1)

    xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.delta, p_ref.delta, rtol=0, atol=1e-13)
    xo.assert_allclose(p_test.pzeta, p_ref.pzeta, rtol=0, atol=1e-13)


@pytest.mark.parametrize(
    'radiation_mode,config',
    [
        ('mean', {}),
        ('mean', {'XTRACK_SYNRAD_KICK_SAME_AS_FIRST': True}),
        ('mean', {'XTRACK_SYNRAD_SCALE_SAME_AS_FIRST': True}),
        ('quantum', {}),
    ],
)
def test_solenoid_with_kicks_radiation(radiation_mode, config):
    test_context = xo.ContextCpu()

    config['XTRACK_USE_EXACT_DRIFTS'] = True

    ks = 0.4
    l = 1.1
    knl = [0.1, 0.4, 0.5]
    ksl = [0.2, 0.3, 0.6]

    sol_ref = xt.Solenoid(ks=ks, length=l)
    sol_1 = xt.Solenoid(ks=ks, length=l, knl=knl, ksl=ksl, num_multipole_kicks=1)
    sol_3 = xt.Solenoid(ks=ks, length=l, knl=knl, ksl=ksl, num_multipole_kicks=3)

    line_ref = xt.Line(elements=[sol_ref])
    line_1 = xt.Line(elements=[sol_1])
    line_3 = xt.Line(elements=[sol_3])

    coords = np.linspace(-0.05, 0.05, 10)
    coords_6d = np.array(list(itertools.product(*(coords,) * 6))).T

    p0 = xt.Particles(
        _context=test_context,
        x=coords_6d[0],
        px=coords_6d[1],
        y=coords_6d[2],
        py=coords_6d[3],
        zeta=coords_6d[4],
        delta=coords_6d[5],
    )
    p0._init_random_number_generator(seeds=np.arange(len(coords_6d[0]), dtype=int))
    p_ref = p0.copy()
    p_1 = p0.copy()
    p_3 = p0.copy()

    line_ref.build_tracker(_context=test_context)
    line_ref.track(p_ref, num_turns=1)

    line_1.build_tracker(_context=test_context)
    line_1.configure_radiation(model=radiation_mode)
    line_1.config.update(config)
    line_1.track(p_1, num_turns=1)

    line_3.build_tracker(_context=test_context)
    line_3.configure_radiation(model=radiation_mode)
    line_3.config.update(config)
    line_3.track(p_3, num_turns=1)

    d_delta_ref = p_ref.delta - p0.delta
    d_delta_1 = p_1.delta - p0.delta
    d_delta_3 = p_3.delta - p0.delta

    xo.assert_allclose(
        d_delta_1 - d_delta_ref, d_delta_3 - d_delta_ref, rtol=0.01, atol=1e-15
    )


@for_all_test_contexts
def test_skew_quadrupole(test_context):
    k1 = 1.0
    k1s = 2.0

    length = 0.5

    quad = xt.Quadrupole(k1=k1, k1s=k1s, length=length, _context=test_context)

    n_slices = 1000
    ele_thin = []
    for ii in range(n_slices):
        ele_thin.append(xt.Drift(length=length/n_slices/2))
        ele_thin.append(xt.Multipole(knl=[0, k1 * length/n_slices],
                                     ksl=[0, k1s * length/n_slices]))
        ele_thin.append(xt.Drift(length=length/n_slices/2))
    lref = xt.Line(ele_thin)
    lref.build_tracker(_context=test_context)

    p_test = xt.Particles(gamma0=1.2, x=0.1, y=0.2, delta=0.5,
                          _context=test_context)
    p_ref = p_test.copy()

    quad.track(p_test)
    lref.track(p_ref)

    p_test.move(_context=xo.context_default)
    p_ref.move(_context=xo.context_default)

    xo.assert_allclose(p_test.x, p_ref.x, atol=1e-8, rtol=0)
    xo.assert_allclose(p_test.px, p_ref.px, atol=5e-8, rtol=0)
    xo.assert_allclose(p_test.y, p_ref.y, atol=1e-8, rtol=0)
    xo.assert_allclose(p_test.py, p_ref.py, atol=5e-8, rtol=0)
    xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-8, rtol=0)
    xo.assert_allclose(p_test.delta, p_ref.delta, atol=5e-8, rtol=0)

@for_all_test_contexts
def test_octupole(test_context):

    k3 = 1.0
    k3s = 2.0

    length = 0.5

    oct = xt.Octupole(k3=k3, k3s=k3s, length=length, _context=test_context)

    ele_thin = []
    ele_thin.append(xt.Drift(length=length/2))
    ele_thin.append(xt.Multipole(knl=[0, 0, 0, k3 * length],
                                ksl=[0, 0, 0, k3s * length]))
    ele_thin.append(xt.Drift(length=length/2))
    lref = xt.Line(ele_thin)
    lref.build_tracker(_context=test_context)

    p_test = xt.Particles(gamma0=1.2, x=0.1, y=0.2, delta=0.5, _context=test_context)
    p_ref = p_test.copy()

    oct.track(p_test)
    lref.track(p_ref)

    p_test.move(_context=xo.context_default)
    p_ref.move(_context=xo.context_default)

    xo.assert_allclose(p_test.x, p_ref.x, atol=1e-12, rtol=0)
    xo.assert_allclose(p_test.px, p_ref.px, atol=1e-12, rtol=0)
    xo.assert_allclose(p_test.y, p_ref.y, atol=1e-12, rtol=0)
    xo.assert_allclose(p_test.py, p_ref.py, atol=1e-12, rtol=0)
    xo.assert_allclose(p_test.zeta, p_ref.zeta, atol=1e-12, rtol=0)
    xo.assert_allclose(p_test.delta, p_ref.delta, atol=1e-12, rtol=0)

@for_all_test_contexts
@pytest.mark.parametrize(
    'element,kn_param_name',
    [('quadrupole', 'k1'), ('sextupole', 'k2'), ('octupole', 'k3')]
)
def test_multipole_fringe(test_context, element, kn_param_name):
    ref_dir = pathlib.Path(__file__).parent.joinpath('../test_data/fringe_vs_madng')

    with open(ref_dir / f'{element}_fringe.json', 'r') as f:
        fringe_effect_madng = json.load(f)

    with open(ref_dir / 'initial_particles.json', 'r') as f:
        initial_particles = json.load(f)
        p0 = xt.Particles.from_dict(initial_particles, _context=test_context)

    length = 0.01
    knl = 1

    result = {}
    element_class = getattr(xt, element.capitalize())

    for has_fringe in (True, False):
        _p = p0.copy()
        line = xt.Line(
            elements=[
                element_class(
                    length=length,
                    edge_entry_active=has_fringe,
                    edge_exit_active=has_fringe,
                    **{kn_param_name: knl / length},
                ),
            ],
            element_names=['q'],
        )
        line.build_tracker(_context=test_context)
        line.track(_p)
        _p.move(_context=xo.ContextCpu())
        _p.sort()

        result[has_fringe] = _p

    coords_xtrack_fringe = np.array([
        result[True].x,
        result[True].px,
        result[True].y,
        result[True].py,
        result[True].zeta / result[True].beta0,
        result[True].ptau]
    )
    coords_xtrack_no_fringe = np.array([
        result[False].x,
        result[False].px,
        result[False].y,
        result[False].py,
        result[False].zeta / result[False].beta0,
        result[False].ptau],
    )

    fringe_effect_xtrack = coords_xtrack_fringe - coords_xtrack_no_fringe
    xo.assert_allclose(fringe_effect_madng[0], fringe_effect_xtrack[0], atol=1.01e-16, rtol=1e-3)  # x
    xo.assert_allclose(fringe_effect_madng[1], fringe_effect_xtrack[1], atol=1e-16, rtol=1e-2)  # px
    xo.assert_allclose(fringe_effect_madng[2], fringe_effect_xtrack[2], atol=1.01e-16, rtol=1e-3)  # y
    xo.assert_allclose(fringe_effect_madng[3], fringe_effect_xtrack[3], atol=1e-16, rtol=1.01e-2)  # py
    xo.assert_allclose(fringe_effect_madng[4], fringe_effect_xtrack[4], atol=1.01e-15, rtol=1.01e-2)  # t
    xo.assert_allclose(fringe_effect_madng[5], fringe_effect_xtrack[5], atol=0, rtol=0)  # pt
