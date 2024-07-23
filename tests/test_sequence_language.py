# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2024.                 #
# ######################################### #
import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
from xtrack.xmad.xmad import Parser, XMadParseError


def test_parser_expressions():
    sequence = """
    a = 1;
    b := 2;
    c := 0;
    d := (a + b) * cos(c);  # check function parsing
    e := 2 * d;  # check deferred expressions
    test_int = 34 + 1200;  # check integer parsing
    test_float = 0. + 1.0e-2 + 43.2 + 5e2 + .6e+4;  # check float parsing
    """

    context = xo.ContextCpu()
    parser = Parser(_context=context)
    parser.parse_string(sequence)

    assert parser.vars['a'] == 1
    assert parser.vars['b'] == 2
    assert parser.vars['c'] == 0
    assert parser.vars['d'] == 3
    assert parser.vars['e'] == 6
    assert parser.vars['test_int'] == 1234
    assert parser.vars['test_float'] == 6543.21

    parser.parse_string('a = 2; c = pi;')

    assert parser.vars['a'] == 2
    assert parser.vars['d'] == -4
    assert parser.vars['e'] == -8


def test_string_errors():
    sequence = """
    x = "abc";
    yz = "eoauaeoa";
    def = "uehos";
    correct = "hello\\"escaped";
    unfinished = "hello;
    """

    context = xo.ContextCpu()
    parser = Parser(_context=context)

    with pytest.raises(XMadParseError) as e:
        parser.parse_string(sequence)

    assert 'line 2 column 9: syntax error, unexpected string' in str(e.value)
    assert 'line 3 column 10: syntax error, unexpected string' in str(e.value)
    assert 'line 4 column 11: syntax error, unexpected string' in str(e.value)
    assert 'line 5 column 15: syntax error, unexpected string' in str(e.value)
    assert 'line 6 column 18: unfinished string' in str(e.value)
    assert 'line 6 column 18: syntax error, unexpected string' in str(e.value)


@pytest.mark.xfail(reason='Not implemented yet')
def test_unfinished_string_error_wont_suppress_next():
    sequence = """
    x = "abc;
    yz = "eoauaeoa";
    """

    context = xo.ContextCpu()
    parser = Parser(_context=context)

    with pytest.raises(XMadParseError) as e:
        parser.parse_string(sequence)

    assert 'line 2 column 9: syntax error, unexpected string' in str(e.value)
    assert 'line 2 column 9: unfinished string' in str(e.value)

    # The below error is suppressed by the previous one for some reason, but
    # it should be raised as well. TODO: This is a bug that should be fixed.
    assert 'line 3 column 10: syntax error, unexpected string' in str(e.value)


def test_line():
    sequence = """
    dr_len = 2;
    bend_len = 2;
    k = 3;
    k2 = 4;

    line: sequence;
        elm_a: Drift, length = 1;
        elm_b: Drift, length := dr_len;
        elm_c: Bend,
            k0 := k,
            knl := {0, 0, k2 / bend_len},
            length := bend_len,
            +edge_entry_active,
            edge_entry_model = "linear",
            -edge_exit_active,
            model = "adaptive";
    endsequence;
    """

    context = xo.ContextCpu()
    parser = Parser(_context=context)
    parser.parse_string(sequence)

    line = parser.get_line('line')

    elm_a = line['elm_a']
    assert type(elm_a) == xt.Drift
    assert elm_a.length == 1

    elm_b = line['elm_b']
    assert type(elm_b) == xt.Drift
    assert elm_b.length == 2

    elm_c = line['elm_c']
    assert type(elm_c) == xt.Bend
    assert elm_c.length == 2
    assert np.all(elm_c.knl[:3] == [0, 0, 2])
    assert elm_c.model == 'adaptive'
    assert elm_c.edge_entry_active
    assert elm_c.edge_entry_model == 'linear'
    assert not elm_c.edge_exit_active

    assert line.element_refs['elm_c'].knl[2]._expr == line.vars['k2'] / line.vars['bend_len']

def test_multiline():
    sequence = """
    cell_l := 1;
    knl_f := 1;
    knl_d := -1;
    angle := 15 * raddeg;
    h := angle / cell_l;
    k0 := h;
    
    silly1: sequence;
        b1: Bend, k0 := k0, h := h, length := cell_l;
        qf1: Multipole, knl := {0, knl_f};
        d12u: Drift, length := cell_l / 2;
        m12: Marker;
        d12d: Drift, length := cell_l / 2;
        qd2: Multipole, knl := {0, knl_d};
    endsequence;

    silly2: sequence;
        b1: Bend, k0 := k0, h := h, length := cell_l;
        qf1: Multipole, knl := {0, knl_f};
        d12u: Drift, length := cell_l / 2;
        m12: Marker;
        d12d: Drift, length := cell_l / 2;
        qd2: Multipole, knl := {0, knl_d};
    endsequence;
    """
    target_tunes = (.21, .17)

    context = xo.ContextCpu()
    multiline = xt.Multiline.from_string(sequence, _context=context)

    line1 = multiline.silly1
    line1.particle_ref = xt.Particles(p0c=7e9, q0=1, mass0=xt.PROTON_MASS_EV)
    line1.vars['__vary_default'] = {}

    # line1.match(
    #     method='4d',
    #     vary=xt.VaryList(['knl_f', 'knl_d'], step=1e-6),
    #     targets=[xt.TargetSet(qx=target_tunes[0], qy=target_tunes[1])]
    # )
    # tw1 = line1.twiss(method='4d')
    #
    # xo.assert_allclose(tw1.qx, target_tunes[0])
    # xo.assert_allclose(tw1.qy, target_tunes[1])
    #
    # line2 = multiline.silly2
    # line2.particle_ref = line1.particle_ref
    # line2.twiss_default['reverse'] = True
    # tw2 = line2.twiss(method='4d')
    #
    # assert tw1.qx == tw2.qx
    # assert tw1.qy == tw2.qy


def test_slice_elements():
    pass
