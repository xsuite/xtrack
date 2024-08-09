# copyright ############################### #
# This file is part of the Xpart Package.   #
# Copyright (c) CERN, 2024.                 #
# ######################################### #
import numpy as np
import pytest
import re

import xobjects as xo
import xtrack as xt
from xtrack.sequence.parser import Parser, ParseError

def _normalize_code(code_string, remove_empty_lines=True, ignore_config=True):
    if ignore_config:
        code_string = re.sub(r'config.*?;\n', '', code_string, flags=re.DOTALL)

    lines = code_string.split('\n')  # split into lines
    lines = map(str.strip, lines)  # remove meaningless whitespace

    if remove_empty_lines:
        lines = filter(bool, lines)

    lines = filter(lambda l: not l.startswith('#'), lines)  # remove lines w/comments
    lines = map(lambda x: x.split('#')[0].strip(), lines)  # remove comments

    lines = filter(lambda l: l != 't_turn_s = 0.0;', lines)

    return list(lines)


def _dict_compare(d1, d2, path='.', atol=1e-16, rtol=1e-16):
    common_keys = set(d1.keys()) & set(d2.keys())
    for key in common_keys:
        if type(d1[key]) != type(d2[key]):
            raise AssertionError(f'{path}[{key}] type mismatch: {type(d1[key])} != {type(d2[key])}')
        if isinstance(d1[key], dict):
            _dict_compare(d1[key], d2[key], f'{path}["{key}"]')
        if isinstance(d1[key], np.ndarray):
            xo.assert_allclose(d1[key], d2[key], atol=atol, rtol=rtol)
        if isinstance(d1[key], list):
            for i, (v1, v2) in enumerate(zip(d1[key], d2[key])):
                if isinstance(v1, dict):
                    _dict_compare(v1, v2, f'{path}["{key}"][{i}]')
                if isinstance(v1, np.ndarray):
                    xo.assert_allclose(v1, v2, atol=atol, rtol=rtol)
                else:
                    assert v1 == v2


def test_parser_expressions():
    sequence = """
    a = 1;
    b = 2;
    c = 0;
    d = (a + b) * cos(c);  # check function parsing
    e = 2 * d;  # check deferred expressions
    test_int = 34 + 1200;  # check integer parsing
    test_float = 0. + 1.0e-2 + 43.2 + 5e2 + .6e+4;  # check float parsing
    """

    parser = Parser(_context=xo.context_default)
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
    x = 12;
    x = 13;
    unfinished = "hello;
    """

    parser = Parser(_context=xo.context_default)

    with pytest.raises(ParseError) as e:
        parser.parse_string(sequence)

    assert 'line 2 column 9: syntax error, unexpected string' in str(e.value)
    assert 'line 3 column 10: syntax error, unexpected string' in str(e.value)
    assert 'line 4 column 11: syntax error, unexpected string' in str(e.value)
    assert 'line 5 column 15: syntax error, unexpected string' in str(e.value)
    assert 'line 7 column 5: redefinition of the variable `x`' in str(e.value)
    assert 'line 8 column 18: unfinished string' in str(e.value)
    assert 'line 8 column 18: syntax error, unexpected string' in str(e.value)


@pytest.mark.xfail(reason='Not implemented yet')
def test_unfinished_string_error_wont_suppress_next():
    sequence = """
    x = "abc;
    yz = "eoauaeoa";
    """

    parser = Parser(_context=xo.context_default)

    with pytest.raises(ParseError) as e:
        parser.parse_string(sequence)

    assert 'line 2 column 9: syntax error, unexpected string' in str(e.value)
    assert 'line 2 column 9: unfinished string' in str(e.value)

    # The below error is suppressed by the previous one for some reason, but
    # it should be raised as well. TODO: This is a bug that should be fixed.
    assert 'line 3 column 10: syntax error, unexpected string' in str(e.value)


def test_minimal():
    sequence = """\
    x = 1;
    line: beamline;
        elm_a: Drift, length = x;
    endbeamline;
    """

    parser = Parser(_context=xo.context_default)
    parser.parse_string(sequence)

    _ = parser.get_line('line')


def test_line():
    sequence = """
    dr_len = 2;
    bend_len = 2;
    k = 3;
    k2 = 4;

    line: beamline;
        elm_a: Drift, length = 1;
        elm_b: Drift, length = dr_len;
        elm_c: Bend,
            k0 = k,
            knl = {0, 0, k2 / bend_len},
            length = bend_len,
            edge_entry_active,
            edge_entry_model = "linear",
            -edge_exit_active,
            model = "adaptive";
    endbeamline;
    """

    parser = Parser(_context=xo.context_default)
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


def test_particle_ref():
    sequence = """\
    x = 1;
    line: beamline;
        particle_ref, p0c = 7e9, q0 = 1, mass0 = pmass;
        elm_a: Drift, length = x;
    endbeamline;
    """

    parser = Parser(_context=xo.context_default)
    parser.parse_string(sequence)

    line = parser.get_line('line')

    p_ref = line.particle_ref
    xo.assert_allclose(p_ref.p0c, 7e9, atol=1e-16)
    xo.assert_allclose(p_ref.q0, 1, atol=1e-16)
    xo.assert_allclose(p_ref.mass0, xt.PROTON_MASS_EV, atol=1e-16)

    assert line['elm_a'].length == 1


def test_multiline_simple_twiss():
    sequence = """\
    cell_l = 1;
    knl_f = 1;
    knl_d = -1;
    angle = 15 * raddeg;
    h = angle / cell_l;
    k0 = h;
    
    silly1: beamline;
        particle_ref, p0c = 7e9, q0 = 1, mass0 = pmass;
        b1: Bend, k0 = k0, h = h, length = cell_l;
        qf1: Multipole, knl = {0, knl_f};
        d12u: Drift, length = cell_l / 2;
        m12: Marker;
        d12d: Drift, length = cell_l / 2;
        qd2: Multipole, knl = {0, knl_d};
    endbeamline;

    silly2: beamline;
        particle_ref, p0c = 7e9, q0 = 1, mass0 = pmass;
        twiss_default, reverse;
        b1: Bend, k0 = k0, h = h, length = cell_l;
        qf1: Multipole, knl = {0, knl_f};
        d12u: Drift, length = cell_l / 2;
        m12: Marker;
        d12d: Drift, length = cell_l / 2;
        qd2: Multipole, knl = {0, knl_d};
    endbeamline;
    """
    target_tunes = (.21, .17)

    multiline = xt.Multiline.from_string(sequence, _context=xo.context_default)

    line1 = multiline.silly1
    assert not line1.twiss_default.get('reverse', None)

    xo.assert_allclose(line1.particle_ref.p0c, 7e9, atol=1e-16)
    xo.assert_allclose(line1.particle_ref.q0, 1, atol=1e-16)
    xo.assert_allclose(line1.particle_ref.mass0, xt.PROTON_MASS_EV, atol=1e-16)

    line1.match(
        method='4d',
        vary=xt.VaryList(['knl_f', 'knl_d'], step=1e-6),
        targets=[xt.TargetSet(qx=target_tunes[0], qy=target_tunes[1])],
    )

    tw1 = line1.twiss(method='4d')

    xo.assert_allclose(tw1.qx, target_tunes[0])
    xo.assert_allclose(tw1.qy, target_tunes[1])

    line2 = multiline.silly2

    assert line2.twiss_default['reverse']

    xo.assert_allclose(line2.particle_ref.p0c, 7e9, atol=1e-16)
    xo.assert_allclose(line2.particle_ref.q0, 1, atol=1e-16)
    xo.assert_allclose(line2.particle_ref.mass0, xt.PROTON_MASS_EV, atol=1e-16)

    tw2 = line2.twiss(method='4d')

    assert tw1.qx == tw2.qx
    assert tw1.qy == tw2.qy


def test_multiline_read_and_dump(tmp_path):
    sequence = """\
        cell_l = 1.0;
        knl_f = 1.0;
        knl_d = -1.0;
        angle = 0.1;
        h = (angle / cell_l);
        k0 = h;
        
        silly1: beamline;
            particle_ref,
                # t_sim will always be there (calculated by the line by default)
                # so we override it so that the assertion is nicer
                t_sim = 10.0;

            b1: Bend, 
                length = cell_l,
                k0 = k0,
                h = h,
                model = "adaptive",
                edge_entry_model = "linear",
                edge_exit_model = "linear",
                order = 5;
            qf1: Multipole, knl = {0.0, knl_f};
            d12u: Drift, length = (cell_l / 2.0);
            m12: Marker;
            d12d: Drift, length = (cell_l / 2.0);
            qd2: Multipole, knl = {0.0, knl_d};
        endbeamline;
        
        silly2: beamline;
            particle_ref,
                p0c = 7100000000.0,  # not "7.1e9" to match the output of writer
                q0 = 2.0,
                t_sim = 10.0;  # ditto

            b1: Bend, 
                length = cell_l,
                k0 = k0,
                h = h,
                model = "adaptive",
                edge_entry_model = "linear",
                edge_exit_model = "linear",
                order = 5;
            qf1: Multipole, knl = {0.0, knl_f};
            d12: Drift, length = cell_l;
            qd2: Multipole, knl = {0.0, knl_d};
        endbeamline;
    """

    multiline = xt.Multiline.from_string(sequence, _context=xo.context_default)

    temp_file = tmp_path / 'test_multiline.xld'

    multiline.to_file(temp_file)

    with temp_file.open('r') as f:
        generated_sequence = f.read()

    generated_lines = _normalize_code(generated_sequence, remove_empty_lines=False)
    original_lines = _normalize_code(sequence, remove_empty_lines=False)

    cmp_args = lambda lines: set(text[0:-1] for text in lines)

    assert cmp_args(generated_lines[0:7]) == cmp_args(original_lines[0:7])
    assert generated_lines[7:11] == original_lines[7:11]
    assert cmp_args(generated_lines[11:19]) == cmp_args(original_lines[11:19])
    assert generated_lines[19:26] == original_lines[19:26]

    assert generated_lines[26:28] == original_lines[26:28]
    assert cmp_args(generated_lines[28:31]) == cmp_args(original_lines[28:31])
    assert generated_lines[31:33] == original_lines[31:33]
    assert cmp_args(generated_lines[33:40]) == cmp_args(original_lines[33:40])
    assert generated_lines[40:45] == original_lines[40:45]


def test_name_shadowing_error():
    sequence = """\
    Marker: Marker;
    
    line: beamline;
        Bend: Bend, length = 1;
    endbeamline;
    """

    parser = Parser(_context=xo.context_default)

    with pytest.raises(ParseError) as e:
        parser.parse_string(sequence)

    assert 'name `Marker` shadows a built-in type.' in str(e.value)
    assert 'name `Bend` shadows a built-in type.' in str(e.value)


def test_slice_elements():
    sequence = """\
    template: Drift, length = 1.9;
    
    line: beamline;
        # Test slices that refer to "global" elements
        elm_a: DriftSlice, parent_name = "template", weight = 0.5;
        elm_a: DriftSlice, parent_name = "template", weight = 0.5;
        # This should produce a replica of the template
        elm_c: template;
    endbeamline;
    
    line2: beamline;
        elm_a: DriftSlice, parent_name = "template", weight = 1.0;
    endbeamline;
    """

    multiline = xt.Multiline.from_string(sequence)
    multiline.build_trackers()

    tab1 = multiline.line.get_table()
    tab2 = multiline.line2.get_table()

    xo.assert_allclose(tab1['s'], [0, 0.95, 1.9, 3.8], atol=1e-16)
    xo.assert_allclose(tab2['s'], [0, 1.9], atol=1e-16)


@pytest.mark.parametrize('line_name', ['same_name', 'different_name'])
def test_parsed_line_to_collider(tmp_path, line_name):
    sequence = f"""\
    dr_len = 5;

    {line_name}: beamline;
        test_element: Drift, length = dr_len;
    endbeamline;
    """

    line = xt.Line.from_string(sequence, _context=xo.context_default)

    multiline = xt.Multiline(lines = {'same_name': line})

    outfile = tmp_path / 'test_parsed_line_to_collider.xld'
    multiline.to_file(outfile)
    with outfile.open('r') as f:
        generated_sequence = f.read()

    expected_lines = _normalize_code(sequence.replace('different_name', 'same_name'))
    generated_lines = _normalize_code(generated_sequence)

    assert expected_lines == generated_lines

    multiline.same_name.vars['dr_len'] = 10
    assert multiline.same_name['test_element'].length == 10


def test_parsed_line_copy():
    sequence = """\
    dr_len = 5;
    k = 3;

    line: beamline;
        dr: Drift, length = dr_len;
        mb: Bend, length = 2, knl = {0, 0, 1}, k0 = k;
    endbeamline;
    """

    line = xt.Line.from_string(sequence, _context=xo.context_default)
    copied_line = line.copy()

    line_dict = line.to_dict(include_var_management=True)
    copied_dict = copied_line.to_dict(include_var_management=True)

    _dict_compare(line_dict, copied_dict)

    line.vars['k'] = 4
    assert line['mb'].k0 == 4

    assert copied_line['mb'].k0 == 3

    copied_line.vars['k'] = 5

    assert line['mb'].k0 == 4
    assert copied_line['mb'].k0 == 5


def test_modify_element_refs_arrow_syntax():
    sequence = """\
    dr_len = 5;
    k = 3;

    line: beamline;
        dr: Drift, length = dr_len;
        mb: Bend, length = 2, knl = {0, 0, 1}, k0 = k;
    endbeamline;

    line->dr->length = dr_len + 1;
    line->mb->k0 = 4;
    """

    line = xt.Line.from_string(sequence, _context=xo.context_default)

    assert line['dr'].length == 6
    assert line['mb'].k0 == 4

def test_commands_on_line():
    sequence = """\
    line: beamline;
        particle_ref, x = 1;
        twiss_default, reverse, method = "4d";
        config, MY_CUSTOM_C_FLAG = "whatever";
        attr, update = "_extra_config", json = "{\\"skip_end_turn_actions\\": true}";
        dr: Drift, length = 2;
    endbeamline;
    """

    line = xt.Line.from_string(sequence, _context=xo.context_default)

    assert line.particle_ref.x == 1
    assert line.twiss_default['reverse']
    assert line.twiss_default['method'] == '4d'
    assert line.config.MY_CUSTOM_C_FLAG == 'whatever'
    assert line._extra_config['skip_end_turn_actions'] == True


def test_arrow_syntax_errors():
    sequence = """\
    line: beamline;
        dr: Drift, length = 2;
    endbeamline;

    line2->mb->k0 = 8;
    line->not_there->h = 3;
    """

    with pytest.raises(ParseError) as e:
        _ = xt.Line.from_string(sequence, _context=xo.context_default)

    assert 'on line 5 column 5: no such line' in str(e.value)
    assert 'cannot access `h`' in str(e.value)


def test_add_expressions_to_preexisting_line():
    line = xt.Line(
        elements={
            'dr': xt.Drift(length=2),
            'mb': xt.Bend(length=2, knl=[0, 0, 1], k0=3),
        },
        element_names=['dr', 'mb'],
    )
    line._init_var_management()

    line.eval('half = 3; dr->length = 2 * half;')