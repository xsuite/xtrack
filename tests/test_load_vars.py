# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #
import pytest
import xtrack as xt
import json
import textwrap


@pytest.mark.parametrize('expression', [
    '-a^2', '-a^2 + 1', 'b + (-a^2)', '-2^2', '1 - a^2', '-(a^2)',
    '(-a)^2', '-a^3', '-a*b', 'a*(-b^2)', 'a/(-b^2)', '-sin(a)^2',
    'a^(-2)', 'a^(-b^2)', 'a^b^2', '-a^2^2', '+a^2', '1e-2^2',
])
def test_madx_power_precedence(expression):
    from cpymad.madx import Madx

    source = f'a = 0.3; b = 0.7; immediate = {expression}; deferred := {expression};'
    env = xt.Environment()
    env.vars.load(string=source, format='madx')
    with Madx(stdout=False) as madx:
        madx.input(source)
        for a in (0.3, 0.4):
            env['a'] = a
            madx.globals['a'] = a
            for name in ('immediate', 'deferred'):
                assert env[name] == pytest.approx(madx.globals[name])


@pytest.mark.parametrize('power', ['^', '**'])
def test_expression_string_power_precedence(power):
    env = xt.Environment()
    env['a'] = 0.3
    env['negative_square'] = f'-a {power} 2'
    env['squared_negative'] = f'(-a) {power} 2'
    env['negative_exponent'] = f'a {power} -2'
    for a in (0.3, -0.4):
        env['a'] = a
        assert env['negative_square'] == pytest.approx(-a**2)
        assert env['squared_negative'] == pytest.approx(a**2)
        assert env['negative_exponent'] == pytest.approx(a**-2)


def test_madx_signed_exponents():
    env = xt.Environment()
    env.vars.load(string='a = 0.3; x := -a^-2; y := a^+2;', format='madx')
    for a in (0.3, -0.4):
        env['a'] = a
        assert env['x'] == pytest.approx(-(a**-2))
        assert env['y'] == pytest.approx(a**2)


@pytest.fixture
def madx_vars():
    return """
        b := 3 * a;
        a = 5;
        c := 4 * z;
    """


@pytest.fixture
def json_vars():
    dct = {
        'b': '3 * a',
        'a': 5,
        'c': '4 * z'
    }
    return json.dumps(dct)


@pytest.fixture
def python_file():
    source = """
        import xtrack as xt
        env = xt.get_environment()
        env.vars.default_to_zero = True
        env['b'] = '3 * a'
        env['a'] = 5
        env['c'] = '4 * z'
    """
    return textwrap.dedent(source)


@pytest.mark.parametrize(
    'input_fixture,format', [
        ('madx_vars', 'madx'),
        ('json_vars', 'json'),
    ]
)
def test_load_vars_string(input_fixture, format, request):
    input_data = request.getfixturevalue(input_fixture)
    env = xt.Environment()
    env.vars.load(string=input_data, format=format)

    assert str(env.get_expr('b')) == "(3.0 * vars['a'])"
    assert env['b'] == 15
    assert env.get_expr('a') is None
    assert env['a'] == 5
    assert str(env.get_expr('c')) == "(4.0 * vars['z'])"
    assert env['c'] == 0
    assert env.get_expr('z') is None
    assert env['z'] == 0

@pytest.mark.parametrize(
    'input_fixture,format,suffix', [
        ('madx_vars', 'madx', 'madx'),
        ('json_vars', 'json', 'json'),
        ('python_file', 'python', 'py'),
    ]
)
@pytest.mark.parametrize('with_format', [True, False])
def test_load_file(input_fixture, format, suffix, with_format, tmpdir, request):
    input_data = request.getfixturevalue(input_fixture)

    temp_file = tmpdir / f'test_input.{suffix}'
    with open(temp_file, 'w') as f:
        f.write(input_data)

    kwargs = {'file': str(temp_file)}
    if with_format:
        kwargs['format'] = format
    env = xt.Environment()
    env.vars.load(**kwargs)

    assert str(env.get_expr('b')) == "(3.0 * vars['a'])"
    assert env['b'] == 15
    assert env.get_expr('a') is None
    assert env['a'] == 5
    assert str(env.get_expr('c')) == "(4.0 * vars['z'])"
    assert env['c'] == 0
    assert env.get_expr('z') is None
    assert env['z'] == 0


@pytest.mark.parametrize(
    'input_fixture,format,suffix', [
        ('madx_vars', 'madx', 'madx'),
        ('json_vars', 'json', 'json'),
    ]
)
@pytest.mark.parametrize('with_format', [True, False])
def test_load_http(input_fixture, format, suffix, with_format, tmpdir, request, requests_mock):
    input_data = request.getfixturevalue(input_fixture)

    url = f'http://example.com/test_input.{suffix}'
    requests_mock.get(url, text=input_data)

    kwargs = {'file': url}
    if with_format:
        kwargs['format'] = format
    env = xt.Environment()
    env.vars.load(**kwargs)

    assert str(env.get_expr('b')) == "(3.0 * vars['a'])"
    assert env['b'] == 15
    assert env.get_expr('a') is None
    assert env['a'] == 5
    assert str(env.get_expr('c')) == "(4.0 * vars['z'])"
    assert env['c'] == 0
    assert env.get_expr('z') is None
    assert env['z'] == 0
