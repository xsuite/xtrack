# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #
import pytest
import xtrack as xt
import json
import textwrap


@pytest.fixture
def madx_sequence():
    return """
        seq: sequence, l=1;
            quad: quadrupole, l=1, k1=0.5, at=0.5;
        endsequence;
    """


@pytest.fixture
def json_line():
    dct = {
        'elements': {
            'quad': {
                '__class__': 'Quadrupole',
                'length': 1.0,
                'k1': 0.5
            }
        },
        'element_names': ['quad'],
    }
    return json.dumps(dct)


@pytest.fixture
def json_environment():
    dct = {
        'elements': {
            'quad': {
                '__class__': 'Quadrupole',
                'length': 1.0,
                'k1': 0.5,
            }
        },
        'xsuite_data_type': 'Environment',
        'lines': {
            'seq': {
                'element_names': ['quad'],
            }
        }
    }
    return json.dumps(dct)


@pytest.fixture
def python_file():
    source = """
        import xtrack as xt
        env = xt.get_environment()
        quad = env.new('quad', 'Quadrupole', length=1.0, k1=0.5)
        env.new_line(name='seq', components=[quad])
    """
    return textwrap.dedent(source)


@pytest.mark.parametrize(
    'input_fixture,format', [
        ('madx_sequence', 'madx'),
        ('json_line', 'json'),
        ('json_environment', 'json'),
    ]
)
def test_load_string(input_fixture, format, request):
    input_data = request.getfixturevalue(input_fixture)
    loaded_entity = xt.load(string=input_data, format=format)  # noqa

    quad = loaded_entity['quad']
    assert quad.length == 1.0 and quad.k1 == 0.5


@pytest.mark.parametrize(
    'input_fixture,format,suffix', [
        ('madx_sequence', 'madx', 'madx'),
        ('json_line', 'json', 'json'),
        ('json_environment', 'json', 'json'),
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
    loaded_entity = xt.load(**kwargs)

    quad = loaded_entity['quad']
    assert quad.length == 1.0 and quad.k1 == 0.5


@pytest.mark.parametrize(
    'input_fixture,format,suffix', [
        ('madx_sequence', 'madx', 'madx'),
        ('json_line', 'json', 'json'),
        ('json_environment', 'json', 'json'),
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
    loaded_entity = xt.load(**kwargs)

    quad = loaded_entity['quad']
    assert quad.length == 1.0 and quad.k1 == 0.5


def test_load_single_element():
    string = json.dumps({
        '__class__': 'Quadrupole',
        'length': 1.0,
        'k1': 0.5
    })
    quad = xt.load(string=string, format='json')
    assert quad.length == 1.0 and quad.k1 == 0.5


def test_load_invalid_input():
    with pytest.raises(ValueError, match='either file or string'):
        xt.load()

    with pytest.raises(ValueError, match='either file or string'):
        xt.load(file='test.madx', string='quad: quadrupole, l=1, k1=0.5;')

    with pytest.raises(ValueError, match='Format must be specified'):
        xt.load(string='{}')

    with pytest.raises(ValueError, match='one of'):
        xt.load(string='{}', format='invalid_format')  # noqa

    with pytest.raises(ValueError, match='Cannot determine class from json data'):
        xt.load(string='{"length": 1.0, "k1": 0.5}', format='json')
