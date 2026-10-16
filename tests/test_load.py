# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #
import pytest
import xtrack as xt
import json
import textwrap
import gzip
import numpy as np


@pytest.mark.parametrize('container', ['line', 'environment'])
@pytest.mark.parametrize('monitor_name,kwargs', [
    ('ParticlesMonitor', dict(start_at_turn=2, stop_at_turn=5, num_particles=2)),
    ('LastTurnsMonitor', dict(n_last_turns=3, particle_id_range=(1, 3), every_n_turns=2)),
    ('BeamPositionMonitor', dict(start_at_turn=2, stop_at_turn=5,
                                 frev=1., sampling_frequency=2.)),
    ('BeamSizeMonitor', dict(start_at_turn=2, stop_at_turn=5,
                             frev=1., sampling_frequency=2.)),
    ('BeamProfileMonitor', dict(start_at_turn=2, stop_at_turn=5,
                                nx=7, ny=5, x_range=(0.003, 0.013),
                                y_range=(-0.007, 0.02))),
    ('BeamStatsMonitor', dict(start_at_turn=2, stop_at_turn=5,
                              stats=['num_particles', 'mean_x'],
                              particle_id_range=(1, 3))),
])
def test_load_lattice_with_monitor(tmp_path, container, monitor_name, kwargs):
    monitor = getattr(xt, monitor_name)(**kwargs)
    if monitor_name == 'LastTurnsMonitor':
        monitor.data.x.to_nparray()[:] = np.arange(6) + 0.5
        monitor.data.lost_at_offset.to_nparray()[:] = [1, 2]
    elif monitor_name == 'BeamProfileMonitor':
        monitor.data.counts_x.to_nparray()[:] = np.arange(21)
        monitor.data.counts_y.to_nparray()[:] = np.arange(15)[::-1]
    env = xt.Environment()
    env.elements['monitor'] = monitor
    env.new('drift', xt.Drift, length=1)
    line = env.new_line(name='seq', components=['monitor', 'drift'])
    lattice = line if container == 'line' else env
    path = tmp_path / 'lattice.json'
    lattice.to_json(path)

    loaded = xt.load(path)
    loaded_line = loaded if container == 'line' else loaded.seq
    assert loaded_line.element_names == line.element_names
    restored = loaded_line.get('monitor')
    assert type(restored) is type(monitor)
    np.testing.assert_equal(restored.to_dict(), monitor.to_dict())


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

@pytest.mark.parametrize(
    'input_fixture,format,suffix', [
        ('json_line', 'json', 'json'),
        ('json_environment', 'json', 'json'),
    ]
)
@pytest.mark.parametrize('with_format', [True, False])
def test_load_http_gz(input_fixture, format, suffix, with_format, tmpdir, request, requests_mock):
    input_data = request.getfixturevalue(input_fixture)
    compressed_data=gzip.compress(input_data.encode("utf-8"))

    url = f'http://example.com/test_input.{suffix}.gz'
    requests_mock.get(url, content=compressed_data)

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
