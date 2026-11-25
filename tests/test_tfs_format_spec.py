import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest

import xtrack as xt
from xtrack.survey import SurveyTable


def sample_table(cls=xt.Table):
    return cls({
        'name': np.array(['ip1', 'qf']),
        's': np.array([0., 1.25]),
        'x': np.array([1.25, -0.]),
        'unused': np.array([100., 200.]),
        'qx': 0.31, 'qy': 0.32, 'title': 'Example lattice',
    }, col_names=['name', 's', 'x', 'unused'])


def sample_spec():
    return {
        'version': '1',
        'numeric_column_width': 0,
        'float_format': '.3f',
        'columns': [
            {'name': 'NAME', 'source_name': 'name', 'type': 'string', 'width': 8},
            {'name': 'X', 'source_name': 'x', 'type': 'float', 'format': '.2f', 'width': 6},
            {'name': 's', 'type': 'float', 'format': '.1e'},
        ],
        'scalars': [
            {'name': 'Q2', 'source_name': 'qy', 'type': 'float'},
            {'name': 'title', 'type': 'string'},
        ],
    }


def render(table, spec, **kwargs):
    target = io.StringIO()
    table.to_tfs(target, format_spec=spec, **kwargs)
    assert not target.closed
    return target.getvalue()


@pytest.mark.parametrize('cls', [xt.Table, xt.TwissTable, SurveyTable])
def test_exact_output_and_independence_from_table_storage(cls, tmp_path):
    table, spec = sample_table(cls), sample_spec()
    original = copy.deepcopy(spec)
    expected = (
        '@ Q2    %le 0.320\n'
        '@ title %s  "Example lattice"\n'
        '* NAME          X       s\n'
        '$ %s          %le     %le\n'
        ' "ip1"      1.25 0.0e+00\n'
        ' "qf"      -0.00 1.2e+00\n'
    )
    assert render(table, spec) == expected
    assert spec == original
    assert table.name.tolist() == ['ip1', 'qf']

    # Fields added by a future Xtrack release, and storage order, are irrelevant.
    table._data['future_metadata'] = {'arbitrary': 'data'}
    table._data['future_column'] = np.array([1, 2])
    table._col_names.append('future_column')
    table._col_names.reverse()
    table._data = dict(reversed(list(table._data.items())))
    assert render(table, spec) == expected
    path = tmp_path / 'output.tfs'
    table.to_tfs(path, format_spec=spec)
    assert path.read_bytes() == expected.encode('utf-8')


def test_json_toml_and_mapping_equivalence(tmp_path):
    table, spec = sample_table(), sample_spec()
    json_file = tmp_path / 'format.json'
    json_file.write_text(json.dumps(spec), encoding='utf-8')
    toml_file = tmp_path / 'format.toml'
    toml_file.write_text('''
version = "1"
numeric_column_width = 0
float_format = ".3f"
columns = [
    { name = "NAME", source_name = "name", type = "string", width = 8 },
    { name = "X", source_name = "x", type = "float", format = ".2f", width = 6 },
    { name = "s", type = "float", format = ".1e" },
]
scalars = [
    { name = "Q2", source_name = "qy", type = "float" },
    { name = "title", type = "string" },
]
''', encoding='utf-8')
    expected = render(table, spec)
    for source in (json_file, str(json_file), toml_file, str(toml_file)):
        assert render(table, source) == expected
        assert xt.format_spec.load(source) == xt.format_spec.load(spec)


@pytest.mark.parametrize('case', ['upper', 'lower', 'preserve'])
def test_case_affects_output_only(case):
    table, spec = sample_table(), sample_spec()
    spec['case'] = case
    output = render(table, spec)
    if case == 'upper':
        assert '@ TITLE %s  "EXAMPLE LATTICE"' in output
        assert '"IP1"' in output
    elif case == 'lower':
        assert '@ q2' in output
        assert '"example lattice"' in output
    else:
        assert '"Example lattice"' in output
    assert '1.2e+00' in output  # case does not override numeric notation
    assert table.name[0] == 'ip1'
    assert table.title == 'Example lattice'
    assert spec['columns'][0]['source_name'] == 'name'


@pytest.mark.parametrize('updates, error, match', [
    ({'version': 1}, TypeError, 'must be a string'),
    ({'version': '2'}, ValueError, 'Unsupported'),
    ({'version': '1.0'}, ValueError, 'Unsupported'),
    ({'version': 'unknown_v1'}, ValueError, 'Unsupported'),
    ({'typo': 1}, ValueError, 'Unknown key'),
    ({'case': 'title'}, ValueError, 'case'),
    ({'float_format': '.3q'}, ValueError, 'float_format'),
    ({'float_format': 'n'}, ValueError, 'float_format'),
    ({'float_format': ',.2f'}, ValueError, 'float_format'),
    ({'float_format': '%12.4f'}, ValueError, 'float_format'),
    ({'float_format': '.\u0663f'}, ValueError, 'float_format'),
    ({'numeric_column_width': -1}, ValueError, 'width'),
    ({'default_column_width': True}, TypeError, 'width'),
    ({'columns': {'x': {}}}, TypeError, 'ordered list'),
    ({'columns': ['x']}, TypeError, 'mapping'),
    ({'columns': [{'source_name': 'x'}]}, ValueError, 'Missing key'),
    ({'columns': [{'name': 'x', 'source': 'x'}]}, ValueError, 'Unknown key'),
    ({'columns': [{'name': 'x', 'type': 'double'}]}, ValueError, 'type'),
    ({'columns': [{'name': 'x', 'type': 'int'}]}, TypeError, 'Expected int'),
    ({'columns': [{'name': 'x', 'format': '.2q'}]}, ValueError, 'format'),
    ({'columns': [{'name': 'x', 'format': 'd'}]}, ValueError, 'incompatible'),
    ({'columns': [{'name': 'x', 'width': 1.5}]}, TypeError, 'width'),
    ({'columns': [{'name': ''}]}, ValueError, 'non-empty'),
    ({'columns': [{'name': 'bad name', 'source_name': 'x'}]}, ValueError, 'whitespace'),
    ({'columns': [{'name': 'X'}]}, KeyError, 'Missing'),
    ({'columns': [{'name': 'qx'}]}, ValueError, 'not a column'),
    ({'scalars': [{'name': 'x'}]}, ValueError, 'not a scalar'),
    ({'scalars': [{'name': 'qx', 'width': 10}]}, ValueError, 'Unknown key'),
    ({'scalars': [{'name': 'qx', 'value': 1, 'source_name': 'qx'}]}, ValueError, 'both'),
    ({'scalars': [{'name': 'qx', 'value': []}]}, TypeError, 'Unsupported'),
    ({'scalars': [{'name': 'label', 'value': 'text', 'format': '.2f'}]}, ValueError, 'incompatible'),
    ({'columns': [{'name': 'x', 'value': 1}]}, ValueError, 'Unknown key'),
    ({'columns': [{'name': 'x'}, {'name': 'x'}]}, ValueError, 'Duplicate'),
    ({'case': 'upper', 'columns': [{'name': 'x'}, {'name': 'X', 'source_name': 's'}]},
     ValueError, 'Duplicate'),
    ({'case': 'upper', 'scalars': [{'name': 'qx'}, {'name': 'QX', 'source_name': 'qy'}]},
     ValueError, 'Duplicate'),
])
def test_invalid_spec_does_not_touch_destination(updates, error, match, tmp_path):
    spec = sample_spec()
    spec.update(updates)
    path = tmp_path / 'existing.tfs'
    path.write_bytes(b'keep me\n')
    with pytest.raises(error, match=match):
        sample_table().to_tfs(path, format_spec=spec)
    assert path.read_bytes() == b'keep me\n'
    stream = io.StringIO('keep me\n')
    with pytest.raises(error, match=match):
        sample_table().to_tfs(stream, format_spec=spec)
    assert stream.getvalue() == 'keep me\n'


@pytest.mark.parametrize('key', ['version', 'columns', 'scalars'])
def test_required_spec_keys(key):
    spec = sample_spec()
    del spec[key]
    with pytest.raises(ValueError, match='Missing key'):
        render(sample_table(), spec)


def test_file_validation(tmp_path):
    path = tmp_path / 'invalid.json'
    path.write_text('{"version": "1", "version": "2", "columns": [], "scalars": []}')
    with pytest.raises(ValueError, match='Duplicate'):
        xt.format_spec.load(path)
    with pytest.raises(ValueError, match='.json or .toml'):
        xt.format_spec.load(tmp_path / 'unsupported.yaml')


def test_constant_headers(tmp_path):
    spec = {'version': '1', 'case': 'upper', 'columns': [], 'scalars': [
        {'name': 'type', 'value': 'twiss'},
        {'name': 'number', 'value': 2.5, 'format': '.2f'},
        {'name': 'count', 'value': 12, 'format': '04d'},
        {'name': 'flag', 'value': True},
    ]}
    expected = ('@ TYPE   %s  "TWISS"\n@ NUMBER %le 2.50\n'
                '@ COUNT  %d  0012\n@ FLAG   %b  1\n')
    assert render(sample_table(), spec) == expected
    path = tmp_path / 'constants.json'
    path.write_text(json.dumps(spec))
    assert render(sample_table(), path) == expected
    path = tmp_path / 'constants.toml'
    path.write_text('''version = "1"
case = "upper"
columns = []
scalars = [
    {name = "type", value = "twiss"},
    {name = "number", value = 2.5, format = ".2f"},
    {name = "count", value = 12, format = "04d"},
    {name = "flag", value = true},
]
''')
    assert render(sample_table(), path) == expected


@pytest.mark.parametrize('cls', [xt.Table, xt.TwissTable, SurveyTable])
@pytest.mark.parametrize('kwargs', [
    {'include': None}, {'exclude': None}, {'missing': 'error'},
    {'include_meta': True}, {'default_column_width': None},
    {'float_precision': 8}, {'numeric_column_width': 16},
    {'column_formats': None}, {'column_widths': None},
])
def test_reject_legacy_options_even_at_their_default(cls, kwargs):
    with pytest.raises(TypeError, match='cannot be combined with'):
        render(sample_table(cls), sample_spec(), **kwargs)


def test_type_tokens_numeric_formats_and_external_reader(tmp_path):
    import tfs

    data = {
        'name': np.array(['a b', 'é']),
        'i': np.array([2**53 + 1, -2], dtype=np.int64),
        'b': np.array([True, False]),
        'f': np.array([1.23456789, -0.]),
        'z': np.array([1+2j, complex(0, -0.)]),
        'q': 0.123456,
    }
    table = xt.Table(data, col_names=['name', 'i', 'b', 'f', 'z'])
    spec = {
        'version': '1', 'float_format': '.3E',
        'columns': [{'name': name} for name in table._col_names],
        'scalars': [{'name': 'q', 'format': '.4f'}],
    }
    path = tmp_path / 'types.tfs'
    table.to_tfs(path, format_spec=spec)
    text = path.read_text()
    assert '%s' in text and '%d' in text and '%b' in text and '%lz' in text
    assert '1.235E+00' in text
    assert '-0.000E+00' in text
    result = tfs.read(path)
    assert result.headers == {'q': 0.1235}
    assert result['name'].tolist() == ['a b', 'é']
    np.testing.assert_array_equal(result['i'], data['i'])
    np.testing.assert_array_equal(result['b'], data['b'])
    np.testing.assert_allclose(result['z'], data['z'])
    assert np.signbit(result['f'][1])


def test_special_values_and_default_float_format():
    table = xt.Table({'name': np.array(['a', 'b', 'c', 'd']),
                      'x': np.array([np.nan, np.inf, -np.inf, -0.]),
                      'q': 1.25}, col_names=['name', 'x'])
    spec = {'version': '1', 'case': 'upper',
            'columns': [{'name': 'x'}], 'scalars': [{'name': 'q'}]}
    text = render(table, spec)
    assert text.splitlines()[0] == '@ Q %le 1.2500000000000000e+00'
    assert [line.strip() for line in text.splitlines()[3:]] == [
        'nan', 'inf', '-inf', '-0.0000000000000000e+00']


@pytest.mark.parametrize('value', [None, {'a': 1}, np.array([1., 2.]),
                                  'quote"', 'back\\slash', 'new\nline'])
def test_unsupported_scalar_values(value):
    table = sample_table()
    table._data['bad'] = value
    with pytest.raises((ValueError, TypeError)):
        render(table, {'version': '1', 'columns': [], 'scalars': [{'name': 'bad'}]})


def test_empty_columns_and_scalar_only_output():
    table = xt.Table({'name': np.array([], dtype=str),
                      'x': np.array([], dtype=float), 'q': 0.5},
                     col_names=['name', 'x'])
    spec = {'version': '1', 'columns': [{'name': 'name', 'type': 'string'},
                                      {'name': 'x'}], 'scalars': []}
    assert len(render(table, spec).splitlines()) == 2
    spec['columns'] = []
    spec['scalars'] = [{'name': 'q', 'format': '.1f'}]
    assert render(table, spec) == '@ q %le 0.5\n'
    spec['scalars'] = []
    assert render(table, spec) == ''


def test_nested_columns_and_mismatched_lengths_fail(tmp_path):
    table, spec = sample_table(), sample_spec()
    table._data['x'] = np.ones((2, 2))
    path = tmp_path / 'not-created.tfs'
    with pytest.raises(ValueError, match='one-dimensional'):
        table.to_tfs(path, format_spec=spec)
    assert not path.exists()
    table._data['x'] = np.ones(1)
    with pytest.raises(ValueError, match='same length'):
        table.to_tfs(path, format_spec=spec)
    assert not path.exists()


def test_presets_are_immutable_and_can_be_copied():
    preset = xt.format_spec.optics_orbit_v1
    with pytest.raises(TypeError):
        preset['case'] = 'upper'
    with pytest.raises(TypeError):
        preset['columns'][0]['name'] = 'other'
    spec = xt.format_spec.load(preset)
    spec['columns'][0]['name'] = 'other'
    assert preset['columns'][0]['name'] == 'NAME'
    assert json.loads(json.dumps(spec)) == spec


@pytest.mark.parametrize('preset_name', ['optics_orbit_v1', 'optics_orbit_strengths_v1'])
def test_preset_output_snapshot(preset_name):
    # These checked-in files pin both the preset contents and the renderer.
    preset = getattr(xt.format_spec, preset_name)
    data = {}
    for field in preset['columns']:
        source = field['source_name']
        if field['type'] == 'string':
            data[source] = np.array(['ip1', 'qf'] if source == 'name'
                                     else ['Marker', 'Quadrupole'])
        elif field['type'] == 'bool':
            data[source] = np.array([False, True])
        else:
            data[source] = np.array([0.125, -2.5])
    columns = list(data)
    data.update(line_length=123.5, qx=0.31, qy=0.32)
    table = xt.Table(data, col_names=columns)
    reference = Path(__file__).parent / 'data' / 'tfs_format_spec' / (preset_name + '.tfs')
    assert render(table, preset).encode('utf-8') == reference.read_bytes()


def test_presets_on_computed_twiss(tmp_path):
    import tfs

    line = xt.Line(elements=[
        xt.Drift(length=1), xt.Multipole(knl=[0, 0.1]),
        xt.Drift(length=1), xt.Multipole(knl=[0, -0.1]),
    ] * 4)
    line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.PROTON_MASS_EV)
    tw = line.twiss4d(strengths=True)
    for name in ('optics_orbit_v1', 'optics_orbit_strengths_v1', 'madx_twiss_v1'):
        preset = getattr(xt.format_spec, name)
        assert preset['version'] == name
        spec_copy = xt.format_spec.load(preset)
        assert xt.format_spec.load(json.loads(json.dumps(spec_copy)))['version'] == name
        path = tmp_path / (name + '.tfs')
        tw.to_tfs(path, format_spec=preset)
        result = tfs.read(path)
        assert list(result.columns) == [field['name'].upper() for field in preset['columns']]
        if name == 'madx_twiss_v1':
            assert list(result.headers) == ['TYPE', 'LENGTH', 'ALFA', 'Q1', 'Q2', 'DQ1', 'DQ2']
            assert result['NAME'].tolist() == [nn.upper() for nn in tw.name]
        else:
            assert list(result.headers) == ['LINE_LENGTH', 'QX', 'QY']
            assert result['NAME'].tolist() == tw.name.tolist()
        np.testing.assert_allclose(result['BETX'], tw.betx, rtol=1e-15)
        if name == 'optics_orbit_strengths_v1':
            np.testing.assert_allclose(result['K1L'], tw.k1l, rtol=1e-15)
            np.testing.assert_array_equal(result['ISTHICK'], tw.isthick)


def test_madx_preset_against_reference_excerpt(tmp_path):
    import tfs

    folder = Path(__file__).parent / 'data' / 'tfs_format_spec'
    reference = tfs.read(folder / 'madx_hllhc14_excerpt.tfs')
    preset = xt.format_spec.madx_twiss_v1
    data = {field['source_name']: reference[field['name'].upper()].to_numpy()
            for field in preset['columns']}
    columns = list(data)
    data.update({field['source_name']: reference.headers[field['name'].upper()]
                 for field in preset['scalars'] if 'source_name' in field})
    output = tmp_path / 'madx.tfs'
    xt.Table(data, col_names=columns).to_tfs(output, format_spec=preset)
    result = tfs.read(output, validate='madx')
    # The subset preserves the reference's column order and TFS data types.
    assert list(result.columns) == [name for name in reference.columns if name in result.columns]
    assert list(result.headers) == ['TYPE', 'LENGTH', 'ALFA', 'Q1', 'Q2', 'DQ1', 'DQ2']
    for name in result.columns:
        if name == 'NAME':
            assert result[name].tolist() == reference[name].tolist()
        else:
            np.testing.assert_allclose(result[name], reference[name], rtol=1e-15, atol=0)


def test_madx_example_optics_conventions(tmp_path):
    import runpy
    from cpymad.madx import Madx
    import tfs

    example = Path(__file__).parents[1] / 'examples/twiss_table_file/005_madx_tfs_format.py'
    prepare = runpy.run_path(str(example))['prepare_madx_conventions']
    mad = Madx(stdout=False)
    mad.input('''
        beam, particle=proton, pc=1;
        d: drift, l=1;
        b: sbend, l=1, angle=0.05;
        qf: quadrupole, l=0.2, k1=0.3;
        qd: quadrupole, l=0.2, k1=-0.3;
        ring: line=(b,qf,d,qd,b,qf,d,qd,b,qf,d,qd,b,qf,d,qd);
        use, period=ring;
        twiss, chrom;
    ''')
    line = xt.Line.from_madx_sequence(mad.sequence.ring)
    # Match MAD-X's nonlinear edge focusing for off-momentum chromaticities.
    line.configure_bend_model(edge='full')
    line.particle_ref = xt.Particles(
        mass0=mad.sequence.ring.beam.mass * 1e9,
        gamma0=mad.sequence.ring.beam.gamma)
    tw = line.twiss4d(strengths=True)
    original_dx = tw.dx.copy()
    prepared = prepare(tw)
    path = tmp_path / 'madx-conventions.tfs'
    prepared.to_tfs(path, format_spec=xt.format_spec.madx_twiss_v1)
    result = tfs.read(path, validate='madx')
    for name in ('s', 'x', 'y', 'px', 'py', 'betx', 'bety', 'alfx', 'alfy',
                 'mux', 'muy', 'dx', 'dy', 'dpx', 'dpy', 'k1l', 'angle'):
        np.testing.assert_allclose(result[name.upper()], mad.table.twiss[name],
                                   rtol=1e-7, atol=2e-9)
    for name in ('q1', 'q2', 'dq1', 'dq2', 'alfa', 'length'):
        np.testing.assert_allclose(result.headers[name.upper()], mad.table.summ[name][0],
                                   rtol=1e-5, atol=2e-7)
    np.testing.assert_array_equal(tw.dx, original_dx)
    assert tw.name[-1] == '_end_point'
    assert len(result) == len(tw) - 1
    mad.quit()


@pytest.mark.parametrize('cls', [xt.Table, xt.TwissTable, SurveyTable])
def test_legacy_defaults_and_options(cls):
    table = sample_table(cls)
    if cls is xt.TwissTable:
        table._data['completed_init'] = 'not exported'
    default = io.StringIO()
    explicit = io.StringIO()
    table.to_tfs(default)
    table.to_tfs(explicit, include=None, exclude=None, missing='error',
                 include_meta=True, default_column_width=None, float_precision=8,
                 numeric_column_width=16, column_formats=None, column_widths=None)
    assert default.getvalue() == explicit.getvalue()
    assert '@ __CLASS__' in default.getvalue()
    exclusions = ['unused']
    table.to_tfs(io.StringIO(), exclude=exclusions)
    assert exclusions == ['unused']
