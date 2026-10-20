import io
import json

import numpy as np
import pytest
import tfs

import xtrack as xt
from xtrack.survey import SurveyTable


@pytest.mark.parametrize('exclude', [None, 'line', 'length', ['length'], ('length',)])
def test_survey_to_tfs(tmp_path, exclude):
    line = xt.Line(
        elements=[xt.Drift(length=1), xt.Bend(length=2, angle=0.1)],
        element_names=['d', 'b'],
    )
    survey = line.survey()
    path = tmp_path / 'survey.tfs'
    survey.to_tfs(path, exclude=exclude, float_precision=16)
    loaded = tfs.read(path)

    assert 'LINE' not in loaded.headers
    assert loaded.headers['__CLASS__'] == 'SurveyTable'
    assert survey.line is line
    assert loaded.headers['ELEMENT0'] == survey.element0
    excluded = [] if exclude is None else (
        [exclude] if isinstance(exclude, str) else exclude)
    assert list(loaded.columns) == [
        name.upper() for name in survey._col_names if name not in excluded]
    for name in survey._col_names:
        if name in excluded:
            continue
        # Compare stored columns, including legacy aliases, without invoking
        # the deprecated public accessors for p0 and W.
        expected = survey._data[name]
        values = loaded[name.upper()].to_numpy()
        if expected.ndim > 1:
            values = np.array([json.loads(value) for value in values])
        if expected.dtype.kind in 'biufc':
            np.testing.assert_allclose(values, expected, atol=1e-15)
        elif name in ('name', 'element_type'):
            np.testing.assert_array_equal(values, expected)
    if isinstance(exclude, list):
        assert exclude == ['length']

    # Loading and exporting again also works without an attached line.
    SurveyTable.from_tfs(path).to_tfs(io.StringIO())


def test_survey_to_tfs_default_and_format_spec():
    line = xt.Line(elements=[xt.Drift(length=1)], element_names=['d'])
    survey = line.survey()
    default = io.StringIO()
    workaround = io.StringIO()
    survey.to_tfs(default)
    survey.to_tfs(workaround, exclude='line')
    assert default.getvalue() == workaround.getvalue()

    spec = {
        'version': '1',
        'columns': [{'name': 'name'}, {'name': 's'}],
        'scalars': [{'name': 'element0'}],
    }
    target = io.StringIO()
    survey.to_tfs(target, format_spec=spec)
    target.seek(0)
    loaded = xt.Table.from_tfs(target)
    np.testing.assert_array_equal(loaded.name, survey.name)
    np.testing.assert_allclose(loaded.s, survey.s)
    assert loaded.element0 == survey.element0
    assert survey.line is line
