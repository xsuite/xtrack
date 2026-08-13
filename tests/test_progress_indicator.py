import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

import xtrack as xt
import xtrack.aperture.aperture as aperture_module
import xtrack.environment as environment_module
import xtrack.line as line_module
import xtrack.mad_loader as mad_loader_module
import xtrack.progress_indicator as progress_indicator
import xtrack.slicing as slicing_module


class _FakeTqdm:
    def __init__(self, iterable, **options):
        self.iterable = iterable
        self.options = options


def _install_fake_tqdm(monkeypatch):
    monkeypatch.setattr(
        progress_indicator._config, 'default_indicator_cls', _FakeTqdm)


def test_tqdm_enabled_by_default(monkeypatch):
    _install_fake_tqdm(monkeypatch)
    with xt.settings.override(progress_indicator='tqdm'):
        indicator = progress_indicator.progress([], desc='Testing')

    assert isinstance(indicator, _FakeTqdm)


def test_text_indicator_selected_by_settings(monkeypatch):
    _install_fake_tqdm(monkeypatch)
    with xt.settings.override(progress_indicator='text'):
        indicator = progress_indicator.progress([], desc='Testing')

        assert isinstance(
            indicator, progress_indicator.DefaultProgressIndicator)


def test_environment_variable_is_startup_default():
    environment = os.environ.copy()
    environment['XSUITE_PROGRESS_INDICATOR'] = 'text'
    code = (
        'import xtrack as xt; '
        'assert xt.settings.progress_indicator == "text"')

    subprocess.run(
        [sys.executable, '-c', code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )


def test_python_setting_overrides_environment_default():
    environment = os.environ.copy()
    environment['XSUITE_PROGRESS_INDICATOR'] = 'text'
    code = (
        'import xtrack as xt; '
        'xt.settings.progress_indicator = "suppress"; '
        'assert xt.settings.progress_indicator == "suppress"')

    subprocess.run(
        [sys.executable, '-c', code],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )


def test_progress_indicator_suppressed_by_settings():
    iterable = range(3)
    with xt.settings.override(progress_indicator='suppress'):
        indicator = progress_indicator.progress(iterable, desc='Testing')

    assert indicator is iterable


@pytest.mark.parametrize('selected_mode', ['tqdm', 'text'])
def test_print_suppression_also_suppresses_progress(
    monkeypatch, selected_mode,
):
    monkeypatch.setattr(
        progress_indicator._config,
        'default_indicator_cls',
        _unexpected_progress,
    )
    iterable = range(3)

    with xt.settings.override(
        print_mode='suppress',
        progress_indicator=selected_mode,
    ):
        indicator = progress_indicator.progress(iterable, desc='Testing')

    assert indicator is iterable


def test_invalid_progress_indicator_setting():
    with pytest.raises(ValueError, match='expected.*tqdm.*text.*suppress'):
        xt.settings.progress_indicator = 'invalid'


def _unexpected_progress(*args, **kwargs):
    raise AssertionError('The progress indicator should not have been used')


def test_with_progress_false_for_deserialization_and_copy(monkeypatch):
    monkeypatch.setattr(environment_module, 'progress', _unexpected_progress)
    line = xt.Line(
        elements={'q': xt.Quadrupole(length=1)}, element_names=['q'])

    xt.Line.from_dict(
        line.to_dict(), verbose=False, with_progress=False)

    env = xt.Environment(element_dict={'q': xt.Quadrupole(length=1)})
    env.copy(with_progress=False)


def test_with_progress_false_for_line_editing(monkeypatch):
    monkeypatch.setattr(slicing_module, 'progress', _unexpected_progress)

    line = xt.Line(
        elements={'q': xt.Quadrupole(length=1)}, element_names=['q'])
    line.slice_thick_elements(
        [xt.Strategy(xt.Uniform(2))], with_progress=False)

    line = xt.Line(
        elements={'q': xt.Quadrupole(length=1)}, element_names=['q'])
    line.cut_at_s([0.5], with_progress=False)

    env = xt.Environment()
    env.new('q', 'Quadrupole', length=1)
    env.new('m', 'Marker')
    line = env.new_line(components=['q'])
    line.insert('m', at=0.5, with_progress=False)


def test_with_progress_false_for_other_progress_loops(monkeypatch):
    monkeypatch.setattr(environment_module, 'progress', _unexpected_progress)
    monkeypatch.setattr(line_module, 'progress', _unexpected_progress)

    env = xt.Environment()
    with pytest.warns(FutureWarning):
        env.set_multipolar_errors({}, with_progress=False)

    line = xt.Line(
        elements={
            'aper_start': xt.LimitEllipse(a=1, b=1),
            'drift': xt.Drift(length=1),
            'aper_end': xt.LimitEllipse(a=1, b=1),
        },
        element_names=['aper_start', 'drift', 'aper_end'],
    )
    line.check_aperture(with_progress=False)


def test_mad_loader_with_progress_false(monkeypatch):
    monkeypatch.setattr(mad_loader_module, 'progress', _unexpected_progress)
    loader = mad_loader_module.MadLoader.__new__(mad_loader_module.MadLoader)
    loader.sequence = SimpleNamespace(
        _madx=object(),
        beam=SimpleNamespace(bv=1),
        expanded_elements=[None] * 11,
        name='seq',
    )
    loader.classes = SimpleNamespace(Line=xt.Line)
    loader.enable_expressions = False
    loader.enable_layout_data = False
    loader.iter_elements = lambda madeval=None: iter(())

    line = loader.make_line(with_progress=False)

    assert len(line) == 0


def test_aperture_factory_propagates_with_progress_false(monkeypatch):
    monkeypatch.setattr(aperture_module, 'progress', _unexpected_progress)
    build_arguments = {}

    def fake_build_aperture_model(cls, **kwargs):
        build_arguments.update(kwargs)
        return object()

    monkeypatch.setattr(
        aperture_module.Aperture,
        '_build_aperture_model',
        classmethod(fake_build_aperture_model),
    )
    line = xt.Line(
        elements={'aper': xt.LimitEllipse(a=1, b=1)},
        element_names=['aper'],
    )
    monkeypatch.setattr(
        line, 'survey',
        lambda: SimpleNamespace(name=['aper', '_end_point']))

    aperture_module.Aperture.from_line_with_limits(
        line, with_progress=False)

    assert build_arguments['with_progress'] is False
