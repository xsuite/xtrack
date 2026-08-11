import pytest

import xtrack.progress_indicator as progress_indicator


class _FakeTqdm:
    def __init__(self, iterable, **options):
        self.iterable = iterable
        self.options = options


def _install_fake_tqdm(monkeypatch):
    monkeypatch.setattr(
        progress_indicator._config, 'default_indicator_cls', _FakeTqdm)


def test_tqdm_enabled_by_default(monkeypatch):
    _install_fake_tqdm(monkeypatch)
    monkeypatch.delenv('XTRACK_PROGRESS_INDICATOR', raising=False)

    indicator = progress_indicator.progress([], desc='Testing')

    assert isinstance(indicator, _FakeTqdm)


def test_text_indicator_selected_by_module_attribute(monkeypatch):
    _install_fake_tqdm(monkeypatch)
    monkeypatch.delenv('XTRACK_PROGRESS_INDICATOR', raising=False)
    monkeypatch.setattr(progress_indicator, 'mode', 'text')

    indicator = progress_indicator.progress([], desc='Testing')

    assert isinstance(indicator, progress_indicator.DefaultProgressIndicator)


def test_text_indicator_selected_by_environment_variable(monkeypatch):
    _install_fake_tqdm(monkeypatch)
    monkeypatch.setenv('XTRACK_PROGRESS_INDICATOR', 'text')

    indicator = progress_indicator.progress([], desc='Testing')

    assert isinstance(indicator, progress_indicator.DefaultProgressIndicator)


def test_environment_variable_takes_precedence(monkeypatch):
    _install_fake_tqdm(monkeypatch)
    monkeypatch.setattr(progress_indicator, 'mode', 'text')
    monkeypatch.setenv('XTRACK_PROGRESS_INDICATOR', 'tqdm')

    indicator = progress_indicator.progress([], desc='Testing')

    assert isinstance(indicator, _FakeTqdm)


def test_progress_indicator_suppressed_by_module_attribute(monkeypatch):
    iterable = range(3)
    monkeypatch.delenv('XTRACK_PROGRESS_INDICATOR', raising=False)
    monkeypatch.setattr(progress_indicator, 'mode', 'suppress')

    indicator = progress_indicator.progress(iterable, desc='Testing')

    assert indicator is iterable


def test_progress_indicator_suppressed_by_environment_variable(monkeypatch):
    iterable = range(3)
    monkeypatch.setenv('XTRACK_PROGRESS_INDICATOR', 'suppress')

    indicator = progress_indicator.progress(iterable, desc='Testing')

    assert indicator is iterable


def test_invalid_progress_indicator_mode(monkeypatch):
    monkeypatch.delenv('XTRACK_PROGRESS_INDICATOR', raising=False)
    monkeypatch.setattr(progress_indicator, 'mode', 'invalid')

    with pytest.raises(ValueError, match='expected.*tqdm.*text.*suppress'):
        progress_indicator.progress([], desc='Testing')
