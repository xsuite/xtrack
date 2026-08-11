import xtrack as xt
from xtrack.mad_parser.parse import warn


def test_xtrack_print_mode(capsys, monkeypatch):
    monkeypatch.delenv('XSUITE_PRINT_MODE', raising=False)
    monkeypatch.setattr(xt._print, 'mode', 'suppress')

    xt._print('hidden')

    assert capsys.readouterr().out == ''


def test_xtrack_print_mode_environment_variable(capsys, monkeypatch):
    monkeypatch.setenv('XSUITE_PRINT_MODE', 'suppress')

    xt._print('hidden')

    assert capsys.readouterr().out == ''


def test_xtrack_runtime_output_uses_configured_print(capsys, monkeypatch):
    monkeypatch.delenv('XSUITE_PRINT_MODE', raising=False)
    monkeypatch.setattr(xt._print, 'mode', 'suppress')

    warn('hidden')

    assert capsys.readouterr().out == ''
