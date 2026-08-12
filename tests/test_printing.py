import numpy as np

import xdeps as xd
import xobjects as xo
import xtrack as xt
from xtrack.mad_parser.parse import warn


def test_xtrack_exposes_shared_settings():
    assert xt.settings is xo.settings


def test_xtrack_print_mode(capsys, monkeypatch):
    with xt.settings.override(print_mode='suppress'):
        xt._print('hidden')

    assert capsys.readouterr().out == ''


def test_xtrack_print_mode_can_be_overridden(capsys):
    with xt.settings.override(print_mode='suppress'):
        xt.settings.print_mode = 'print'
        xt._print('visible')

    assert capsys.readouterr().out == 'visible\n'


def test_xtrack_runtime_output_uses_configured_print(capsys, monkeypatch):
    with xt.settings.override(print_mode='suppress'):
        warn('hidden')

    assert capsys.readouterr().out == ''


def test_xtrack_optimizer_uses_configured_print(capsys):
    with xt.settings.override(print_mode='suppress'):
        opt = xt.match.opt_from_callable(
            lambda value: value,
            x0=[1.],
            steps=[1e-6],
            tar=[0.],
            tols=[1e-12],
        )
        opt.vary_status()

    assert capsys.readouterr().out == ''


def test_xtrack_line_optimizer_uses_configured_print(capsys):
    environment = xt.Environment()
    environment['k'] = 0.

    class TestAction(xd.Action):
        def run(self):
            return {'value': environment['k']}

    with xt.settings.override(print_mode='suppress'):
        opt = xt.match.OptimizeLine(
            line=environment,
            vary=[xt.Vary('k', container=environment.vars, step=1e-6)],
            targets=[xt.Target('value', value=1., action=TestAction())],
        )
        opt.vary_status()

    assert capsys.readouterr().out == ''


def test_xtrack_table_uses_configured_print(capsys):
    table = xt.Table({'name': np.array(['a']), 'value': np.array([1])})

    with xt.settings.override(print_mode='suppress'):
        table.show()

    assert capsys.readouterr().out == ''
