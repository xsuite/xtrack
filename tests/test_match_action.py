from unittest.mock import Mock

import pytest

import xtrack as xt
from xtrack.match import ActionTwiss


@pytest.mark.parametrize('fail_first', [False, True])
def test_action_twiss_prepare(monkeypatch, fail_first):
    line = xt.Line(elements=[xt.LineSegmentMap(qx=0.31, qy=0.32)])
    line.particle_ref = xt.Particles(p0c=1e9)
    line.build_tracker()
    twiss = Mock(wraps=line.twiss)
    monkeypatch.setattr(line, 'twiss', twiss)
    action = ActionTwiss(line, method='4d')

    if fail_first:
        twiss.side_effect = RuntimeError('Preparation failed')
        with pytest.raises(RuntimeError):
            action.prepare()
        assert not action._already_prepared
        twiss.side_effect = None
        twiss.reset_mock()

    action.prepare()
    assert twiss.call_count == 1
    initial_twiss = action._tw0

    action.prepare()
    assert twiss.call_count == 1
    assert action._tw0 is initial_twiss

    action.prepare(force=True)
    assert twiss.call_count == 2
    assert action._tw0 is not initial_twiss

    action.prepare()
    assert twiss.call_count == 2
