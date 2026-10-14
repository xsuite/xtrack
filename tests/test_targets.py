import warnings

import pytest

import xtrack as xt
from xtrack.general import DEPRECATION_INFO_PREP_1_0


@pytest.mark.parametrize('weight', [None, 0, 2.5])
def test_separation_targets_weight(weight):
    with warnings.catch_warnings():
        warnings.simplefilter('error', FutureWarning)
        orthogonal = xt.TargetSeparationOrthogonalToCrossing('ip1')
        separation = xt.TargetSeparation(
            'ip1', separation=1e-3, plane='x', tol=1e-6, weight=weight)

    assert orthogonal.weight == 1
    assert separation.weight == weight
    assert separation.value == 1e-3
    assert separation.tol == 1e-6


@pytest.mark.parametrize('scale', [0, 2.5])
@pytest.mark.parametrize('positional', [False, True])
def test_separation_scale_deprecated(scale, positional):
    with pytest.warns(FutureWarning) as caught:
        if positional:
            target = xt.TargetSeparation(
                'ip1', 1e-3, None, 'x', None, None, 1e-6, scale, '<')
        else:
            target = xt.TargetSeparation(
                'ip1', separation=1e-3, plane='x', tol=1e-6,
                scale=scale, ineq_sign='<')

    assert len(caught) == 1
    assert '`weight`' in str(caught[0].message)
    assert DEPRECATION_INFO_PREP_1_0 in str(caught[0].message)
    assert caught[0].filename == __file__
    assert target.weight == scale
    assert target.tol == 1e-6
    assert target.ineq_sign == '<'


def test_separation_rejects_weight_and_scale():
    with pytest.warns(FutureWarning), pytest.raises(ValueError) as caught:
        xt.TargetSeparation(
            'ip1', separation=1e-3, plane='x', weight=2, scale=3)
    assert 'Cannot specify both `weight` and `scale`' in str(caught.value)
