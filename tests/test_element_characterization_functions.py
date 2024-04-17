import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

@for_all_test_contexts
def test_is_aperture(test_context):
    elements={
        'e0': xt.Bend(k0=0.4, h=0.3, length=1),
        'e1': xt.Replica(parent_name='e0'),
        'e2': xt.Replica(parent_name='e1'),
        'a0': xt.LimitRect(),
        'a1': xt.Replica(parent_name='a0'),
        'a2': xt.Replica(parent_name='a1'),
    }

    line = xt.Line(elements=elements, element_names=list(elements.keys()))
    line.build_tracker(_context=test_context)

    _is_aperture = xt.line._is_aperture

    assert not _is_aperture(line['e0'], line)
    assert not _is_aperture(line['e1'], line)
    assert not _is_aperture(line['e2'], line)
    assert _is_aperture(line['a0'], line)
    assert _is_aperture(line['a1'], line)
    assert _is_aperture(line['a2'], line)


@for_all_test_contexts
def test_has_backtrack(test_context):
    elements={
        'e0': xt.Bend(k0=0.4, h=0.3, length=1),
        'e1': xt.Replica(parent_name='e0'),
        'e2': xt.Replica(parent_name='e1'),
        'a0': xt.Drift(),
        'a1': xt.Replica(parent_name='a0'),
        'a2': xt.Replica(parent_name='a1'),
    }

    line = xt.Line(elements=elements, element_names=list(elements.keys()))
    line.build_tracker(_context=test_context)

    _has_backtrack = xt.line._has_backtrack
    _allow_backtrack = xt.line._allow_backtrack

    assert _has_backtrack(line['e0'], line)
    assert _has_backtrack(line['e1'], line)
    assert _has_backtrack(line['e2'], line)
    assert _has_backtrack(line['a0'], line)
    assert _has_backtrack(line['a1'], line)
    assert _has_backtrack(line['a2'], line)

    assert not _allow_backtrack(line['e0'], line)
    assert not _allow_backtrack(line['e1'], line)
    assert not _allow_backtrack(line['e2'], line)
    assert _allow_backtrack(line['a0'], line)
    assert _allow_backtrack(line['a1'], line)
    assert _allow_backtrack(line['a2'], line)