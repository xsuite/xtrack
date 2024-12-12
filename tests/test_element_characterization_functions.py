import xtrack as xt
import numpy as np
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
    _allow_loss_refinement = xt.line._allow_loss_refinement

    assert _has_backtrack(line['e0'], line)
    assert _has_backtrack(line['e1'], line)
    assert _has_backtrack(line['e2'], line)
    assert _has_backtrack(line['a0'], line)
    assert _has_backtrack(line['a1'], line)
    assert _has_backtrack(line['a2'], line)

    assert not _allow_loss_refinement(line['e0'], line)
    assert not _allow_loss_refinement(line['e1'], line)
    assert not _allow_loss_refinement(line['e2'], line)
    assert _allow_loss_refinement(line['a0'], line)
    assert _allow_loss_refinement(line['a1'], line)
    assert _allow_loss_refinement(line['a2'], line)

def test_is_drift_behaves_like_drift():

    bend = xt.Bend(k0=0.4, h=0.3, length=1)

    line = xt.Line(
        elements=[bend, xt.Replica(parent_name='e0'), xt.Replica(parent_name='e1'),
                            xt.Drift(length=1.), xt.Replica(parent_name='e3'),
                            xt.Replica(parent_name='e4')])
    length = line.get_length()

    tt = line.get_table()

    assert np.all(tt['name'][:-1] == ['e0', 'e1', 'e2', 'e3', 'e4', 'e5'])

    _is_drift = xt.line._is_drift
    _behaves_like_drift = xt.line._behaves_like_drift

    assert not _is_drift(line['e0'], line)
    assert not _is_drift(line['e1'], line)
    assert not _is_drift(line['e2'], line)
    assert _is_drift(line['e3'], line)
    assert _is_drift(line['e4'], line)
    assert _is_drift(line['e5'], line)

    assert not _behaves_like_drift(line['e0'], line)
    assert not _behaves_like_drift(line['e1'], line)
    assert not _behaves_like_drift(line['e2'], line)
    assert _behaves_like_drift(line['e3'], line)
    assert _behaves_like_drift(line['e4'], line)
    assert _behaves_like_drift(line['e5'], line)

    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(xt.Uniform(3), name='e0'),
            xt.Strategy(xt.Uniform(3, mode='thick'), name='e1'),
            xt.Strategy(xt.Uniform(3), name='e2')])
    assert_allclose = np.testing.assert_allclose
    assert_allclose(length, line.get_length(), rtol=0, atol=1e-15)

    assert np.all(line.get_table().name == [
        'e0_entry', 'e0..entry_map', 'drift_e0..0', 'e0..0', 'drift_e0..1',
        'e0..1', 'drift_e0..2', 'e0..2', 'drift_e0..3', 'e0..exit_map',
        'e0_exit', 'e1_entry', 'e1..entry_map', 'e1..0', 'e1..1', 'e1..2',
        'e1..exit_map', 'e1_exit', 'e2_entry', 'e2..entry_map',
        'drift_e2..0', 'e2..0', 'drift_e2..1', 'e2..1', 'drift_e2..2',
        'e2..2', 'drift_e2..3', 'e2..exit_map', 'e2_exit', 'e3', 'e4',
        'e5', '_end_point'])

    for nn, ee in line.items():
        if nn.startswith('drift'):
            assert _is_drift(ee, line)
            assert _behaves_like_drift(ee, line)
        elif nn in ['e3', 'e4', 'e5']:
            assert _is_drift(ee, line)
            assert _behaves_like_drift(ee, line)
        elif nn.endswith('_entry') or nn.endswith('_exit'): # Markers
            assert not _is_drift(ee, line)
            assert _behaves_like_drift(ee, line)
        else:
            assert not _is_drift(ee, line)
            assert not _behaves_like_drift(ee, line)

    line.cut_at_s([0.6, 1.5])
    assert_allclose(length, line.get_length(), rtol=0, atol=1e-15)
    assert_allclose(line.get_s_position('drift_e0..2..1'), 0.6, rtol=0, atol=1e-15)
    assert_allclose(line.get_s_position('e1..1..1'), 1.5, rtol=0, atol=1e-15)

    assert _is_drift(line['drift_e0..2..1'], line)
    assert _behaves_like_drift(line['drift_e0..2..1'], line)
    assert not _is_drift(line['e1..1..1'], line)
    assert not _behaves_like_drift(line['e1..1..1'], line)

    line.slice_thick_elements(
        slicing_strategies=[
            xt.Strategy(None),
            xt.Strategy(xt.Uniform(2), name='e3'),
            xt.Strategy(xt.Uniform(3, mode='thick'), name='e4'),
            xt.Strategy(xt.Uniform(4), name='e5')])

    assert np.all(line.get_table().name == [
        'e0_entry', 'e0..entry_map', 'drift_e0..0', 'e0..0', 'drift_e0..1',
        'e0..1', 'drift_e0..2..0', 'drift_e0..2..1', 'e0..2',
        'drift_e0..3', 'e0..exit_map', 'e0_exit', 'e1_entry',
        'e1..entry_map', 'e1..0', 'e1..1_entry', 'e1..1..0', 'e1..1..1',
        'e1..1_exit', 'e1..2', 'e1..exit_map', 'e1_exit', 'e2_entry',
        'e2..entry_map', 'drift_e2..0', 'e2..0', 'drift_e2..1', 'e2..1',
        'drift_e2..2', 'e2..2', 'drift_e2..3', 'e2..exit_map', 'e2_exit',
        'drift_e3..0', 'drift_e3..1', 'drift_e3..2', 'e4_entry', 'e4..0',
        'e4..1', 'e4..2', 'e4_exit', 'e5_entry', 'drift_e5..0',
        'drift_e5..1', 'drift_e5..2', 'drift_e5..3', 'drift_e5..4',
        'e5_exit', '_end_point'])

    for nn, ee in line.items():
        if nn.startswith('drift'):
            assert _is_drift(ee, line)
            assert _behaves_like_drift(ee, line)
        elif nn in ['e4..0', 'e4..1', 'e4..2']:
            assert _is_drift(ee, line)
            assert _behaves_like_drift(ee, line)
        elif nn.endswith('_entry') or nn.endswith('_exit'): # Markers
            assert not _is_drift(ee, line)
            assert _behaves_like_drift(ee, line)
        else:
            assert not _is_drift(ee, line)
            assert not _behaves_like_drift(ee, line)

def test_replica_loops():

    bend = xt.Bend(k0=0.4, h=0.3, length=1)

    elements = {
        # correct case
        'a0': bend,
        'a1': xt.Replica(parent_name='a0'),
        'a2': xt.Replica(parent_name='a1'),
        # create a simple loop
        'b0': xt.Replica(parent_name='b0'),
        # loop with two replicas
        'c0': xt.Replica(parent_name='c1'),
        'c1': xt.Replica(parent_name='c0'),
        # bigger loop
        'd0': xt.Replica(parent_name='d1'),
        'd1': xt.Replica(parent_name='d2'),
        'd2': xt.Replica(parent_name='d0'),
        'd3': xt.Replica(parent_name='d1'),
    }

    line = xt.Line(elements=elements, element_names=list(elements.keys()))

    ok = ['a1', 'a2']
    for kk in ok:
        assert line[kk].resolve(line) is line.get('a0')

    error = ['b0', 'c0', 'c1', 'd0', 'd1', 'd2', 'd3']
    for kk in error:
        try:
            line[kk].resolve(line)
        except RecursionError:
            pass
        else:
            raise Exception(f'Element {kk} should not be resolvable')