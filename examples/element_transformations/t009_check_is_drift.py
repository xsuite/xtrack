import numpy as np
import xtrack as xt

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