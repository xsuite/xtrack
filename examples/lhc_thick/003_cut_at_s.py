import numpy as np
import xtrack as xt

line = xt.Line.from_json('lhc_thin.json')
tw0 = line.twiss()

line.discard_tracker()

s0 = 'mq.28r3.b1_entry'
s1 = 'mq.29r3.b1_exit'

s_cuts = np.linspace(line.get_s_position(s0), line.get_s_position(s1), 100)

s_cuts = np.sort(s_cuts)

s_tol = 0.5e-6

tt = line.get_table()

i_next = np.array([np.argmax(tt['s'] > s_cut) for s_cut in s_cuts])
i_ele_containing = i_next - 1

needs_cut = np.abs(tt['s'][i_ele_containing] - s_cuts) > s_tol

assert np.all(s_cuts[needs_cut] > tt.s[i_ele_containing[needs_cut]])
assert np.all(s_cuts[needs_cut] < tt.s[i_ele_containing[needs_cut]+1])
assert np.all(tt.element_type[i_ele_containing[needs_cut]] == 'Drift')

i_drifts_to_cut = set(i_ele_containing[needs_cut])

for idr in i_drifts_to_cut:
    name_drift = tt.name[idr]
    drift = line[name_drift]
    assert isinstance(drift, xt.Drift)
    _buffer = drift._buffer
    l_drift = drift.length
    s_start = tt['s'][idr]
    s_end = s_start + l_drift
    s_cut_dr = np.sort([s_start] + list(s_cuts[i_ele_containing==idr]) + [s_end])

    drifts_for_replacement = []
    i_new_drifts = 0
    new_drift_names = []
    for ll in np.diff(s_cut_dr):
        if ll > s_tol:
            drifts_for_replacement.append(xt.Drift(length=ll, _buffer=_buffer))
            new_drift_names.append(f'{name_drift}_{i_new_drifts}')
            assert new_drift_names[-1] not in line.element_names
            i_new_drifts += 1

    insert_at = line.element_names.index(name_drift)
    line.element_names.remove(name_drift)
    for nn, dd in zip(new_drift_names, drifts_for_replacement):
        line.element_dict[nn] = dd
        line.element_names.insert(insert_at, nn)
        insert_at += 1

line.build_tracker()
