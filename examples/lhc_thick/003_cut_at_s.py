import time
import numpy as np
import xtrack as xt

line = xt.Line.from_json('lhc_thin.json')
tw0 = line.twiss()

line.discard_tracker()

e0 = 'mq.28r3.b1_entry'
e1 = 'mq.29r3.b1_exit'

s0 = line.get_s_position(e0)
s1 = line.get_s_position(e1)

elements_to_insert = [
    # s .    # elements to insert (name, element)
    (s0,     [(f'm0_at_a', xt.Marker()), (f'm1_at_a', xt.Marker()), (f'm2_at_a', xt.Marker())]),
    (s0+10., [(f'm0_at_b', xt.Marker()), (f'm1_at_b', xt.Marker()), (f'm2_at_b', xt.Marker())]),
    (s1,     [(f'm0_at_c', xt.Marker()), (f'm1_at_c', xt.Marker()), (f'm2_at_c', xt.Marker())]),
]

# s_cuts = np.linspace(e0, e1, 100)

s_cuts = [ee[0] for ee in elements_to_insert]

s_cuts = np.sort(s_cuts)

s_tol = 0.5e-6

tt_before_cut = line.get_table()

i_next = np.array([np.argmax(tt_before_cut['s'] > s_cut) for s_cut in s_cuts])
i_ele_containing = i_next - 1

needs_cut = np.abs(tt_before_cut['s'][i_ele_containing] - s_cuts) > s_tol

assert np.all(s_cuts[needs_cut] > tt_before_cut.s[i_ele_containing[needs_cut]])
assert np.all(s_cuts[needs_cut] < tt_before_cut.s[i_ele_containing[needs_cut]+1])
assert np.all(tt_before_cut.element_type[i_ele_containing[needs_cut]] == 'Drift')

i_drifts_to_cut = set(i_ele_containing[needs_cut])

t1 = time.time()
for idr in i_drifts_to_cut:
    name_drift = tt_before_cut.name[idr]
    drift = line[name_drift]
    assert isinstance(drift, xt.Drift)
    _buffer = drift._buffer
    l_drift = drift.length
    s_start = tt_before_cut['s'][idr]
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
    cpd_name = line.compound_container.compound_name_for_element(name_drift)
    if cpd_name is not None:
        cpd = line.compound_container.compound_for_name(cpd_name)
        assert name_drift in cpd.core
        cpd.core.remove(name_drift)
    else:
        cpd = None
    for nn, dd in zip(new_drift_names, drifts_for_replacement):
        line.element_dict[nn] = dd
        line.element_names.insert(insert_at, nn)
        if cpd is not None:
            cpd.core.add(nn)
            line.compound_container._compound_name_for_element[nn] = cpd_name
        insert_at += 1
t2 = time.time()

tt_after_cut = line.get_table()

# Names for insertions
ele_name_insertions = []
for s_insert, ee in elements_to_insert:
    # Find element_name for insertion
    ii_ins = np.where(tt_after_cut['s'] >= s_insert)[0][0]
    ele_name_insertions.append(tt_after_cut['name'][ii_ins])
    assert np.abs(s_insert - tt_after_cut['s'][ii_ins]) < s_tol

# Add all elements to line.element_dict
for s_insert, ee in elements_to_insert:
    for nn, el in ee:
        assert nn not in line.element_names
        line.element_dict[nn] = el

# Insert elements
for i_ins, (s_insert, ee) in enumerate(elements_to_insert):
    ele_name_ins = ele_name_insertions[i_ins]
    cpd_name_ins = line.compound_container.compound_name_for_element(ele_name_ins)
    if cpd_name_ins is not None:
        cpd_ins = line.compound_container.compound_for_name(cpd_name_ins)
    else:
        cpd_ins = None

    insert_at = line.element_names.index(ele_name_ins)
    for nn, el in ee:

        assert el.isthick == False
        line.element_names.insert(insert_at, nn)

        if cpd_ins is None:
            pass # No compound
        elif ele_name_ins in cpd_ins.core:
            cpd_ins.core.add(nn)
            line.compound_container._compound_name_for_element[nn] = cpd_name_ins
        elif ele_name_ins in cpd.entry:
            pass # Goes in front ot the compound but does not belong to it
        elif ele_name_ins in cpd.exit:
            assert len(cpd.exit_transform) == 0
            cpd.core.add(nn)
            line.compound_container._compound_name_for_element[nn] = cpd_name_ins
        elif ele_name_ins in cpd.exit_transform:
            cpd.core.add(nn)
            line.compound_container._compound_name_for_element[nn] = cpd_name_ins
        else:
            raise ValueError(f'Inconsistent insertion in compound {cpd_name_ins}')

        insert_at += 1

tt_final = line.get_table()

print(f'Time cut drifts: {t2-t1}')

line.build_tracker()
