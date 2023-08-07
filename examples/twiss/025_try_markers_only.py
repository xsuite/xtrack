import numpy as np
import xtrack as xt

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()
collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

# Check on b1 (no reverse)

line = collider.lhcb1

tw_init_start = line.twiss().get_twiss_init('s.ds.l5.b1')
tw_init_end = line.twiss().get_twiss_init('e.ds.r5.b1')

tw = line.twiss(ele_start='s.ds.l5.b1', ele_stop='e.ds.r5.b1', twiss_init=tw_init_start)
tw2 = line.twiss(ele_start='s.ds.l5.b1', ele_stop='e.ds.r5.b1', twiss_init=tw_init_end)

tw_mk = line.twiss(ele_start='s.ds.l5.b1', ele_stop='e.ds.r5.b1', twiss_init=tw_init_start,
                   only_markers=True)
tw2_mk = line.twiss(ele_start='s.ds.l5.b1', ele_stop='e.ds.r5.b1', twiss_init=tw_init_end,
                    only_markers=True)

# Check names are the right ones
ltable = line.get_table()
expected_names = np.concatenate([
    ltable.rows[ltable.element_type == 'Marker'].rows['s.ds.l5.b1':'e.ds.r5.b1'].name,
    ['_end_point']])

assert np.all(tw_mk.name == expected_names)
assert np.all(tw2_mk.name == expected_names)
assert np.all(tw2.name == tw.name)

# Consistency checks on other columns
for tt in [tw, tw2, tw_mk, tw2_mk]:
    assert tw.name[0] == 's.ds.l5.b1'
    assert tw.name[-1] == '_end_point'
    assert tw.name[-2] == 'e.ds.r5.b1'

    assert tt['s', 'e.ds.r5.b1'] == line.get_s_position('e.ds.r5.b1')
    assert tt['s', 'e.ds.r5.b1'] == tt['s', '_end_point']
    assert tt['s', 's.ds.l5.b1'] == line.get_s_position('s.ds.l5.b1')

    for kk in tw._col_names:
        if kk == 'name':
            continue
        atol = dict(alfx=1e-7, alfy=1e-7, dx=1e-7, dy=1e-7, dpx=1e8, dpy=1e-8,
                    dx_zeta=1e-8, W_matrix=1e-7).get(kk, 1e-10)
        assert np.allclose(tt[kk], tw.rows[tt.name][kk], rtol=1e-6, atol=atol)

line = collider.lhcb2

tw_init_ip5 = line.twiss().get_twiss_init('s.ds.l5.b2')

tw = line.twiss(ele_start='s.ds.l5.b2', ele_stop='e.ds.r5.b2', twiss_init=tw_init_ip5)
tw2 = line.twiss(ele_start='s.ds.l5.b2', ele_stop='e.ds.r5.b2', twiss_init=tw_init_ip5)

tw_mk = line.twiss(ele_start='s.ds.l5.b2', ele_stop='e.ds.r5.b2', twiss_init=tw_init_ip5,
                   only_markers=True)
tw2_mk = line.twiss(ele_start='s.ds.l5.b2', ele_stop='e.ds.r5.b2', twiss_init=tw_init_ip5,
                    only_markers=True)

# Check on b2 (with reverse)
# Check names are the right ones
ltable = line.get_table()
expected_names = np.concatenate([
    ltable.rows[ltable.element_type == 'Marker'].rows['e.ds.r5.b2':'s.ds.l5.b2'].name[::-1],
    ['_end_point']])

assert np.all(tw_mk.name == expected_names)
assert np.all(tw2_mk.name == expected_names)
assert np.all(tw2.name == tw.name)

# Consistency checks on other columns
for tt in [tw, tw2, tw_mk, tw2_mk]:
    assert tw.name[0] == 's.ds.l5.b2'
    assert tw.name[-1] == '_end_point'
    assert tw.name[-2] == 'e.ds.r5.b2'

    assert tt['s', 'e.ds.r5.b2'] == line.get_length() - line.get_s_position('e.ds.r5.b2')
    assert tt['s', 'e.ds.r5.b2'] == tt['s', '_end_point']
    assert tt['s', 's.ds.l5.b2'] == line.get_length() - line.get_s_position('s.ds.l5.b2')

    for kk in tw._col_names:
        if kk == 'name':
            continue
        assert np.allclose(tt[kk], tw.rows[tt.name][kk], rtol=1e-6, atol=1e-10)