import xtrack as xt
import numpy as np

def _plot_line(line):
    tt = line.get_table(attr=True)
    xt.twiss.TwissTable.plot(tt, yl='', yr='')

class Place:
    def __init__(self, name, at=None, from_=None, anchor=None, from_anchor=None):

        if anchor is not None:
            raise ValueError('anchor not implemented')
        if from_anchor is not None:
            raise ValueError('from_anchor not implemented')

        self.name = name
        self.at = at
        self.from_ = from_
        self.anchor = anchor
        self.from_anchor = from_anchor
        self._before = False

    def __repr__(self):
        return f'Place({self.name}, at={self.at}, from_={self.from_})'

env = xt.Environment()

seq = [
    Place(env.new_element('ip', xt.Marker), at=10),
    # Place(env.new_element('right',xt.Quadrupole, length=1), at=+5, from_='ip'),
    (
        env.new_element('before_before_right', xt.Marker),
        env.new_element('before_right', xt.Sextupole, length=1),
        Place(env.new_element('right',xt.Quadrupole, length=1), at=+5, from_='ip'),
        env.new_element('after_right', xt.Marker),
        env.new_element('after_right2', xt.Marker),
    ),
    Place(env.new_element('left', xt.Quadrupole, length=1), at=-5, from_='ip'),
    env.new_element('after_left', xt.Marker),
    env.new_element('after_left2', xt.Bend, length=0.5),
]

def _all_places(seq):
    seq_all_places = []
    for ss in seq:
        if isinstance(ss, Place):
            seq_all_places.append(ss)
        elif not isinstance(ss, str) and hasattr(ss, '__iter__'):
            # Find first place
            i_first = None
            for ii, sss in enumerate(ss):
                if isinstance(sss, Place):
                    i_first = ii
                    break
            if i_first is None:
                raise ValueError('No Place in sequence')
            ss_aux = _all_places(ss)
            for ii in range(i_first):
                ss_aux[ii]._before = True
            seq_all_places.extend(ss_aux)
        else:
            seq_all_places.append(Place(ss, at=None, from_=None))
    return seq_all_places

seq_all_places = _all_places(seq)


names_unsorted = [ss.name for ss in seq_all_places]
aux_line = env.new_line(components=names_unsorted)
aux_tt = aux_line.get_table()
aux_tt['length'] = np.diff(aux_tt._data['s'], append=0)

s_center_dct = {}
n_resolved = 0
n_resolved_prev = -1
while n_resolved != n_resolved_prev:
    n_resolved_prev = n_resolved
    for ii, ss in enumerate(seq_all_places):
        if ss.name in s_center_dct:
            continue
        if ss.at is None and not ss._before:
            ss_prev = seq_all_places[ii-1]
            if ss_prev.name in s_center_dct:
                s_center_dct[ss.name] = (s_center_dct[ss_prev.name]
                                         + aux_tt['length', ss_prev.name] / 2
                                         + aux_tt['length', ss.name] / 2)
                n_resolved += 1
        elif ss.at is None and ss._before:
            ss_next = seq_all_places[ii+1]
            if ss_next.name in s_center_dct:
                s_center_dct[ss.name] = (s_center_dct[ss_next.name]
                                         - aux_tt['length', ss_next.name] / 2
                                         - aux_tt['length', ss.name] / 2)
                n_resolved += 1
        elif ss.from_ is None:
            s_center_dct[ss.name] = ss.at
            n_resolved += 1
        else:
            if ss.from_ in s_center_dct:
                s_center_dct[ss.name] = s_center_dct[ss.from_] + ss.at
                n_resolved += 1

assert n_resolved == len(seq_all_places)

aux_s_center = np.array([s_center_dct[nn] for nn in aux_tt.name[:-1]])
aux_tt['s_center'] = np.concatenate([aux_s_center, [0]])

i_sorted = np.argsort(aux_s_center, stable=True)

name_sorted = [str(aux_tt.name[ii]) for ii in i_sorted]

tt_sorted = aux_tt.rows[name_sorted]
tt_sorted['s_entry'] = tt_sorted['s_center'] - tt_sorted['length'] / 2
tt_sorted['s_exit'] = tt_sorted['s_center'] + tt_sorted['length'] / 2
tt_sorted['ds_upstream'] = 0 * tt_sorted['s_entry']
tt_sorted['ds_upstream'][1:] = tt_sorted['s_entry'][1:] - tt_sorted['s_exit'][:-1]
tt_sorted['ds_upstream'][0] = tt_sorted['s_entry'][0]
tt_sorted['s'] = tt_sorted['s_center']
assert np.all(tt_sorted.name == np.array(name_sorted))

s_tol = 1e-12
names_with_drifts = []
# Create drifts
for nn in name_sorted:
    ds_upstream = tt_sorted['ds_upstream', nn]
    if np.abs(ds_upstream) > s_tol:
        assert ds_upstream > 0, f'Negative drift length: {ds_upstream}, upstream of {nn}'
        drift_name = env._get_a_drift_name()
        drift = env.new_element(drift_name, xt.Drift, length=ds_upstream)
        names_with_drifts.append(drift_name)
    names_with_drifts.append(nn)

line = env.new_line(components=names_with_drifts)

