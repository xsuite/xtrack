import xtrack as xt
import numpy as np

# TODO:
# - one/multiple thin elements
# - one/multiple thick elements
# - absolute s
# - relative s
# - at start or at end
# - specify length of the line by place(xt.END, at=...)
# - Archors, transform `refer` into `anchor_default`
# - Check on a sliced line
# - Sort out center/centre
# - What happens with repeated elements

env = xt.Environment()

line = env.new_line(
    components=[
        env.new('q0', 'Quadrupole', length=2.0, at=20.0),
        env.new('ql', 'Quadrupole', length=2.0, at=-10.0, from_='q0'),
        env.new('qr', 'Quadrupole', length=2.0, at=10.0, from_='q0'),
    ])

_is_drift = xt.line._is_drift
_all_places = xt.environment._all_places
_resolve_s_positions = xt.environment._resolve_s_positions

# Insert single thick element at absolute s

env.new('ss', 'Sextupole', length='0.1')
pp_ss = env.place('ss')
what = [
    env.place('q0', at=5.0),
    pp_ss,
    env.place('q0', at=15.0),
    pp_ss]

if len(what) != len(set(what)):
    what = [ww.copy() for ww in what]

tt = line.get_table()
tt['length'] = np.diff(tt._data['s'], append=0)
tt['s_center'] = tt.s + 0.5 * tt.length


line_places = []
for nn in tt.name:
    if nn == '_end_point':
        continue
    if not _is_drift(line.element_dict[nn], line):
        line_places.append(env.place(nn, tt['s_center', nn]))

seq_all_places = _all_places(line_places + what)

tab_sorted = _resolve_s_positions(seq_all_places, env, refer='centre',
                                  allow_duplicate_places=False)