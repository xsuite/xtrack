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

# General rule: I want to keep anything I can!

def _add_entry_exit_center(tt):
    tt['length'] = np.diff(tt['s'], append=tt['s'][-1])
    tt['s_center'] = tt.s + 0.5 * tt.length
    tt['s_entry'] = tt.s
    tt['s_exit'] = tt.s + tt.length


env = xt.Environment()

line = env.new_line(
    components=[
        env.new('q0', 'Quadrupole', length=2.0, at=20.0),
        env.new('ql', 'Quadrupole', length=2.0, at=-10.0, from_='q0'),
        env.new('qr', 'Quadrupole', length=2.0, at=10.0, from_='q0'),
    ])

s_tol = 1e-10

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

# Resolve s positions
tt = line.get_table()
_add_entry_exit_center(tt)

line_places = []
for nn in tt.name:
    if nn == '_end_point':
        continue
    # if _is_drift(line.element_dict[nn], line):
    #     continue
    line_places.append(env.place(nn, tt['s_center', nn]))

seq_all_places = _all_places(line_places + what)

tab_sorted = _resolve_s_positions(seq_all_places, env, refer='centre',
                                  # I will use the ids of the places afterwards, hence:
                                  allow_duplicate_places=False)

assert len(seq_all_places) == len(tab_sorted)

# Get table with new insertions only
idx_insertions = []
for ii in range(len(tab_sorted)):
    if tab_sorted['place_obj', ii] in what:
        idx_insertions.append(ii)
tab_insertions = tab_sorted.rows[idx_insertions]

# Make cuts
s_cuts = list(tab_insertions['s_entry']) + list(tab_insertions['s_exit'])
s_cuts = list(set(s_cuts))
line.cut_at_s(s_cuts, s_tol=1e-06)

tt_after_cut = line.get_table()
_add_entry_exit_center(tt_after_cut)

# Identify old elements falling inside the insertions
idx_remove = []
for ii in range(len(tab_insertions)):
    s_ins_entry = tab_insertions['s_entry', ii]
    s_ins_exit = tab_insertions['s_exit', ii]

    entry_is_inside = ((tt_after_cut.s_entry >= s_ins_entry - s_tol)
                     & (tt_after_cut.s_entry <= s_ins_exit - s_tol))
    exit_is_inside = ((tt_after_cut.s_exit >= s_ins_entry + s_tol)
                    & (tt_after_cut.s_exit <= s_ins_exit + s_tol))
    idx_remove.extend(list(np.where(entry_is_inside | exit_is_inside)[0]))

# TODO: Remember to handle adjacent markers

places_to_keep = []
for ii in range(len(tt_after_cut)):
    nn = tt_after_cut['name', ii]
    if ii in idx_remove:
        continue
    if nn == '_end_point':
        continue
    places_to_keep.append(env.place(nn, at=tt_after_cut['s_center', ii]))

l_aux = env.new_line(components=[places_to_keep + what])
