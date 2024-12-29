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

env = xt.Environment()

line = env.new_line(
    components=[
        env.new('q0', 'Quadrupole', length=2.0, at=20.0),
        env.new('ql', 'Quadrupole', length=2.0, at=-10.0, from_='q0'),
        env.new('qr', 'Quadrupole', length=2.0, at=10.0, from_='q0'),
        env.new('mk1', 'Marker', at=40),
        env.new('mk2', 'Marker', at=42),
        env.new('end', 'Marker', at=50.),
    ])

s_tol = 1e-10

_is_drift = xt.line._is_drift
_all_places = xt.environment._all_places
_resolve_s_positions = xt.environment._resolve_s_positions
_sort_places = xt.environment._sort_places

# Insert single thick element at absolute s

env.new('ss', 'Sextupole', length='0.1')
pp_ss = env.place('ss')
what = [
    env.place('q0', at=5.0),
    pp_ss,
    env.place('q0', at=15.0),
    pp_ss,
    env.place('q0', at=41.0),
    pp_ss,
]

if len(what) != len(set(what)):
    what = [ww.copy() for ww in what]

# Resolve s positions of insertions
tt = line.get_table()

line_places = []
for nn in tt.name:
    if nn == '_end_point':
        continue
    line_places.append(env.place(nn, at= tt['s_center', nn]))

seq_all_places = _all_places(line_places + what)
mask_insertions = np.array([pp in what for pp in seq_all_places])

tab_unsorted = _resolve_s_positions(seq_all_places, env, refer='centre')
prrrr


tab_unsorted['is_insertion'] = mask_insertions
tab_sorted = _sort_places(tab_unsorted)

assert len(seq_all_places) == len(tab_sorted)

# Get table with new insertions only
tab_insertions = tab_sorted.rows[tab_sorted.is_insertion]

# Make cuts
s_cuts = list(tab_insertions['s_start']) + list(tab_insertions['s_end'])
s_cuts = list(set(s_cuts))
line.cut_at_s(s_cuts, s_tol=1e-06)

tt_after_cut = line.get_table()

# Identify old elements falling inside the insertions
idx_remove = []
for ii in range(len(tab_insertions)):
    s_ins_start = tab_insertions['s_start', ii]
    s_ins_end = tab_insertions['s_end', ii]
    entry_is_inside = ((tt_after_cut.s_start >= s_ins_start - s_tol)
                     & (tt_after_cut.s_start <= s_ins_end - s_tol))
    exit_is_inside = ((tt_after_cut.s_end >= s_ins_start + s_tol)
                    & (tt_after_cut.s_end <= s_ins_end + s_tol))
    thin_at_entry = ((tt_after_cut.s_start >= s_ins_start - s_tol)
                    & (tt_after_cut.s_end <= s_ins_start + s_tol))
    thin_at_exit = ((tt_after_cut.s_start >= s_ins_end - s_tol)
                  & (tt_after_cut.s_end <= s_ins_end + s_tol))
    remove = (entry_is_inside | exit_is_inside) & (~thin_at_entry) & (~thin_at_exit)
    idx_remove.extend(list(np.where(remove)[0]))

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

line.discard_tracker()
line.element_names.clear()
line.element_names.extend(l_aux.element_names)
