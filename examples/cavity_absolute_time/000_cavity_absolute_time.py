import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo

line = xt.Line.from_json(
    '../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json')
line.build_tracker()

for vv in line.vars.get_table().rows[
    'on_x.*|on_sep.*|on_crab.*|on_alice|on_lhcb|corr_.*'].name:
    line.vars[vv] = 0

line.vars['f_rf'] = 400789598.98582596 + 1
tt = line.get_table()
for nn in tt.rows[tt.element_type=='Cavity'].name:
    line.element_refs[nn].absolute_time = 1
    line.element_refs[nn].frequency = line.vars['f_rf']


tw1 = line.twiss()
line.particle_ref.t_sim = tw1.T_rev0

p = line.build_particles()
p.t_sim = line.particle_ref.t_sim
line.track(p, num_turns=10000, with_progress=True, turn_by_turn_monitor=True)
rec = line.record_last_track