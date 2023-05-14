import xtrack as xt
import xpart as xp

line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.build_tracker()



# REMEMBER to check also the SPS and to add a global backtrackability check
# when building the tracker



line.vars['on_crab1'] = -190
line.vars['on_crab5'] = -190
line.vars['on_x1'] = 130
line.vars['on_x5'] = 130

p = xp.Particles(
    p0c=7000e9, x=1e-4, px=1e-6, y=2e-4, py=3e-6, zeta=0.01, delta=1e-4)

line.track(p, turn_by_turn_monitor='ONE_TURN_EBE')
mon_forward = line.record_last_track

line.track(p, backtrack=True, turn_by_turn_monitor='ONE_TURN_EBE')
mon_backtrack = line.record_last_track

