import xtrack as xt

d_zeta = 1e-3

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_collider/collider_00_from_mad.json'
)
collider.build_trackers()

collider.vars['vrf400'] = 16
collider.vars['on_crab1'] = -190
collider.vars['on_crab5'] = -190

line = collider.lhcb1
line.cycle('ip2', inplace=True)

tw6d_rf_on = line.twiss()
tw4d_rf_on = line.twiss(method='4d')

p = line.build_particles(num_particles=2)
dzeta = 1e-3

p.zeta[0] = -dzeta
p.zeta[1] = dzeta

with xt.line._preserve_config(line):
    line.freeze_energy()
    line.track(p, turn_by_turn_monitor='ONE_TURN_EBE')
mon = line.record_last_track

dx_dzeta =(mon.x[1, :] - mon.x[0, :])/ (mon.zeta[1, :] - mon.zeta[0, :])