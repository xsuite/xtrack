import xtrack as xt

env = xt.load('fccee_z_lcc_non_local_boris_solenoid.json')
line = env.fccee_p_ring
tw0 = line.twiss4d()

tt = line.get_table()
tt_quad = tt.rows.match(element_type='Quadrupole')
for nn in tt_quad.name:
    env[nn].k1 = 0

tt_acb = line.vars.get_table().rows['acb.*']
line.set(tt_acb.name, 0)

two = line.twiss(
    betx=tw0['betx', 'ipg'],
    bety=tw0['bety', 'ipg'],
    init_at='ipg')
two.zero_at('ipg')
