import xtrack as xt

env = xt.load('temp_fcc_ee_lcc_non_local_boris_solenoid.json')
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

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(figsize=(12, 6))
plt.plot(two.s, two.betx2, label='betx2')
plt.plot(two.s, two.bety1, label='bety1')
plt.xlim(-20, 20)
plt.ylim(0, 20)
plt.legend()
plt.show()
