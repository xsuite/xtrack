import xtrack as xt
import numpy as np

env = xt.load(['../../test_data/sps_thick/sps.seq',
               '../../test_data/sps_thick/lhc_q20.str'])
line = env.sps
line.set_particle_ref('proton', p0c=26e9)

line['actcse.31632'].voltage = 4.5e6
line['actcse.31632'].frequency = 200e6
line['actcse.31632'].lag = 180

tw6d = line.twiss6d()
tw4d = line.twiss4d()

ddelta = 1e-5

dzeta = tw6d.slip_factor * ddelta * tw6d.circumference

p_co_plus = line.find_closed_orbit(delta_zeta=dzeta)
p_co_minus = line.find_closed_orbit(delta_zeta=-dzeta)

p_co_plus.zeta += dzeta
p_co_minus.zeta -= dzeta

tw_plus = line.twiss6d(particle_on_co=p_co_plus)
tw_minus = line.twiss6d(particle_on_co=p_co_minus)

delta_plus_ave = np.trapezoid(tw_plus.delta, tw_plus.s) / tw_plus.s[-1]
delta_minus_ave = np.trapezoid(tw_minus.delta, tw_minus.s) / tw_minus.s[-1]

ddx = (tw_plus.dx - tw_minus.dx) / (tw_plus.delta - tw_minus.delta)

ddqx = (tw_plus.dqx - tw_minus.dqx) / (delta_plus_ave - delta_minus_ave)

# ddx not perfectly closed