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

ddelta = 5e-4

# 6d calculation
RR = tw6d.R_matrix
dz_test = 1e-3
xx = np.linalg.solve(RR - np.eye(6), np.array([0,0,0,0,dz_test,0]))
delta_test = xx[5]
slip_factor = dz_test / delta_test / tw6d.circumference

# 4d calculation
RR = tw4d.R_matrix.copy()
solve_mat = RR - np.eye(6)
solve_mat[4, :] = np.array([0,0,0,0,1,0]) # dummy
solve_mat[5, :] = np.array([0,0,0,0,0,1]) # delta

delta_test = 1e-3
xx = np.linalg.solve(solve_mat, np.array([0,0,0,0,0,delta_test]))
# measure slippage on original matrix
xx_out = tw4d.R_matrix @ xx
dz_test = xx_out[4] - xx[4]
slip_factor_4d = dz_test / delta_test / tw4d.circumference


dzeta = slip_factor * ddelta * tw6d.circumference

p_co_plus = line.find_closed_orbit(delta_zeta=dzeta)
p_co_minus = line.find_closed_orbit(delta_zeta=-dzeta)

p_co_plus.zeta += dzeta
p_co_minus.zeta -= dzeta

tw_plus = line.twiss6d(particle_on_co=p_co_plus, compute_chromatic_properties=False)
tw_minus = line.twiss6d(particle_on_co=p_co_minus, compute_chromatic_properties=False)




tw_center = tw6d


delta_plus_mean = np.trapezoid(tw_plus.delta, tw_plus.s) / tw_plus.s[-1]
delta_minus_mean = np.trapezoid(tw_minus.delta, tw_minus.s) / tw_minus.s[-1]
delta_center_mean = np.trapezoid(tw_center.delta, tw_center.s) / tw_center.s[-1]

dqx_plus = (tw_plus.mux[-1] - tw_center.mux[-1]) / (delta_plus_mean - delta_center_mean)
dqx_minus = (tw_center.mux[-1] - tw_minus.mux[-1]) / (delta_center_mean - delta_minus_mean)
dqy_plus = (tw_plus.muy[-1] - tw_center.muy[-1]) / (delta_plus_mean - delta_center_mean)
dqy_minus = (tw_center.muy[-1] - tw_minus.muy[-1]) / (delta_center_mean - delta_minus_mean)

delta_dqxy_plus = 0.5 * (delta_plus_mean + delta_center_mean)
delta_dqxy_minus = 0.5 * (delta_center_mean + delta_minus_mean)
ddqx = (dqx_plus - dqx_minus) / (delta_dqxy_plus - delta_dqxy_minus)
ddqy = (dqy_plus - dqy_minus) / (delta_dqxy_plus - delta_dqxy_minus)

delta_dxdy_plus = 0.5 * (tw_plus.delta + tw_center.delta)
delta_dxdy_minus = 0.5 * (tw_center.delta + tw_minus.delta)

dx_plus = (tw_plus.x - tw_center.x) / (tw_plus.delta - tw_center.delta)
dpx_plus = (tw_plus.px - tw_center.px) / (tw_plus.delta - tw_center.delta)
dy_plus = (tw_plus.y - tw_center.y) / (tw_plus.delta - tw_center.delta)
dpy_plus = (tw_plus.py - tw_center.py) / (tw_plus.delta - tw_center.delta)

dx_minus = (tw_center.x - tw_minus.x) / (tw_center.delta - tw_minus.delta)
dpx_minus = (tw_center.px - tw_minus.px) / (tw_center.delta - tw_minus.delta)
dy_minus = (tw_center.y - tw_minus.y) / (tw_center.delta - tw_minus.delta)
dpy_minus = (tw_center.py - tw_minus.py) / (tw_center.delta - tw_minus.delta)

ddx = (dx_plus - dx_minus) / (delta_dxdy_plus - delta_dxdy_minus)
ddpx = (dpx_plus - dpx_minus) / (delta_dxdy_plus - delta_dxdy_minus)
ddy = (dy_plus - dy_minus) / (delta_dxdy_plus - delta_dxdy_minus)
ddpy = (dpy_plus - dpy_minus) / (delta_dxdy_plus - delta_dxdy_minus)

from cpymad.madx import Madx
mad = Madx()
mad.call('../../test_data/sps_thick/sps.seq')
mad.call('../../test_data/sps_thick/lhc_q20.str')
mad.beam(particle='proton', pc=26e9)
mad.use('sps')
tw = mad.twiss(chrom=True)


# ddqx = (tw_plus.dqx - tw_minus.dqx) / (delta_plus_ave - delta_minus_ave)

# ddx not perfectly closed