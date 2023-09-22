import numpy as np
from scipy.constants import c as clight
from scipy.constants import hbar
from scipy.constants import epsilon_0

from cpymad.madx import Madx
import xtrack as xt
import xpart as xp
import xobjects as xo

mad = Madx()
mad.call('../../test_data/sps_thick/sps.seq')

# mad.input('beam, particle=proton, pc=26;')
# mad.input('beam, particle=electron, pc=20;')

# # realistic
# mad.input('beam, particle=electron, pc=20;')
# v_mv = 25
# num_turns = 8000

# # higher energy
mad.input('beam, particle=electron, pc=50;')
v_mv = 250
num_turns = 500

mad.call('../../test_data/sps_thick/lhc_q20.str')

mad.use(sequence='sps')
mad.input('twiss, table=tw4d;')
twm4d = mad.table.tw4d

mad.sequence.sps.elements['actcse.31632'].volt = v_mv * 10   # To stay in the linear region
mad.sequence.sps.elements['actcse.31632'].freq = 350 / 10  # having the same qs
mad.sequence.sps.elements['actcse.31632'].lag = 0.5

# # Some vertical orbit
mad.sequence.sps.elements['mdv.10107'].kick = 100e-6

mad.input('twiss, table=tw6d;')
twm6d = mad.table.tw6d

mad.sequence.sps.beam.radiate = True
mad.emit()

line = xt.Line.from_madx_sequence(mad.sequence.sps, allow_thick=True)
line.particle_ref = xp.Particles(mass0=xp.ELECTRON_MASS_EV,
                                    q0=-1, gamma0=mad.sequence.sps.beam.gamma)
line.build_tracker()
tw_thick = line.twiss()

Strategy = xt.slicing.Strategy
Teapot = xt.slicing.Teapot

line.discard_tracker()
slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default
    Strategy(slicing=Teapot(2), element_type=xt.Bend),
    Strategy(slicing=Teapot(8), element_type=xt.Quadrupole),
]

line.slice_thick_elements(slicing_strategies)
line.build_tracker()

tw = line.twiss()

line.configure_radiation(model='mean')

# Tapering!!!
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True, method='6d',
                    use_full_inverse=False)

tw_rad2 = line.twiss(eneloss_and_damping=True, method='6d',
                     radiation_method='full',
                     compute_R_element_by_element=True)

assert tw_rad.eq_gemitt_x is not None
assert tw_rad.eq_gemitt_y is not None
assert tw_rad.eq_gemitt_zeta is not None

assert tw_rad2.eq_gemitt_x is None
assert tw_rad2.eq_gemitt_y is None
assert tw_rad2.eq_gemitt_zeta is None

ex = tw_rad.eq_nemitt_x / (tw_rad.gamma0 * tw_rad.beta0)
ey = tw_rad.eq_nemitt_y / (tw_rad.gamma0 * tw_rad.beta0)
ez = tw_rad.eq_nemitt_zeta / (tw_rad.gamma0 * tw_rad.beta0)

EE = tw_rad2.EE
SS = xt.linear_normal_form.S
KK = SS @ EE @ SS.T

d_delta_sq_ave = tw_rad.n_dot_delta_kick_sq_ave * tw_rad.dl_radiation /clight

RR_ebe = tw_rad2.R_matrix_ebe
RR = RR_ebe[0, :, :]
DSigma = np.zeros_like(RR_ebe)
DSigma[:-1, 1, 1] = (d_delta_sq_ave * 0.5 * (tw_rad2.px[:-1]**2 + tw_rad2.px[1:]**2)
                                            / (tw_rad2.delta[:-1] + 1)**2)
DSigma[:-1, 3, 3] = (d_delta_sq_ave * 0.5 * (tw_rad2.py[:-1]**2 + tw_rad2.py[1:]**2)
                                             / (tw_rad2.delta[:-1] + 1)**2)
DSigma[:-1, 5, 5] = d_delta_sq_ave

RR_ebe_inv = np.linalg.inv(RR_ebe)

DSigma0 = np.zeros((6, 6))

n_calc = d_delta_sq_ave.shape[0]
for ii in range(n_calc):
    print(f'{ii}/{n_calc}    ', end='\r', flush=True)
    if d_delta_sq_ave[ii] > 0:
        DSigma0 += RR_ebe_inv[ii, :, :] @ DSigma[ii, :, :] @ RR_ebe_inv[ii, :, :].T

WW = tw_rad2.W_matrix[0, :, :]
lam_eig = tw_rad2.eigenvalues
Rot = tw_rad2.rotation_matrix

lnf = xt.linear_normal_form
CC_split, _, RRR, reig = lnf.compute_linear_normal_form(Rot)
reig_full = np.zeros_like(Rot, dtype=complex)
reig_full[0, 0] = reig[0]
reig_full[1, 1] = reig[0].conjugate()
reig_full[2, 2] = reig[1]
reig_full[3, 3] = reig[1].conjugate()
reig_full[4, 4] = reig[2]
reig_full[5, 5] = reig[2].conjugate()

lam_eig_full = np.zeros_like(reig_full, dtype=complex)
lam_eig_full[0] = lam_eig[0]
lam_eig_full[1] = lam_eig[0].conjugate()
lam_eig_full[2] = lam_eig[1]
lam_eig_full[3] = lam_eig[1].conjugate()
lam_eig_full[4] = lam_eig[2]
lam_eig_full[5] = lam_eig[2].conjugate()

CC = np.zeros_like(CC_split, dtype=complex)
CC[:, 0] = 0.5*np.sqrt(2)*(CC_split[:, 0] + 1j*CC_split[:, 1])
CC[:, 1] = 0.5*np.sqrt(2)*(CC_split[:, 0] - 1j*CC_split[:, 1])
CC[:, 2] = 0.5*np.sqrt(2)*(CC_split[:, 2] + 1j*CC_split[:, 3])
CC[:, 3] = 0.5*np.sqrt(2)*(CC_split[:, 2] - 1j*CC_split[:, 3])
CC[:, 4] = 0.5*np.sqrt(2)*(CC_split[:, 4] + 1j*CC_split[:, 5])
CC[:, 5] = 0.5*np.sqrt(2)*(CC_split[:, 4] - 1j*CC_split[:, 5])

BB = WW @ CC

BB_inv = np.linalg.inv(BB)

EE_norm = BB_inv @ DSigma0 @ BB_inv.T

ex_forest = EE_norm[0, 1]/(1 - np.abs(lam_eig[0])**2)
ey_forest = EE_norm[2, 3]/(1 - np.abs(lam_eig[1])**2)
ez_forest = EE_norm[4, 5]/(1 - np.abs(lam_eig[2])**2)

Sigma_norm = np.zeros_like(EE_norm, dtype=complex)
for ii in range(6):
    for jj in range(6):
        Sigma_norm[ii, jj] = EE_norm[ii, jj]/(1 - lam_eig_full[ii, ii]*lam_eig_full[jj, jj])

Sigma = BB @ Sigma_norm @ BB.T