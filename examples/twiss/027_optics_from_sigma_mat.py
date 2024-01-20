# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt

# Load a line and build tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.particle_ref = xt.Particles(
                    mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()

line.vv['vrf400'] = 16
line.vv['kqsx3.l1'] = 3e-6

tw = line.twiss(method='6d')

cov = tw.get_beam_covariance(nemitt_x=2e-6, nemitt_y=1e-6, nemitt_zeta=5e-3)

Sig =  cov.Sigma[0]

S = xt.linear_normal_form.S

eival, eivec = np.linalg.eig(Sig @ S)

nex = eival[0].imag * tw.gamma0 * tw.beta0
ney = eival[1].imag * tw.gamma0 * tw.beta0
nez = eival[2].imag * tw.gamma0 * tw.beta0

print(f'{nex=} {ney=} {nez=}')

dummy_lam = np.diag([
    np.exp(-1j*np.pi/3), np.exp(+1j*np.pi/3),
    np.exp(-1j*np.pi/4), np.exp(+1j*np.pi/4),
    np.exp(-1j*np.pi/5), np.exp(+1j*np.pi/5),
])

dummy_R = eivec @ dummy_lam @ np.linalg.inv(eivec)

dummy_line = xt.Line(elements=[xt.Drift(length=1e-12)])
p_dummy = line.build_particles(x=0)

tw_from_sigmas = dummy_line.twiss(
                        particle_on_co=p_dummy,
                        R_matrix=dummy_R,
                        compute_chromatic_properties=False)

print('betx/bety')
print(f'betx (from line)    = {tw.betx[0]}')
print(f'betx (from sigmas)  = {tw_from_sigmas.betx[0]}')
print(f'bety (from line)    = {tw.bety[0]}')
print(f'bety (from sigmas)  = {tw_from_sigmas.bety[0]}')
print()
print('alfx/alfy')
print(f'alfx (from line)    = {tw.alfx[0]}')
print(f'alfx (from sigmas)  = {tw_from_sigmas.alfx[0]}')
print(f'alfy (from line)    = {tw.alfy[0]}')
print(f'alfy (from sigmas)  = {tw_from_sigmas.alfy[0]}')
print()
print('dx/dy')
print(f'dx (from line)      = {tw.dx[0]}')
print(f'dx (from sigmas)    = {tw_from_sigmas.dx[0]}')
print(f'dy (from line)      = {tw.dy[0]}')
print(f'dy (from sigmas)    = {tw_from_sigmas.dy[0]}')
print()
print('coupled betas')
print(f'betx2 (from line)   = {tw.betx2[0]}')
print(f'betx2 (from sigmas) = {tw_from_sigmas.betx2[0]}')
print(f'bety2 (from line)   = {tw.bety2[0]}')
print(f'bety2 (from sigmas) = {tw_from_sigmas.bety2[0]}')



