# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json
import numpy as np

import xobjects as xo
import xtrack as xt
import xpart as xp

fname_line_particles = '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json'
line = xt.Line.from_json(fname_line_particles)
line.particle_ref = xp.Particles(p0c=7e12, mass0=xp.PROTON_MASS_EV)
line.build_tracker()

tw = line.twiss()

R_IP3_IP6 = tw.get_R_matrix(ele_start=0, ele_stop='ip6')
R_IP6_IP3 = tw.get_R_matrix(ele_start='ip6', ele_stop=len(tw.name)-1)

# # Checks
R_prod = R_IP6_IP3 @ R_IP3_IP6

from xtrack.linear_normal_form import compute_linear_normal_form
eig = np.linalg.eig
norm = np.linalg.norm

R_matrix = tw.R_matrix

W_ref, invW_ref, Rot_ref = compute_linear_normal_form(R_matrix)
W_prod, invW_prod, Rot_prod = compute_linear_normal_form(R_prod)


for i_mode in range(3):
    lam_ref = eig(Rot_ref[2*i_mode:2*i_mode+2, 2*i_mode:2*i_mode+2])[0][0]
    lam_prod = eig(Rot_prod[2*i_mode:2*i_mode+2, 2*i_mode:2*i_mode+2])[0][0]

    assert np.isclose(np.abs(np.angle(lam_ref)) / 2 / np.pi,
                      np.abs(np.angle(lam_prod)) / 2 / np.pi,
                      rtol=0, atol=1e-6)

    assert np.isclose(
        norm(W_prod[:, 2*i_mode] - W_ref[:, 2*i_mode], ord=2)/norm(W_ref[:, 2*i_mode], ord=2),
        0, rtol=0, atol=5e-4)
    assert np.isclose(
        norm(W_prod[:4, 2*i_mode] - W_ref[:4, 2*i_mode], ord=2)/norm(W_ref[:4, 2*i_mode], ord=2),
        0, rtol=0, atol=5e-5)

# Check method=4d

tw4d = line.twiss(method='4d', freeze_longitudinal=True)

R_IP3_IP6_4d = tw4d.get_R_matrix(ele_start=0, ele_stop='ip6')
R_IP6_IP3_4d = tw4d.get_R_matrix(ele_start='ip6', ele_stop=len(tw4d.name)-1)

R_prod_4d = R_IP6_IP3_4d @ R_IP3_IP6_4d

# Checks
from xtrack.linear_normal_form import compute_linear_normal_form
eig = np.linalg.eig
norm = np.linalg.norm

R_matrix_4d = tw4d.R_matrix

W_ref_4d, invW_ref_4d, Rot_ref_4d = compute_linear_normal_form(
    R_matrix_4d, only_4d_block=True)
W_prod_4d, invW_prod_4d, Rot_prod_4d = compute_linear_normal_form(
    R_prod_4d, only_4d_block=True)

for i_mode in range(3):
    lam_ref_4d = eig(Rot_ref_4d[2*i_mode:2*i_mode+2, 2*i_mode:2*i_mode+2])[0][0]
    lam_prod_4d = eig(Rot_prod_4d[2*i_mode:2*i_mode+2, 2*i_mode:2*i_mode+2])[0][0]

    assert np.isclose(np.abs(np.angle(lam_ref_4d)) / 2 / np.pi,
                      np.abs(np.angle(lam_prod_4d)) / 2 / np.pi,
                      rtol=0, atol=1e-6)

    assert np.isclose(
        norm(W_prod_4d[:, 2*i_mode] - W_ref_4d[:, 2*i_mode], ord=2)/norm(W_ref_4d[:, 2*i_mode], ord=2),
        0, rtol=0, atol=5e-5)


