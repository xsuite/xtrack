import numpy as np
from scipy import constants

import xobjects as xo
context = xo.ContextCpu(omp_num_threads=0)
import xtrack as xt
import xpart as xp

protonMass = constants.value('proton mass energy equivalent in MeV')*1E6

energy = 7E12
p0c = np.sqrt(energy**2-protonMass**2)
sigma_z = 8E-2
sigma_delta = 1.1E-4
beta_s = sigma_z/sigma_delta

beta = 100.0
dispersion = 1.0

n_part = 1
particles = xp.Particles(_context=context,
                         q0 = 1,
                         mass0 = protonMass,
                         p0c=p0c,
                         x=0.0,
                         px=0.0,
                         y=0.0,
                         py=0.0,
                         zeta=0.0,
                         delta=0.0,
                         )

arc1 = xt.LinearTransferMatrix(
        alpha_x_0=0.0, beta_x_0=beta, disp_x_0=0.0,
        alpha_x_1=0.0, beta_x_1=beta, disp_x_1=dispersion,
        alpha_y_0=0.0, beta_y_0=beta, disp_y_0=0.0,
        alpha_y_1=0.0, beta_y_1=beta, disp_y_1=0.0,
        Q_x=0.0, Q_y=0.0,
        beta_s=beta_s, Q_s=0.0,
        energy_ref_increment=0.0,energy_increment=0.0)

elements = [arc1]
line = xt.Line(elements=elements)
tracker = xt.Tracker(line=line)
mat = tracker.compute_one_turn_matrix_finite_differences(particle_on_co=particles)
print(mat)

det_mat = np.linalg.det(mat)
print('determinant:',det_mat)



S = np.array([[0,1,0,0,0,0],[-1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,-1,0,0,0],[0,0,0,0,0,1],[0,0,0,0,-1,0]])
simplectic_check = np.dot(np.dot(np.transpose(mat),S),mat)-S
print('symplectic check:',np.max(np.abs(simplectic_check)))

