import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

line = xt.Line.from_json('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
line.particle_ref.gamma0 = 89207.78287659843 # to have a spin tune of 103.45
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]

line['vrfc231'] = 12.65 # qs=0.6

# All off
line['on_sol.2'] = 1
line['on_sol.4'] = 1
line['on_sol.6'] = 1
line['on_sol.8'] = 1
line['on_spin_bump.2'] = 1
line['on_spin_bump.4'] = 1
line['on_spin_bump.6'] = 1
line['on_spin_bump.8'] = 1
line['on_coupl_sol.2'] = 1
line['on_coupl_sol.4'] = 1
line['on_coupl_sol.6'] = 1
line['on_coupl_sol.8'] = 1
line['on_coupl_sol_bump.2'] = 1
line['on_coupl_sol_bump.4'] = 1
line['on_coupl_sol_bump.6'] = 1
line['on_coupl_sol_bump.8'] = 1

tw = line.twiss4d(spin=True, radiation_integrals=True)
line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False

n0 = np.array([tw.spin_x[0], tw.spin_y[0], tw.spin_z[0]])

n0 = n0 / np.linalg.norm(n0)

# Build two orthogonal vectors
tmp = np.array([0, 0, 1]) # Needs generalization
l0 = np.cross(n0, tmp)
l0 = l0 / np.linalg.norm(l0)
m0 = np.cross(l0, n0)
m0 = m0 / np.linalg.norm(m0)

# Build a particle with the three and track them
p0 = line.build_particles(particle_ref=tw.particle_on_co,
                          mode='shift', x=np.zeros(3))
p0.spin_x = np.array([l0[0], n0[0], m0[0]])
p0.spin_y = np.array([l0[1], n0[1], m0[1]])
p0.spin_z = np.array([l0[2], n0[2], m0[2]])

line.track(p0, num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
mon0 = line.record_last_track

# ll = np.zeros((3, len(tw)))
# mm = np.zeros((3, len(tw)))
# nn = np.zeros((3, len(tw)))

# ll[0, :] = mon0.spin_x[0, :]
# ll[1, :] = mon0.spin_y[0, :]
# ll[2, :] = mon0.spin_z[0, :]
# nn[0, :] = mon0.spin_x[1, :]
# nn[1, :] = mon0.spin_y[1, :]
# nn[2, :] = mon0.spin_z[1, :]
# mm[0, :] = mon0.spin_x[2, :]
# mm[1, :] = mon0.spin_y[2, :]
# mm[2, :] = mon0.spin_z[2, :]

out = line.compute_one_turn_matrix_finite_differences(particle_on_co=tw.particle_on_co,
                                                      element_by_element=True)
mon_r_ebe = out['mon_ebe']
part = out['part_temp']

spin = np.zeros((3, len(part.spin_x)))
spin[0, :] = part.spin_x
spin[1, :] = part.spin_y
spin[2, :] = part.spin_z

alpha = np.zeros(len(part.spin_x))
beta = np.zeros(len(part.spin_x))
for ii in range(len(part.spin_x)):
    alpha[ii] = np.dot(spin[:, ii], l0)
    beta[ii] = np.dot(spin[:, ii], m0)


steps_r_matrix = out['steps_r_matrix']

dx = steps_r_matrix["dx"]
dpx = steps_r_matrix["dpx"]
dy = steps_r_matrix["dy"]
dpy = steps_r_matrix["dpy"]
dzeta = steps_r_matrix["dzeta"]
ddelta = steps_r_matrix["ddelta"]

dpzeta = float(part.ptau[6] - part.ptau[12])/2/part.beta0[0]

temp_mat = np.zeros((2, len(part.spin_x)))
temp_mat[0, :] = alpha
temp_mat[1, :] = beta

DD = np.zeros((2, 6))

for jj, dd in enumerate([dx, dpx, dy, dpy, dzeta, dpzeta]):
    DD[:, jj] = (temp_mat[:, jj+1] - temp_mat[:, jj+1+6])/(2*dd)

RR = np.eye(8)
RR[:6, :6] = out['R_matrix']
RR[6:, :6] = DD

def symplectic_matrix(n):

    assert n%2 == 0, "n must be even"
    n = int(n)
    M = np.zeros((n, n))
    for i in range(n//2):
        M[2*i, 2*i+1] = 1
        M[2*i+1, 2*i] = -1

    return M

S = symplectic_matrix(8)