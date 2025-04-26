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

# Based on:
# A. Chao, valuation of Radiative Spin Polarization in an Electron Storage Ring
# https://inspirehep.net/literature/154360

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

steps_r_matrix = tw.steps_r_matrix

out = line.compute_one_turn_matrix_finite_differences(particle_on_co=tw.particle_on_co,
                                                      element_by_element=True,
                                                      steps_r_matrix=steps_r_matrix)
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

eival, eivec = np.linalg.eig(RR)

##### Sort modes in pairs of conjugate modes #####
w0 = eival
v0 = eivec
index_list = [0,7,5,1,2,6,3,4] # we mix them up to check the algorithm
conj_modes = np.zeros([4,2], dtype=np.int64)
for j in [0,1,2]:
    conj_modes[j,0] = index_list[0]
    del index_list[0]

    min_index = 0
    min_diff = abs(np.imag(w0[conj_modes[j,0]] + w0[index_list[min_index]]))
    for i in range(1,len(index_list)):
        diff = abs(np.imag(w0[conj_modes[j,0]] + w0[index_list[i]]))
        if min_diff > diff:
            min_diff = diff
            min_index = i

    conj_modes[j,1] = index_list[min_index]
    del index_list[min_index]

conj_modes[3,0] = index_list[0]
conj_modes[3,1] = index_list[1]

modes = np.empty(4, dtype=np.int64)
modes[0] = conj_modes[0, 0]
modes[1] = conj_modes[1, 0]
modes[2] = conj_modes[2, 0]
modes[3] = conj_modes[3, 0]

# Sort modes such that (1,2,3) is close to (x,y,zeta,spin)
# Identify the spin mode
for i in [0,1,2]:
    if abs(v0[:,modes[3]])[6] < abs(v0[:,modes[i]])[6]:
        modes[3], modes[i] = modes[i], modes[3]
# Identify the longitudinal mode
for i in [0,1]:
    if abs(v0[:,modes[2]])[5] < abs(v0[:,modes[i]])[5]:
        modes[2], modes[i] = modes[i], modes[2]
# Identify the vertical mode
if abs(v0[:,modes[1]])[2] < abs(v0[:,modes[0]])[2]:
    modes[0], modes[1] = modes[1], modes[0]

a1 = v0[:6, modes[0]].real
a2 = v0[:6, modes[1]].real
a3 = v0[:6, modes[2]].real
b1 = v0[:6, modes[0]].imag
b2 = v0[:6, modes[1]].imag
b3 = v0[:6, modes[2]].imag

S = np.array([[0., 1., 0., 0., 0., 0.],
              [-1., 0., 0., 0., 0., 0.],
              [ 0., 0., 0., 1., 0., 0.],
              [ 0., 0.,-1., 0., 0., 0.],
              [ 0., 0., 0., 0., 0., 1.],
              [ 0., 0., 0., 0.,-1., 0.]])

n1_inv_sq = np.abs(np.matmul(np.matmul(a1, S), b1))
n2_inv_sq = np.abs(np.matmul(np.matmul(a2, S), b2))
n3_inv_sq = np.abs(np.matmul(np.matmul(a3, S), b3))

n1 = 1./np.sqrt(n1_inv_sq)
n2 = 1./np.sqrt(n2_inv_sq)
n3 = 1./np.sqrt(n3_inv_sq)

e1 = eivec[:, modes[0]] / n1
e2 = eivec[:, modes[1]] / n2
e3 = eivec[:, modes[2]] / n3

scale_e1 = np.max([np.abs(e1[0])/dx, np.abs(e1[1])/dpx])
e1_scaled = e1 / scale_e1
scale_e2 = np.max([np.abs(e2[2])/dy, np.abs(e2[3])/dpy])
e2_scaled = e2 / scale_e2
scale_e3 = np.max([np.abs(e3[4])/dzeta, np.abs(e3[5])/dpzeta])
e3_scaled = e3 / scale_e3

e1_trk_re = e1_scaled.real
e1_trk_im = e1_scaled.imag
e2_trk_re = e2_scaled.real
e2_trk_im = e2_scaled.imag
e3_trk_re = e3_scaled.real
e3_trk_im = e3_scaled.imag

e1_spin_re = e1_trk_re[6] * l0 + e1_trk_re[7] * m0
e1_spin_im = e1_trk_im[6] * l0 + e1_trk_im[7] * m0
e2_spin_re = e2_trk_re[6] * l0 + e2_trk_re[7] * m0
e2_spin_im = e2_trk_im[6] * l0 + e2_trk_im[7] * m0
e3_spin_re = e3_trk_re[6] * l0 + e3_trk_re[7] * m0
e3_spin_im = e3_trk_im[6] * l0 + e3_trk_im[7] * m0

x = tw.x[0] + np.array([
    e1_trk_re[0], e1_trk_im[0],
    e2_trk_re[0], e2_trk_im[0],
    e3_trk_re[0], e3_trk_im[0],
])
px = tw.px[0] + np.array([
    e1_trk_re[1], e1_trk_im[1],
    e2_trk_re[1], e2_trk_im[1],
    e3_trk_re[1], e3_trk_im[1],
])
y = tw.y[0] + np.array([
    e1_trk_re[2], e1_trk_im[2],
    e2_trk_re[2], e2_trk_im[2],
    e3_trk_re[2], e3_trk_im[2],
])
py = tw.py[0] + np.array([
    e1_trk_re[3], e1_trk_im[3],
    e2_trk_re[3], e2_trk_im[3],
    e3_trk_re[3], e3_trk_im[3],
])
zeta = tw.zeta[0] + np.array([
    e1_trk_re[4], e1_trk_im[4],
    e2_trk_re[4], e2_trk_im[4],
    e3_trk_re[4], e3_trk_im[4],
])
ptau = tw.ptau[0] + 1/tw.beta0 * np.array([ # in the eigenvector there is pzeta
    e1_trk_re[5], e1_trk_im[5],
    e2_trk_re[5], e2_trk_im[5],
    e3_trk_re[5], e3_trk_im[5],
])
spin_x = np.array([
    e1_spin_re[0], e1_spin_im[0],
    e2_spin_re[0], e2_spin_im[0],
    e3_spin_re[0], e3_spin_im[0],
])
spin_y = np.array([
    e1_spin_re[1], e1_spin_im[1],
    e2_spin_re[1], e2_spin_im[1],
    e3_spin_re[1], e3_spin_im[1],
])
spin_z = np.array([
    e1_spin_re[2], e1_spin_im[2],
    e2_spin_re[2], e2_spin_im[2],
    e3_spin_re[2], e3_spin_im[2],
])

par_track = xp.build_particles(
    particle_ref=tw.particle_on_co, mode='set',
    x=x, px=px, y=y, py=py, zeta=zeta, ptau=ptau,
    spin_x=spin_x, spin_y=spin_y, spin_z=spin_z,
)