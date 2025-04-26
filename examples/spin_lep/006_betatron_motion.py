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
line['on_sol.2'] = 0
line['on_sol.4'] = 0
line['on_sol.6'] = 0
line['on_sol.8'] = 0
line['on_spin_bump.2'] = 0
line['on_spin_bump.4'] = 0
line['on_spin_bump.6'] = 0
line['on_spin_bump.8'] = 0
line['on_coupl_sol.2'] = 0
line['on_coupl_sol.4'] = 0
line['on_coupl_sol.6'] = 0
line['on_coupl_sol.8'] = 0
line['on_coupl_sol_bump.2'] = 0
line['on_coupl_sol_bump.4'] = 0
line['on_coupl_sol_bump.6'] = 0
line['on_coupl_sol_bump.8'] = 0

tw = line.twiss(spin=True, radiation_integrals=True)
line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False # For spin

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

ll = np.zeros((3, len(tw)))
mm = np.zeros((3, len(tw)))
nn = np.zeros((3, len(tw)))

ll[0, :] = mon0.spin_x[0, :]
ll[1, :] = mon0.spin_y[0, :]
ll[2, :] = mon0.spin_z[0, :]
nn[0, :] = mon0.spin_x[1, :]
nn[1, :] = mon0.spin_y[1, :]
nn[2, :] = mon0.spin_z[1, :]
mm[0, :] = mon0.spin_x[2, :]
mm[1, :] = mon0.spin_y[2, :]
mm[2, :] = mon0.spin_z[2, :]

two = line.twiss(   spin=True,
                    betx = tw.betx[0],
                    bety = tw.bety[0],
                    alfx = tw.alfx[0],
                    alfy = tw.alfy[0],
                    dx = tw.dx[0],
                    dpx = tw.dpx[0],
                    dy = tw.dy[0],
                    dpy = tw.dpy[0],
                    x = tw.x[0],
                    px = tw.px[0],
                    y = tw.y[0] + 1e-3,
                    py = tw.py[0],
                    delta = tw.delta[0],
                    zeta = tw.zeta[0],
                    spin_x = tw.spin_x[0],
                    spin_y = tw.spin_y[0],
                    spin_z = tw.spin_z[0])

spin = np.zeros((3, len(tw)))
spin[0, :] = two.spin_x
spin[1, :] = two.spin_y
spin[2, :] = two.spin_z

alpha = np.sum(spin * ll, axis=0)
beta = np.sum(spin * mm, axis=0)

