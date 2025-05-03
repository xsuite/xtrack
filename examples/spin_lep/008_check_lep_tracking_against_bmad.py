import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

num_turns = 500

line = xt.Line.from_json('lep_sol.json')
line.particle_ref.anomalous_magnetic_moment=0.00115965218128
line.particle_ref.gamma0 = 89207.78287659843 # to have a spin tune of 103.45
spin_tune = line.particle_ref.anomalous_magnetic_moment[0]*line.particle_ref.gamma0[0]

# simplifly the line to facilitate bmad comparison

tt = line.get_table(attr=True)
tt_bend = tt.rows[(tt.element_type == 'RBend') | (tt.element_type == 'Bend')]

for nn in tt_bend.name:
    line[nn].k1 = 0
    line[nn].knl[2] = 0
    line[nn].edge_entry_angle = 0
    line[nn].edge_exit_angle = 0


line['on_sol.2'] = 0
line['on_sol.4'] = 0
line['on_sol.6'] = 1
line['on_sol.8'] = 0
line['on_spin_bump.2'] = 0
line['on_spin_bump.4'] = 0
line['on_spin_bump.6'] = 1
line['on_spin_bump.8'] = 0
line['on_coupl_sol.2'] = 0
line['on_coupl_sol.4'] = 0
line['on_coupl_sol.6'] = 0
line['on_coupl_sol.8'] = 0
line['on_coupl_sol_bump.2'] = 0
line['on_coupl_sol_bump.4'] = 0
line['on_coupl_sol_bump.6'] = 0
line['on_coupl_sol_bump.8'] = 0

tw = line.twiss(betx=1, bety=1,
                x=0.,
                px=0.,
                y=1e-3,
                py=0.,
                delta=1e-3,
                spin_x=0.,
                spin_y=1.,
                spin_z=0.,
                spin=True)

from bmad_track_twiss_spin import bmad_run

bmout = bmad_run(line, track=dict(x=tw.x[0], px=tw.px[0],
                              y=tw.y[0], py=tw.py[0],
                              delta=tw.delta[0],
                             spin_x=tw.spin_x[0],
                             spin_y=tw.spin_y[0],
                             spin_z=tw.spin_z[0]))


import matplotlib.pyplot as plt
plt.close('all')

plt.figure(figsize=(9, 5))
ax1 = plt.subplot(211)
plt.plot(tw.s, tw.x, label='xtrack')
plt.plot(bmout['optics'].s, bmout['optics'].orbit_x_mm*1e-3, label='bmad')
plt.xlabel('s [m]')
plt.ylabel('x [m]')
plt.legend()
plt.subplot(212, sharex=ax1)
plt.plot(tw.s, tw.y, label='xtrack')
plt.plot(bmout['optics'].s, bmout['optics'].orbit_y_mm*1e-3, label='bmad')
plt.xlabel('s [m]')
plt.ylabel('y [m]')
plt.legend()

plt.figure(figsize=(9, 5))
ax1 = plt.subplot(311)
plt.plot(tw.s, tw.spin_x, label='xtrack')
plt.plot(bmout['optics'].s, bmout['spin'].spin_x, label='bmad')
plt.xlabel('s [m]')
plt.ylabel('spin_x')
plt.legend()
plt.subplot(312, sharex=ax1)
plt.plot(tw.s, tw.spin_y, label='xtrack')
plt.plot(bmout['optics'].s, bmout['spin'].spin_y, label='bmad')
plt.xlabel('s [m]')
plt.ylabel('spin_y')
plt.legend()
plt.subplot(313, sharex=ax1)
plt.plot(tw.s, tw.spin_z, label='xtrack')
plt.plot(bmout['optics'].s, bmout['spin'].spin_z, label='bmad')
plt.xlabel('s [m]')
plt.ylabel('spin_z')
plt.legend()

plt.show()


