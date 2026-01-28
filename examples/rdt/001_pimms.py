import xtrack as xt
import numpy as np

env = xt.load('../../test_data/pimms/PIMM.seq')
line = env.pimms
line.set_particle_ref('proton', kinetic_energy0=100e6)

env['qf1k1'] =  3.15396e-01
env['qd1k1'] = -5.24626e-01
env['qf2k1'] =  5.22717e-01

env['k2xrr_a'] = 1.
env['k2yrr_b'] = -6.5

# Generate 20 particles on the x axis
x_gen = np.linspace(0, 2.5e-2, 20)
particles = line.build_particles(x=x_gen, px=0, y=0, py=0, zeta=0, delta=0)

# Inspect the particles
particles.get_table()

# Track 1000 turns logging turn-by-turn data
num_turns = 1000
line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True)
rec = line.record_last_track

# Plot turn by turn data
plt.figure()
plt.plot(rec.x.T, rec.px.T, '.', markersize=1, color='C0')
plt.ylabel(r'$p_x$')
plt.xlabel(r'$x$ [m]')
plt.xlim(-4e-2, 4e-2)
plt.ylim(-4e-3, 4e-3)
plt.subplots_adjust(left=.15)

# Mark septum position
x_septum = 3.5e-2
plt.axvline(x=x_septum, color='k', alpha=0.4, linestyle='--')