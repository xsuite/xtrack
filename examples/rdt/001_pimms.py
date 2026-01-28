import xtrack as xt
import numpy as np
from rdt_first_order import compute_rdt_first_order_perturbation

env = xt.load('../../test_data/pimms/PIMM.seq')
line = env.pimms
line.set_particle_ref('proton', kinetic_energy0=100e6)

env['qf1k1'] =  3.15396e-01
env['qd1k1'] = -5.24626e-01
env['qf2k1'] =  5.22717e-01

env['k2xrr_a'] = 0.8
env['k2yrr_b'] = 0

tw = line.twiss4d()
strengths = line.get_table(attr=True)

# Generate 20 particles on the x axis
x_gen = np.linspace(0, 2.5e-2, 20)
particles = line.build_particles(x=x_gen, px=0, y=0, py=0, zeta=0, delta=0)

# Inspect the particles
particles.get_table()

# Track 1000 turns logging turn-by-turn data
num_turns = 50000
line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=100)
rec = line.record_last_track

nc  = tw.get_normalized_coordinates(rec)

# Mad-ng twiss including RDTs
tw_ng = line.madng_twiss(rdts=['f3000', 'f0300'])

# Compute RDTs via first-order perturbation theory
f3000 = compute_rdt_first_order_perturbation('f3000', tw, strengths)
f0300 = compute_rdt_first_order_perturbation('f0300', tw, strengths)
f1020 = compute_rdt_first_order_perturbation('f1020', tw, strengths)

i_part_analyze = 5
x_norm = nc.x_norm[i_part_analyze, :]
px_norm = nc.px_norm[i_part_analyze, :]

z_norm = x_norm - 1j * px_norm
z_spectrum = np.fft.fft(z_norm)
freqs = np.fft.fftfreq(num_turns)

# Plot turn by turn data
import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(rec.x.T, rec.px.T, '.', markersize=1, color='C0')
plt.ylabel(r'$p_x$')
plt.xlabel(r'$x$ [m]')
plt.xlim(-4e-2, 4e-2)
plt.ylim(-4e-3, 4e-3)
plt.subplots_adjust(left=.15)

# Mark septum position
x_septum = 3.5e-2
plt.axvline(x=_septum, color='k', alpha=0.4, linestyle='--')

# Plot frequency spectrum
plt.figure(2)
plt.plot(freqs, np.abs(z_spectrum), '.', markersize=4, color='C0')
plt.yscale('log')
plt.xlabel('Frequency [1/turn]')
plt.ylabel('FFT Amplitude')
plt.subplots_adjust(left=.15)