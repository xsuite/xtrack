import xtrack as xt
import numpy as np
from rdt_first_order import compute_rdt_first_order_perturbation
from tracking_from_rdt import tracking_from_rdt

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
num_turns = 100_000
line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=100)
rec = line.record_last_track

nc  = tw.get_normalized_coordinates(rec)

rdts = ['f3000', 'f1200', 'f1020', 'f0120', 'f0111']

# Mad-ng twiss including RDTs
tw_ng = line.madng_twiss(rdts=rdts)

# Compute RDTs via first-order perturbation theory
# f3000 = compute_rdt_first_order_perturbation('f3000', tw, strengths)
# f0300 = compute_rdt_first_order_perturbation('f0300', tw, strengths)
# f1020 = compute_rdt_first_order_perturbation('f1020', tw, strengths)
rdt_vals = {}
rdt_vals_ng = {}
for rr in rdts:
    rdt_vals[rr] = compute_rdt_first_order_perturbation(rr, tw, strengths)
    rdt_vals_ng[rr] = tw_ng[rr]

i_part_analyze = 5
x_norm = nc.x_norm[i_part_analyze, :]
px_norm = nc.px_norm[i_part_analyze, :]
y_norm = nc.y_norm[i_part_analyze, :]
py_norm = nc.py_norm[i_part_analyze, :]

# tracking from RDTs
Ix = 0.5 * (x_norm**2 + px_norm**2)
Iy = 0.5 * (y_norm**2 + py_norm**2)

hx_minus, hy_minus = tracking_from_rdt(
    rdts={rr: (rdt_vals[rr][0]) for rr in rdts},
    Ix=Ix,
    Iy=Iy,
    Qx=tw.qx,
    Qy=tw.qy,
    psi_x0=0.0,
    psi_y0=0.0,
    num_turns=num_turns
)

z_norm = x_norm - 1j * px_norm

z_spectrum = np.fft.fft(z_norm)
h_spectrum = np.fft.fft(hx_minus)
freqs = np.fft.fftfreq(num_turns)

import nafflib
f_h, s_h = nafflib.get_tunes_all(hx_minus, N=100)
f_x, s_x = nafflib.get_tunes_all(z_norm, N=100)

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

# Plot frequency spectrum
plt.figure(2)
ax_mod = plt.subplot(2,1,1)
plt.plot(freqs, np.abs(z_spectrum), '.',markersize=4, color='C0')
plt.plot(freqs, np.abs(h_spectrum), '.', markersize=4, color='C1')
plt.yscale('log')
ax_phase = plt.subplot(2,1,2, sharex=ax_mod)
plt.plot(freqs, np.rad2deg(np.angle(z_spectrum)), '.', markersize=4, color='C0')
plt.plot(freqs, np.rad2deg(np.angle(h_spectrum)), '.', markersize=4, color='C1')
plt.xlabel('Frequency [1/turn]')
plt.ylabel('FFT Phase [deg]')
plt.subplots_adjust(left=.15)

plt.figure(3)
plt.subplot(2,1,1)
plt.plot(f_x, np.abs(s_x), 'o', markersize=4, color='C0', label='x')
plt.plot(f_h, np.abs(s_h), 'x', markersize=4, color='C1', label='from RDTs')
plt.xlabel('Tune')
plt.ylabel('Spectral amplitude')
plt.yscale('log')
plt.legend()
plt.subplot(2,1,2, sharex=plt.gca().axes)
plt.plot(f_x, np.rad2deg(np.angle(s_x)), 'o', markersize=4
            , color='C0', label='x')
plt.plot(f_h, np.rad2deg(np.angle(s_h)), 'x', markersize=4
            , color='C1', label='from RDTs')
plt.xlabel('Tune')
plt.ylabel('Spectral phase [deg]')

plt.show()