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
particles = line.build_particles(x=3e-3, px=5e-4, y=0, py=0, zeta=0, delta=0)
# particles = line.build_particles(x=3e-3, px=5e-4, y=2e-3, py=3e-5, zeta=0, delta=0)

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

i_part_analyze = 0
x_norm = nc.x_norm[i_part_analyze, :]
px_norm = nc.px_norm[i_part_analyze, :]
y_norm = nc.y_norm[i_part_analyze, :]
py_norm = nc.py_norm[i_part_analyze, :]

zx_norm = x_norm - 1j * px_norm
zy_norm = y_norm - 1j * py_norm
# tracking from RDTs
Ix = 0.5 * (zx_norm[0].real**2 + zx_norm[0].imag**2)
Iy = 0.5 * (zy_norm[0].real**2 + zy_norm[0].imag**2)
psi_x0 = np.angle(zx_norm[0].real + 1j * zx_norm[0].imag)
psi_y0 = np.angle(zy_norm[0].real + 1j * zy_norm[0].imag)

rdt_use = rdt_vals

def initial_conditions(Ix, Iy, psi_x0, psi_y0):

    hx_minus, hy_minus = tracking_from_rdt(
        rdts={rr: (rdt_use[rr][0]) for rr in rdts},
        Ix=Ix,
        Iy=Iy,
        Qx=tw.qx,
        Qy=tw.qy,
        psi_x0=psi_x0,
        psi_y0=psi_y0,
        num_turns=1
    )

    return np.array([hx_minus[0].real,
                     hx_minus[0].imag,
                     hy_minus[0].real,
                     hy_minus[0].imag])

opt = xt.match.opt_from_callable(
        lambda xx: initial_conditions(xx[0], xx[1], xx[2], xx[3]),
        x0=np.array([Ix, Iy, psi_x0, psi_y0]),
        steps=np.array([Ix*1e-4, Ix*1e-4, 1e-4, 1e-4]),
        tar=np.array([zx_norm[0].real, zx_norm[0].imag,
                         zy_norm[0].real, zy_norm[0].imag]),
        tols=[1e-10, 1e-10, 1e-10, 1e-10],
    )
opt.step()
res = opt.get_knob_values()
Ix = res[0]
Iy = res[1]
psi_x0 = res[2]
psi_y0 = res[3]


hx_minus, hy_minus = tracking_from_rdt(
    rdts={rr: (rdt_use[rr][0]) for rr in rdts},
    Ix=Ix,
    Iy=Iy,
    Qx=tw.qx,
    Qy=tw.qy,
    psi_x0=psi_x0,
    psi_y0=psi_y0,
    num_turns=num_turns
)

z_spectrum = np.fft.fft(zx_norm)
h_spectrum = np.fft.fft(hx_minus)
freqs = np.fft.fftfreq(num_turns)

import nafflib
f_h, s_h = nafflib.get_tunes_all(hx_minus, N=100)
f_x, s_x = nafflib.get_tunes_all(zx_norm, N=100)

# find sronges line in the resonsne region
qx_resonance = np.mod(2 * tw.qx, 1)
dq_search = 0.001
mask_search_x = (np.abs(f_x - qx_resonance) < dq_search)
i_max_x = np.argmax(np.abs(s_x[mask_search_x]))
f_x_max = f_x[mask_search_x][i_max_x]
s_x_max = s_x[mask_search_x][i_max_x]
mask_search_h = (np.abs(f_h - qx_resonance) < dq_search)
i_max_h = np.argmax(np.abs(s_h[mask_search_h]))
f_h_max = f_h[mask_search_h][i_max_h]
s_h_max = s_h[mask_search_h][i_max_h]

# print comparison on abs and phase of the strongest line
print(f'Strongest line near 2qx={qx_resonance:.6f}:')
print(f' From tracking: tune={f_x_max:.6f}, amp={np.abs(s_x_max):.6e}, phase={np.angle(s_x_max, deg=True):.2f} deg')
print(f' From RDTs:     tune={f_h_max:.6f}, amp={np.abs(s_h_max):.6e}, phase={np.angle(s_h_max, deg=True):.2f} deg')

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

plt.figure(4)
plt.plot(zx_norm.real, zx_norm.imag, '.', markersize=1, color='C0')
plt.plot(hx_minus.real, hx_minus.imag, '.', markersize=1, color='C1')
plt.xlabel('Re(hx_minus)')
plt.ylabel('Im(hx_minus)')

plt.show()