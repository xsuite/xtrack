import xtrack as xt
import numpy as np
from rdt_first_order import compute_rdt_first_order_perturbation
from tracking_from_rdt import tracking_from_rdt, frequency_for_rdt

env = xt.load('../../test_data/pimms/PIMM.seq')
line = env.pimms
line.set_particle_ref('proton', kinetic_energy0=100e6)

env['qf1k1'] =  3.15396e-01
env['qd1k1'] = -5.24626e-01
env['qf2k1'] =  5.22717e-01

tw = line.twiss4d()

# # Normal sextupole
# env['xrra'].k2 = 0.8
# rdts = ['f3000', 'f1200', 'f1020', 'f0120', 'f0111']

# Skew sextupole
# env['xrrb'].k2s = 0.8
# rdts = ['f0030', 'f0012', 'f2010', 'f0210', 'f1110']

# Skew quadrupole
env['xrra'].ksl[1] = 2e-3
rdts = ['f1001', 'f1010', 'f0110']


strengths = line.get_table(attr=True)

# Generate 20 particles on the x axis
# particles = line.build_particles(x=3e-3, px=5e-4, y=0, py=0, zeta=0, delta=0)
particles = line.build_particles(x=3e-3, px=5e-4, y=2e-3, py=3e-5, zeta=0, delta=0)

# Inspect the particles
particles.get_table()

# Track 1000 turns logging turn-by-turn data
num_turns = 100_000
line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True,
           with_progress=100)
rec = line.record_last_track

nc  = tw.get_normalized_coordinates(rec)



freq_val_dict = frequency_for_rdt(rdts, tw.qx, tw.qy)

# Mad-ng twiss including RDTs
# tw_ng = line.madng_twiss(rdts=rdts)

# Compute RDTs via first-order perturbation theory
rdt_vals = {}
rdt_vals_ng = {}
for rr in rdts:
    rdt_vals[rr] = compute_rdt_first_order_perturbation(rr, tw, strengths)
    # rdt_vals_ng[rr] = tw_ng[rr]

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

zx_spectrum = np.fft.fft(zx_norm)
zy_spectrum = np.fft.fft(zy_norm)

hx_spectrum = np.fft.fft(hx_minus)
hy_spectrum = np.fft.fft(hy_minus)

freqs = np.fft.fftfreq(num_turns)
freqs[freqs < 0] += 1.0

import nafflib
f_hx, s_hx = nafflib.get_tunes_all(hx_minus, N=100)
f_hy, s_hy = nafflib.get_tunes_all(hy_minus, N=100)
f_x, s_x = nafflib.get_tunes_all(zx_norm, N=100)
f_y, s_y = nafflib.get_tunes_all(zy_norm, N=100)

f_x[f_x < 0] += 1.0
f_hx[f_hx < 0] += 1.0
f_y[f_y < 0] += 1.0
f_hy[f_hy < 0] += 1.0

f_h = {'x': f_hx, 'y': f_hy}
s_h = {'x': s_hx, 'y': s_hy}
f_z = {'x': f_x, 'y': f_y}
s_z = {'x': s_x, 'y': s_y}

# # find strongest line in the resonance region
# qx_resonance = np.mod(2 * tw.qx, 1)
# dq_search = 0.001
# mask_search_x = (np.abs(f_x - qx_resonance) < dq_search)
# i_max_x = np.argmax(np.abs(s_x[mask_search_x]))
# f_x_max = f_x[mask_search_x][i_max_x]
# s_x_max = s_x[mask_search_x][i_max_x]
# mask_search_h = (np.abs(f_hx - qx_resonance) < dq_search)
# i_max_h = np.argmax(np.abs(s_hx[mask_search_h]))
# f_h_max = f_hx[mask_search_h][i_max_h]
# s_h_max = s_hx[mask_search_h][i_max_h]
# print comparison on abs and phase of the strongest line
# print(f'Strongest line near 2qx={qx_resonance:.6f}:')
# print(f' From tracking: tune={f_x_max:.6f}, amp={np.abs(s_x_max):.6e}, phase={np.angle(s_x_max, deg=True):.2f} deg')
# print(f' From RDTs:     tune={f_h_max:.6f},

dq_search = 0.001
print()
for rr in rdts:
    print(f'Spectral lines excited by {rr}:')
    for pp in ['x', 'y']:
        if freq_val_dict[rr + f'_ampl_{pp}_expr'] == '0':
            print(f'  Expected {pp} freq: not excited')
        elif freq_val_dict[rr + f"_freq_{pp}"] == 0:
            print(f'  Expected {pp} freq: zero frequency')
        else:
            print(f'  Expected {pp} freq: {freq_val_dict[rr + f"_freq_{pp}_expr"]} = {freq_val_dict[rr + f"_freq_{pp}"]:.6f}')
            mask_search_z = (np.abs(f_z[pp] - freq_val_dict[rr + f'_freq_{pp}']) < dq_search)
            i_max_z = np.argmax(np.abs(s_z[pp][mask_search_z]))
            f_z_max = f_z[pp][mask_search_z][i_max_z]
            s_z_max = s_z[pp][mask_search_z][i_max_z]
            mask_search_h = (np.abs(f_h[pp] - freq_val_dict[rr + f'_freq_{pp}']) < dq_search)
            i_max_h = np.argmax(np.abs(s_h[pp][mask_search_h]))
            f_h_max = f_h[pp][mask_search_h][i_max_h]
            s_h_max = s_h[pp][mask_search_h][i_max_h]
            print(f'    From tracking: freq={f_z_max:.6f}, amp={np.abs(s_z_max):.6e}, phase={np.angle(s_z_max, deg=True):.2f} deg')
            print(f'    From RDTs:     freq={f_h_max:.6f}, amp={np.abs(s_h_max):.6e}, phase={np.angle(s_h_max, deg=True):.2f} deg')
    print('')


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
plt.figure(20)
plt.suptitle('X plane')
ax_mod = plt.subplot(2,1,1)
plt.plot(freqs, np.abs(zx_spectrum), '.',markersize=4, color='C0')
plt.plot(freqs, np.abs(hx_spectrum), '.', markersize=4, color='C1', alpha=0.5)
plt.yscale('log')
ax_phase = plt.subplot(2,1,2, sharex=ax_mod)
plt.plot(freqs, np.rad2deg(np.angle(zx_spectrum)), '.', markersize=4, color='C0')
plt.plot(freqs, np.rad2deg(np.angle(hx_spectrum)), '.', markersize=4, color='C1', alpha=0.5)
plt.xlabel('Frequency [1/turn]')
plt.ylabel('FFT x Phase [deg]')
plt.subplots_adjust(left=.15)

plt.figure(21)
plt.suptitle('Y plane')
ax_mod = plt.subplot(2,1,1)
plt.plot(freqs, np.abs(zy_spectrum), '.',markersize=4, color='C0')
plt.plot(freqs, np.abs(hy_spectrum), '.', markersize=4, color='C1', alpha=0.5)
plt.yscale('log')
ax_phase = plt.subplot(2,1,2, sharex=ax_mod)
plt.plot(freqs, np.rad2deg(np.angle(zy_spectrum)), '.', markersize=4, color='C0')
plt.plot(freqs, np.rad2deg(np.angle(hy_spectrum)), '.', markersize=4, color='C1', alpha=0.5)
plt.xlabel('Frequency [1/turn]')
plt.ylabel('FFT y Phase [deg]')
plt.subplots_adjust(left=.15)

plt.figure(30)
plt.suptitle('X plane - NAFF')
plt.subplot(2,1,1)
plt.plot(f_x, np.abs(s_x), 'o', markersize=4, color='C0', label='x')
plt.plot(f_hx, np.abs(s_hx), 'x', markersize=4, color='C1', label='from RDTs')
plt.xlabel('Tune')
plt.ylabel('Spectral amplitude')
plt.yscale('log')
plt.legend()
plt.subplot(2,1,2, sharex=plt.gca().axes)
plt.plot(f_x, np.rad2deg(np.angle(s_x)), 'o', markersize=4, color='C0', label='x')
plt.plot(f_hx, np.rad2deg(np.angle(s_hx)), 'x', markersize=4, color='C1', label='from RDTs')
plt.xlabel('Tune')
plt.ylabel('Spectral phase [deg]')

plt.figure(31)
plt.suptitle('Y plane - NAFF')
plt.subplot(2,1,1)
plt.plot(f_y, np.abs(s_y), 'o', markersize=4, color='C0', label='y')
plt.plot(f_hy, np.abs(s_hy), 'x', markersize=4, color='C1', label='from RDTs')
plt.xlabel('Tune')
plt.ylabel('Spectral amplitude')
plt.yscale('log')
plt.legend()
plt.subplot(2,1,2, sharex=plt.gca().axes)
plt.plot(f_y, np.rad2deg(np.angle(s_y)), 'o', markersize=4
            , color='C0', label='y')
plt.plot(f_hy, np.rad2deg(np.angle(s_hy)), 'x', markersize=4
            , color='C1', label='from RDTs')
plt.xlabel('Tune')
plt.ylabel('Spectral phase [deg]')

plt.figure(4)
plt.plot(zx_norm.real, zx_norm.imag, '.', markersize=1, color='C0')
plt.plot(hx_minus.real, hx_minus.imag, '.', markersize=1, color='C1')
plt.xlabel('Re(hx_minus)')
plt.ylabel('Im(hx_minus)')

plt.show()