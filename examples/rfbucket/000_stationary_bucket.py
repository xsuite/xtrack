import xtrack as xt
import xpart as xp
import numpy as np
import xobjects as xo

np.random.seed(12345)

shift_zeta = -0.4

sigma_z = 12e-2 # triggers non-linear matching
# sigma_z = 1.2e-2 # triggers linear matching


env = xt.load(['../../test_data/sps_thick/sps.seq',
                '../../test_data/sps_thick/lhc_q20.str'])
line = env.sps
line.set_particle_ref('proton', p0c=26e9)

line.append('timedelay', xt.TimeDelay(shift_zeta=shift_zeta))

env['actcse.31632'].frequency = 200e6
env['actcse.31632'].voltage = 4.5e6
env['actcse.31632'].phase = np.pi

tw = line.twiss6d()

p = xp.generate_matched_gaussian_bunch(
    line=line,
    num_particles=100_000,
    nemitt_x=2e-6,
    nemitt_y=2e-6,
    sigma_z=sigma_z)

rfb = line._get_bucket()

z_separatrix = np.linspace(rfb.z_left, rfb.z_right, 1000)
delta_separatrix = rfb.separatrix(z_separatrix)
delta_separatrix_neg = rfb.separatrix(z_separatrix, sgn=-1)

xo.assert_allclose(rfb.dp0, tw.delta[0], rtol=0, atol=1e-12)
xo.assert_allclose(rfb.z_sfp, tw.zeta[0], rtol=1e-3, atol=1e-9)

# For this stationary single-RF bucket, the bucket is centered at zeta0.
xo.assert_allclose(0.5 * (rfb.z_left + rfb.z_right), tw.zeta[0],
                   rtol=1e-3, atol=1e-12)

# The stable fixed point Hamiltonian must be evaluated at the shifted
# momentum coordinate dp0.
xo.assert_allclose(
    rfb.h_sfp(),
    rfb.hamiltonian(rfb.z_sfp, rfb.dp0),
    rtol=0,
    atol=1e-12,
)

# The upper and lower separatrix branches are symmetric around dp0.
xo.assert_allclose(
    (delta_separatrix + delta_separatrix_neg) / 2,
    rfb.dp0,
    rtol=0,
    atol=1e-14,
)
xo.assert_allclose(delta_separatrix[[0, -1]], rfb.dp0,
                   rtol=0, atol=1e-12)
xo.assert_allclose(delta_separatrix_neg[[0, -1]], rfb.dp0,
                   rtol=0, atol=1e-12)

assert rfb.is_in_separatrix(tw.zeta[0], tw.delta[0])
assert np.all(rfb.is_in_separatrix(np.array(p.zeta), np.array(p.delta)))

xo.assert_allclose(np.mean(p.zeta), tw.zeta[0], rtol=0, atol=2e-3)
xo.assert_allclose(np.std(p.zeta), sigma_z, rtol=2e-2, atol=0)
xo.assert_allclose(np.mean(p.delta), tw.delta[0], rtol=0, atol=2e-5)

z_probe =  np.array([-0.25, 0, 0.25]) * (rfb.z_right - rfb.z_left)
delta_probe_sep = rfb.separatrix(z_probe)

# Launch three probe particles just inside the separatrix and three just
# outside it, at the same zeta coordinates.
delta_probe_inside = rfb.dp0 + 0.9 * (delta_probe_sep - rfb.dp0)
delta_probe_outside = rfb.dp0 + 1.1 * (delta_probe_sep - rfb.dp0)
z_probe_all = np.r_[z_probe, z_probe]
delta_probe_all = np.r_[delta_probe_inside, delta_probe_outside]

probe = line.build_particles(
    zeta=z_probe_all,
    delta=delta_probe_all,
    x = tw.dx[0] * delta_probe_all,
    px = tw.dpx[0] * delta_probe_all,
)

# assert np.all(rfb.is_in_separatrix(probe.zeta[:3], probe.delta[:3]))
# assert not np.any(rfb.is_in_separatrix(probe.zeta[3:], probe.delta[3:]))

line.track(probe, num_turns=200, turn_by_turn_monitor=True)
mon = line.record_last_track

inside_is_accepted = rfb.is_in_separatrix(
    np.array(mon.zeta[:3, :]),
    np.array(mon.delta[:3, :]),
)
outside_is_accepted = rfb.is_in_separatrix(
    np.array(mon.zeta[3:, :]),
    np.array(mon.delta[3:, :]),
)

# The particles launched inside stay in the bucket, while the particles
# launched outside drift away instead of being captured.
assert np.all(inside_is_accepted)
assert not np.any(outside_is_accepted)
assert np.all(np.ptp(np.array(mon.zeta[:3, :]), axis=1)
              < (rfb.z_right - rfb.z_left))
assert np.all(np.ptp(np.array(mon.zeta[3:, :]), axis=1)
              > 3 * (rfb.z_right - rfb.z_left))

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(p.zeta, p.delta, '.', markersize=0.5, alpha=0.5)
plt.plot(z_separatrix, delta_separatrix)
plt.plot(z_separatrix, delta_separatrix_neg)
plt.plot(tw.zeta[0], tw.delta[0], 'x')

plt.figure(2)
plt.plot(z_separatrix, delta_separatrix, color='k', linewidth=1.5)
plt.plot(z_separatrix, delta_separatrix_neg, color='k', linewidth=1.5)
for ii in range(3):
    mask_inside = mon.state[ii, :] >= 1
    mask_outside = mon.state[ii + 3, :] >= 1
    plt.plot(mon.zeta[ii, mask_inside], mon.delta[ii, mask_inside],
             color='C0', linewidth=1)
    plt.plot(mon.zeta[ii + 3, mask_outside],
             mon.delta[ii + 3, mask_outside], color='C3',
             linewidth=1)
plt.plot(z_probe_all[:3], delta_probe_all[:3], 'o', color='C0')
plt.plot(z_probe_all[3:], delta_probe_all[3:], 'o', color='C3')
plt.plot(tw.zeta[0], tw.delta[0], 'x', color='k')
z_margin = 0.15 * (rfb.z_right - rfb.z_left)
delta_sep_min = min(np.min(delta_separatrix), np.min(delta_separatrix_neg))
delta_sep_max = max(np.max(delta_separatrix), np.max(delta_separatrix_neg))
delta_margin = 0.15 * (delta_sep_max - delta_sep_min)
plt.xlim(rfb.z_left - z_margin, rfb.z_right + z_margin)
plt.ylim(delta_sep_min - delta_margin, delta_sep_max + delta_margin)
plt.xlabel('zeta [m]')
plt.ylabel('delta')
plt.title('Probe particle trajectories')
plt.show()
