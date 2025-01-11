import numpy as np
from cpymad.madx import Madx
import xtrack as xt

from scipy.constants import c as clight

delta0 = 0 #-1e-2
delta_range = 0
num_turns=100
num_particles = 100_000

# delta0 = 1e-2
# delta_range = 0
# num_turns=20
# num_particles = 5_000_000

# # To see the different number of turns
# delta0 = 0e-2
# delta_range = 10e-3
# num_turns=5000
# num_particles = 5000

line = xt.Line.from_json(
    '../../test_data/psb_injection/line_and_particle.json')

# RF off!
tt = line.get_table()
ttcav = tt.rows[tt.element_type == 'Cavity']
for nn in ttcav.name:
    line.element_refs[nn].voltage=0

line.configure_bend_model(core='bend-kick-bend', edge='full')
line.twiss_default['method'] = '4d'

tw = line.twiss()
twom = line.twiss(delta0=delta0)
line.discard_tracker()

# Install dummy collective elements
s_sync = np.linspace(0, tw.circumference, 10)
for ii, ss in enumerate(s_sync):
    nn = f'sync_here_{ii}'
    line.insert(nn, obj=xt.Marker(), at=ss)
    line[nn].iscollective = True

import xtrack.synctime as st
st.install_sync_time_at_collective_elements(line)

import xobjects as xo
line.build_tracker(_context=xo.ContextCpu(omp_num_threads='auto'))

beta1 = tw.beta0 / 0.9

circumference = tw.circumference
zeta_min0 = -circumference/2*tw.beta0/beta1
zeta_max0 = circumference/2*tw.beta0/beta1


p = line.build_particles(
    delta=delta0 + delta_range * np.random.uniform(-1, 1, num_particles),
    x_norm=0, y_norm=0
)

# Need to take beta of actual particles to convert the distribution along the
# circumference to a distribution in time
p.zeta = (np.random.uniform(0, circumference, num_particles) / p.rvv * 0.999
          + (zeta_max0 - circumference) / p.rvv)

st.prepare_particles_for_sync_time(p, line)

p.y[(p.zeta > 1) & (p.zeta < 2)] = 1e-3  # kick
p.weight[(p.zeta > 5) & (p.zeta < 10)] *= 1.3

initial_histogram, z_init_hist = np.histogram(p.zeta, bins=200,
                                  range=(zeta_max0 - circumference, zeta_max0),
                                  weights=p.weight)

p0 = p.copy()

def particles(_, p):
    return p.copy()

def intensity(line, particles):
    return (np.sum(particles.weight[particles.state > 0])
                  / ((zeta_max0 - zeta_min0)/tw.beta0/clight))

def z_range(line, particles):
    mask_alive = particles.state > 0
    return particles.zeta[mask_alive].min(), particles.zeta[mask_alive].max()

def long_density(line, particles):
    mask_alive = particles.state > 0
    if not(np.any(particles.at_turn[mask_alive] == 0)): # don't check at the first turn
        assert np.all(particles.zeta[mask_alive] > zeta_min0)
        assert np.all(particles.zeta[mask_alive] < zeta_max0)
    return np.histogram(particles.zeta[mask_alive], bins=200,
                        range=(zeta_min0, zeta_max0),
                        weights=particles.weight[mask_alive])

def y_mean_hist(line, particles):

    mask_alive = particles.state > 0
    if not(np.any(particles.at_turn[mask_alive] == 0)): # don't check at the first turn
        assert np.all(particles.zeta[mask_alive] > zeta_min0)
        assert np.all(particles.zeta[mask_alive] < zeta_max0)
    return np.histogram(particles.zeta[mask_alive], bins=200,
                        range=(zeta_min0, zeta_max0), weights=particles.y[mask_alive])


line.enable_time_dependent_vars = True

line.track(p, num_turns=num_turns, log=xt.Log(intensity=intensity,
                                         long_density=long_density,
                                         y_mean_hist=y_mean_hist,
                                         z_range=z_range,
                                         particles=particles
                                         ), with_progress=2)

inten = line.log_last_track['intensity']

f_rev_ave = 1 / tw.T_rev0 * (1 - tw.slip_factor * p.delta.mean())
t_rev_ave = 1 / f_rev_ave

inten_exp =  np.sum(p0.weight) / t_rev_ave

z_axis = line.log_last_track['long_density'][0][1]
hist_mat = np.array([rr[0] for rr in line.log_last_track['long_density']])
hist_y = np.array([rr[0] for rr in line.log_last_track['y_mean_hist']])

dz = z_axis[1] - z_axis[0]
y_vs_t = np.fliplr(hist_y).flatten() # need to flip because of the minus in z = -beta0 c t
intensity_vs_t = np.fliplr(hist_mat).flatten()
z_unwrapped = np.arange(0, len(y_vs_t)) * dz
t_unwrapped = z_unwrapped / (tw.beta0 * clight)

z_range_size = z_axis[-1] - z_axis[0]
t_range_size = z_range_size / (tw.beta0 * clight)

import nafflib
f_harmons = nafflib.get_tunes(intensity_vs_t, N=50)[0] / (t_unwrapped[1] - t_unwrapped[0])
f_nominal = 1 / tw.T_rev0
dt_expected = -(twom.zeta[-1] - twom.zeta[0]) / tw.beta0 / clight
f_expected = 1 / (tw.T_rev0 + dt_expected)

f_measured = f_harmons[np.argmin(np.abs(f_harmons - f_nominal))]

print('f_nominal:  ', f_nominal, ' Hz')
print('f_expected: ', f_expected, ' Hz')
print('f_measured: ', f_measured, ' Hz')
print('Error:      ', f_measured - f_expected, 'Hz')

assert np.isclose(f_expected, f_measured, rtol=0, atol=2) # 2 Hz tolerance
assert np.isclose(np.mean(inten), inten_exp, rtol=1e-2, atol=0)
assert np.allclose(p.at_turn, num_turns*0.9, rtol=3e-2, atol=0) #beta1 defaults to 0.1

tt = line.get_table()
tt_synch = tt.rows[tt.element_type=='SyncTime']
assert len(tt_synch) == 12
assert tt_synch.name[0] == 'synctime_start'
assert tt_synch.name[-1] == 'synctime_end'
assert np.all(tt_synch.name[5] == 'synctime_4')
assert line['synctime_start'].at_start
assert not line['synctime_end'].at_start
assert not line['synctime_4'].at_start
assert line['synctime_end'].at_end
assert not line['synctime_start'].at_end
assert not line['synctime_4'].at_end

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(inten, label='xtrack')
plt.axhline(inten_exp, color='C1', label='expected')
plt.axhline(np.sum(p0.weight) / tw.T_rev0, color='C3', label='N/T_rev0')
plt.legend(loc='best')
plt.xlabel('Turn')
plt.ylim(inten_exp*0.95, inten_exp*1.05)

plt.figure(2)
plt.plot(p.delta, p.at_turn, '.')
plt.ylabel('Number of revolutions')
plt.xlabel(r'$\Delta P / P_0$')

plt.figure(3)
plt.plot([zz[1]-zz[0] for zz in line.log_last_track['z_range']])
plt.ylabel('z range [m]')
plt.xlabel('Turn')

plt.figure(4)
plt.plot(np.array([0.5*(zz[1] + zz[0]) for zz in line.log_last_track['z_range']]),
         label='z range center')
plt.plot([zz[0] for zz in line.log_last_track['z_range']], color='C1', linestyle='--', label='z range min')
plt.plot([zz[1] for zz in line.log_last_track['z_range']], color='C1', linestyle='--', label='z range max')
plt.plot()
plt.ylabel('z range[m]')
plt.legend(loc='best')
plt.xlabel('Turn')


plt.figure(5)
plt.pcolormesh(z_axis, np.arange(0, hist_mat.shape[0],1),
           hist_mat[:-1,:])


plt.figure(6)
plt.pcolormesh(z_axis, np.arange(0, hist_y.shape[0],1),
           hist_y[:-1,:])

plt.figure(7)
mask_alive = p.state>0
plt.plot(p.zeta[mask_alive], p.y[mask_alive], '.')
plt.axvline(x=circumference/2*tw.beta0/beta1, color='C1')
plt.axvline(x=-circumference/2*tw.beta0/beta1, color='C1')
plt.xlabel('z [m]')
plt.ylabel('x [m]')

f8 = plt.figure(8)
ax1 = plt.subplot(2, 1, 1)
plt.plot(t_unwrapped*1e6, y_vs_t, '-')
plt.ylabel('y mean [m]')
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(t_unwrapped*1e6, intensity_vs_t/np.mean(intensity_vs_t), '-')
plt.ylabel('Beam line density [a.u.]')
plt.xlabel('t [us]')
plt.ylim(bottom=0)
for tt in np.arange(0, t_unwrapped[-1], 1/f_nominal):
    for ax in [ax1, ax2]:
        ax.axvline(x=tt*1e6, color='red', linestyle='--', alpha=0.5)

# zoom
ax1.set_xlim(0, 15)


plt.show()