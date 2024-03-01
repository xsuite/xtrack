import numpy as np
from cpymad.madx import Madx
import xtrack as xt

from scipy.constants import c as clight

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


line.discard_tracker()

# Install dummy collective elements
s_sync = np.linspace(0, tw.circumference, 10)
line.cut_at_s(s_sync)
for ii, ss in enumerate(s_sync):
    nn = f'sync_here_{ii}'
    line.insert_element(element=xt.Marker(), name=nn, at_s=ss)
    line[nn].iscollective = True

import xtrack.synctime as st
st.install_sync_time_at_collective_elements(line)
line.build_tracker()

beta1 = tw.beta0 / 0.9

circumference = tw.circumference
zeta_min0 = -circumference/2*tw.beta0/beta1
zeta_max0 = circumference/2*tw.beta0/beta1

num_particles = 10000
p = line.build_particles(
    zeta=np.random.uniform(zeta_max0 - circumference, zeta_max0, num_particles),
    delta=2e-2*np.random.uniform(-1, 1, num_particles),
    x_norm=0, y_norm=0
)

p.y[(p.zeta > 1) & (p.zeta < 2)] = 1e-3  # kick

mask_stop = p.zeta < zeta_min0
p.state[mask_stop] = -st.COAST_STATE_RANGE_START
p.zeta[mask_stop] += circumference * tw.beta0 / beta1

p0 = p.copy()

def intensity(line, particles):
    return np.sum(particles.weight[particles.state > 0])/(
        (zeta_max0 - zeta_min0)/tw.beta0/clight)

def z_range(line, particles):
    mask_alive = particles.state > 0
    return particles.zeta[mask_alive].min(), particles.zeta[mask_alive].max()

def long_density(line, particles):
    mask_alive = particles.state > 0
    if not(np.any(particles.at_turn[mask_alive] == 0)): # don't check at the first turn
        assert np.all(particles.zeta[mask_alive] > zeta_min0)
        assert np.all(particles.zeta[mask_alive] < zeta_max0)
    return np.histogram(particles.zeta[mask_alive], bins=200,
                        range=(zeta_min0, zeta_max0))

def y_mean_hist(line, particles):

    mask_alive = particles.state > 0
    if not(np.any(particles.at_turn[mask_alive] == 0)): # don't check at the first turn
        assert np.all(particles.zeta[mask_alive] > zeta_min0)
        assert np.all(particles.zeta[mask_alive] < zeta_max0)
    return np.histogram(particles.zeta[mask_alive], bins=200,
                        range=(zeta_min0, zeta_max0), weights=particles.y[mask_alive])


line.enable_time_dependent_vars = True
line.track(p, num_turns=200, log=xt.Log(intensity=intensity,
                                         long_density=long_density,
                                         y_mean_hist=y_mean_hist,
                                         z_range=z_range,
                                         ), with_progress=10)

inten = line.log_last_track['intensity']

f_rev_ave = 1 / tw.T_rev0 * (1 - tw.slip_factor * p.delta.mean())
t_rev_ave = 1 / f_rev_ave

inten_exp =  np.sum(p0.weight) / t_rev_ave

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(inten, label='xtrack')
plt.axhline(inten_exp, color='C1', label='expected')
plt.legend(loc='best')
plt.xlabel('Turn')

plt.figure(2)
plt.plot(p.delta, p.at_turn, '.')
plt.ylabel('Number of turns')
plt.xlabel(r'$\delta$')

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

z_axis = line.log_last_track['long_density'][0][1]

hist_mat = np.array([rr[0] for rr in line.log_last_track['long_density']])
plt.figure(5)
plt.pcolormesh(z_axis, np.arange(0, hist_mat.shape[0],1),
           hist_mat[:-1,:])

hist_y = np.array([rr[0] for rr in line.log_last_track['y_mean_hist']])
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

dz = z_axis[1] - z_axis[0]
y_vs_t = np.fliplr(hist_y).flatten() # need to flip because of the minus in z = -beta0 c t
intensity_vs_t = np.fliplr(hist_mat).flatten()
z_unwrapped = np.arange(0, len(y_vs_t)) * dz
t_unwrapped = z_unwrapped / (tw.beta0 * clight)

z_range_size = z_axis[-1] - z_axis[0]
t_range_size = z_range_size / (tw.beta0 * clight)

plt.figure(8)
ax1 = plt.subplot(2, 1, 1)
plt.plot(t_unwrapped*1e6, y_vs_t, '-')
plt.ylabel('y mean [m]')
plt.grid()
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(t_unwrapped*1e6, intensity_vs_t, '-')
plt.ylabel('intensity')
plt.xlabel('t [us]')
for tt in t_range_size * np.arange(0, hist_y.shape[0]):
    ax1.axvline(x=tt*1e6, color='red', linestyle='--', alpha=0.5)


plt.show()