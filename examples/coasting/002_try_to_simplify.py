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

COAST_STATE_RANGE_START= 1000000

tw = line.twiss()
beta1 = tw.beta0 * 1.1
class CoastWrap:

    def __init__(self, circumference, id, beta1, at_end=False):
        assert id > COAST_STATE_RANGE_START
        self.id = id
        self.beta1 = beta1
        self.circumference = circumference
        self.at_end = at_end

    def track(self, particles):

        if (particles.state == 0).any():
            import pdb; pdb.set_trace()

        # Resume particles previously stopped
        particles.state[particles.state==-self.id] = 1
        particles.reorganize()

        beta0_beta1 = tw.beta0 / self.beta1

        # Identify particles that need to be stopped
        zeta_min = -circumference/2*tw.beta0/beta1 + particles.s * (1 - beta0_beta1)
        mask_alive = particles.state > 0
        mask_stop = mask_alive & (particles.zeta < zeta_min)

        # Update zeta for particles that are stopped
        particles.zeta[mask_stop] += beta0_beta1 * self.circumference
        particles.pdg_id[mask_stop] += 1 # HACK!!!!!

        # Stop particles
        particles.state[mask_stop] = -self.id

        if self.at_end:
            mask_alive = particles.state > 0
            particles.zeta[mask_alive] -= (
                self.circumference * (1 - tw.beta0 / self.beta1))

        if (particles.state == 0).any():
            import pdb; pdb.set_trace()

        if self.at_end and particles.at_turn[0] == 0:
            particles.state[particles.state==-COAST_STATE_RANGE_START] = 1

circumference = line.get_length()

line.discard_tracker()
s_wrap = np.linspace(0, circumference, 10)
line.cut_at_s(s_wrap)

for ii, ss in enumerate(s_wrap):
    nn = f'coast_sync_{ii}'
    line.insert_element(element=xt.Marker(), name=nn, at_s=ss)
    line[nn].iscollective = True

wrap_end = CoastWrap(circumference=circumference, beta1=beta1, id=1000001, at_end=True)
wrap_start = CoastWrap(circumference=circumference, beta1=beta1, id=10002)
wrap_mid = CoastWrap(circumference=circumference, beta1=beta1, id=10003)

line.insert_element(element=wrap_start, name='wrap_start', at_s=0)
# line.insert_element(element=wrap_mid, name='wrap_mid', at_s=circumference/2)
line.append_element(wrap_end, name='wrap_end')
line.build_tracker()

zeta_min0 = -circumference/2*tw.beta0/beta1
zeta_max0 = circumference/2*tw.beta0/beta1

num_particles = 10000
p = line.build_particles(
    zeta=np.random.uniform(zeta_max0 - circumference, zeta_max0, num_particles),
    delta=1e-2*np.random.uniform(-1, 1, num_particles),
    x_norm=0, y_norm=0
)

p.y[(p.zeta > 1) & (p.zeta < 2)] = 1e-3  # kick

mask_stop = p.zeta < zeta_min0
p.state[mask_stop] = -10000
p.zeta[mask_stop] += circumference * tw.beta0 / beta1

p0 = p.copy()




def intensity(line, particles):
    return np.sum(particles.state > 0)/((zeta_max0 - zeta_min0)/tw.beta0/clight)

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

inten_exp =  len(p.zeta) / t_rev_ave

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(inten, label='xtrack')
plt.axhline(inten_exp, color='C1', label='expected')
plt.axhline(len(p.zeta) / tw.T_rev0, color='C3', label='N/T_rev0')
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