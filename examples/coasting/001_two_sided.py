import numpy as np
from cpymad.madx import Madx
import xtrack as xt

from scipy.constants import c as clight

# We get the model from MAD-X
mad = Madx()
folder = ('../../test_data/elena')
mad.call(folder + '/elena.seq')
mad.call(folder + '/highenergy.str')
mad.call(folder + '/highenergy.beam')
mad.use('elena')

# Build xsuite line
seq = mad.sequence.elena
line = xt.Line.from_madx_sequence(seq)
line.particle_ref = xt.Particles(gamma0=seq.beam.gamma,
                                    mass0=seq.beam.mass * 1e9,
                                    q0=seq.beam.charge)
line.configure_bend_model(core='adaptive', edge='full', num_multipole_kicks=10)

line.twiss_default['method'] = '4d'

tw = line.twiss()
beta1 = tw.beta0 * 1.1
class CoastWrap:

    def __init__(self, circumference, id, beta1, at_start=False):
        assert id > 10000
        self.id = id
        self.beta1 = beta1
        self.circumference = circumference
        self.at_start = at_start

    def track(self, particles):

        if self.at_start:
            mask_alive = particles.state > 0
            particles.zeta[mask_alive] -= (
                self.circumference * (1 - tw.beta0 / self.beta1))

        # # ---- For debugging
        # particles.sort(interleave_lost_particles=True)
        # particles.get_table().cols['zeta state delta s at_turn'].show()
        # import pdb; pdb.set_trace()
        # particles.reorganize()

        # Resume particles previously stopped
        particles.state[particles.state==-self.id] = 1
        particles.reorganize()

        zeta_prime = self.zeta_to_zeta_prime(particles.zeta,
                                             particles.beta0, particles.s,
                                             particles.at_turn)

        # Identify particles that need to be stopped
        mask_alive = particles.state > 0
        mask_stop = mask_alive & (zeta_prime < -self.circumference / 2)

        # Update zeta for particles that are stopped
        zeta_prime[mask_stop] += self.circumference
        particles.at_turn[mask_stop] += 1
        particles.pdg_id[mask_stop] += 1 # HACK!!!!!
        zeta_stopped = self.zeta_prime_to_zeta(zeta_prime[mask_stop],
                                               particles.beta0[mask_stop],
                                               particles.s[mask_stop],
                                               particles.at_turn[mask_stop])
        # zeta_stopped -= self.circumference * (1 - tw.beta0 / self.beta1)
        particles.zeta[mask_stop] = zeta_stopped

        # Stop particles
        particles.state[mask_stop] = -self.id


        # assert np.all(particles.zeta.max() - particles.zeta.min()
        #               < self.circumference * tw.beta0 / self.beta1)

        # # ---- For debugging
        # particles.sort(interleave_lost_particles=True)
        # particles.get_table().cols['zeta state delta s at_turn'].show()
        # import pdb; pdb.set_trace()
        # particles.reorganize()

    def zeta_to_zeta_prime(self, zeta, beta0, s, at_turn):
        S_capital = s + at_turn * self.circumference
        beta1_beta0 = self.beta1 / beta0
        beta0_beta1 = beta0 / self.beta1
        zeta_full = zeta + (1 - beta0_beta1) * self.circumference * at_turn
        zeta_prime =  zeta_full * beta1_beta0 + (1 - beta1_beta0) * S_capital
        return zeta_prime

    def zeta_prime_to_zeta(self, zeta_prime, beta0, s, at_turn):
        S_capital = s + at_turn * self.circumference
        beta0_beta1 = beta0 / self.beta1
        zeta_full = zeta_prime * beta0_beta1 + (1 - beta0_beta1) * S_capital
        zeta = zeta_full - (1 - beta0_beta1) * self.circumference * at_turn
        return zeta

circumference = line.get_length()
wrap_end = CoastWrap(circumference=circumference, beta1=beta1, id=10001)
wrap_start = CoastWrap(circumference=circumference, beta1=beta1, id=10002, at_start=True)

zeta_prime_min = -circumference/2
zeta_prime_max = circumference/2
zeta_min = wrap_start.zeta_prime_to_zeta(zeta_prime_min, tw.beta0, 0, 0)
zeta_max = wrap_start.zeta_prime_to_zeta(zeta_prime_max, tw.beta0, 0, 0)

num_particles = 1000
p = line.build_particles(
    zeta=np.random.uniform(zeta_min, zeta_min + circumference, num_particles),
    delta=np.random.uniform(0e-2, 5e-2, num_particles)
)
# zeta_grid= np.linspace(zeta_max-circumference, zeta_max, 5)
# delta_grid = np.linspace(-1e-2, 0, 3)
# ZZ, DD = np.meshgrid(zeta_grid, delta_grid)
# p = line.build_particles(
#     zeta=ZZ.flatten(),
#     delta=DD.flatten()
# )
p.i_frame = 0
wrap_start.track(p)

line.discard_tracker()
line.insert_element(element=wrap_start, name='wrap_start', at_s=0)
line.append_element(wrap_end, name='wrap_end')
line.build_tracker()

def intensity(line, particles):
    mask_alive = particles.state > 0
    # particles.sort(interleave_lost_particles=True)
    # particles.get_table().cols['zeta state delta s'].show()
    # import pdb; pdb.set_trace()
    # particles.reorganize()


    return np.sum(particles.state > 0)/((zeta_max - zeta_min)/tw.beta0/clight)

def z_range(line, particles):
    mask_alive = particles.state > 0
    return particles.zeta[mask_alive].min(), particles.zeta[mask_alive].max()

line.enable_time_dependent_vars = True
line.track(p, num_turns=1000, log=xt.Log(intensity=intensity,
                                         z_range=z_range), with_progress=True)

inten = line.log_last_track['intensity']

f_rev_ave = 1 / tw.T_rev0 * (1 - tw.slip_factor * p.delta.mean())
t_rev_ave = 1 / f_rev_ave

inten_exp =  len(p.zeta) / t_rev_ave

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(inten, label='xtrack')
plt.axhline(inten_exp, color='C1', label='expected')
plt.axhline(len(p.zeta) /tw.T_rev0, color='C3', label='N/T_rev0')
plt.legend(loc='best')
plt.xlabel('Turn')

plt.figure(2)
plt.plot(p.delta, p.pdg_id, '.')
plt.ylabel('Skipped turns')
plt.xlabel(r'\delta')

plt.figure(3)
plt.plot([zz[1]-zz[0] for zz in line.log_last_track['z_range']])
plt.ylabel('z range [m]')
plt.xlabel('Turn')

plt.figure(4)
plt.plot(
    np.array([0.5*(zz[1] + zz[0]) for zz in line.log_last_track['z_range']])
    - np.arange(1000) * circumference * (1 - tw.beta0/beta1))
plt.plot()
plt.ylabel('z range center [m]')
plt.xlabel('Turn')

plt.show()