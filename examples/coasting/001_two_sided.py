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

class CoastWrap:

    def __init__(self, length, id):
        self.length = length
        assert id > 10000
        self.id = id
        self.beta1 = tw.beta0*1.1

    def track(self, particles):

        # Resume particles previously stopped
        particles.state[particles.state==-self.id] = 1
        particles.reorganize()

        zeta_prime = self.zeta_to_zeta_prime(particles.zeta,
                                             particles.beta0, particles.s)

        # Identify particles that need to be stopped
        mask_alive = particles.state > 0
        mask_stop = mask_alive & (zeta_prime < -self.length / 2)

        # Update zeta for particles that are stopped
        zeta_prime[mask_stop] += self.length
        zeta_stopped = self.zeta_prime_to_zeta(zeta_prime[mask_stop],
                                               particles.beta0[mask_stop],
                                               particles.s[mask_stop])
        particles.zeta[mask_stop] = zeta_stopped

        # Stop particles
        particles.state[mask_stop] = -self.id
        particles.at_turn[mask_stop] += 1

    def zeta_to_zeta_prime(self, zeta, beta0, s):
        beta1_beta0 = self.beta1 / beta0
        zeta_prime =  zeta * beta1_beta0 + (1 - beta1_beta0) * s
        return zeta_prime

    def zeta_prime_to_zeta(self, zeta_prime, beta0, s):
        beta0_beta1 = beta0 / self.beta1
        zeta = zeta_prime * beta0_beta1 + (1 - beta0_beta1) * s
        return zeta

circumference = line.get_length()
wrap = CoastWrap(length=circumference, id=10001)

zeta_prime_min = -circumference/2
zeta_prime_max = circumference/2
zeta_min = wrap.zeta_prime_to_zeta(zeta_prime_min, tw.beta0, 0)
zeta_max = wrap.zeta_prime_to_zeta(zeta_prime_max, tw.beta0, 0)

num_particles = 1000
p = line.build_particles(
    zeta=np.random.uniform(zeta_min, zeta_min + circumference, num_particles),
    delta=np.random.uniform(-1e-2, 0, num_particles)
)
wrap.track(p)

line.discard_tracker()
line.append_element(wrap, name='wrap')
line.build_tracker()

def intensity(line, particles):
    mask_alive = particles.state > 0
    zeta_alive = particles.zeta[mask_alive]

    zeta_min = wrap.zeta_prime_to_zeta(zeta_prime_min, tw.beta0, particles.s[0])
    zeta_max = wrap.zeta_prime_to_zeta(zeta_prime_max, tw.beta0, particles.s[0])
    assert np.all(zeta_alive >= zeta_min)
    assert np.all(zeta_alive <= zeta_max)
    return np.sum(particles.state > 0)/((zeta_max - zeta_min)/tw.beta0/clight)

line.enable_time_dependent_vars = True
line.track(p, num_turns=1000, log=xt.Log(intensity=intensity), with_progress=True)

inten = line.log_last_track['intensity']

f_rev_ave = 1 / tw.T_rev0 * (1 - tw.slip_factor * p.delta.mean())
t_rev_ave = 1 / f_rev_ave

inten_exp = num_particles / t_rev_ave

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(inten, label='xtrack')
plt.axhline(inten_exp, color='C1', label='expected')
plt.axhline(num_particles/tw.T_rev0, color='C3', label='N/T_rev0')
plt.legend(loc='best')
plt.xlabel('Turn')
plt.show()