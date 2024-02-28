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

    def track(self, particles):

        particles.state[particles.state==-self.id] = 1
        particles.reorganize()

        mask_stop = particles.zeta < -self.length / 2
        particles.state[mask_stop] = -self.id
        particles.at_turn[mask_stop] += 1
        particles.zeta[mask_stop] += self.length

circumference = line.get_length()
num_particles = 10000
p = line.build_particles(
    zeta=np.random.uniform(-circumference/2, circumference/2, num_particles),
    delta=np.random.uniform(-1e-2, 0, num_particles)
)

wrap = CoastWrap(length=circumference, id=10001)

line.discard_tracker()
line.append_element(wrap, name='wrap')
line.build_tracker()

def intensity(line, particles):
    return np.sum(particles.state > 0)/(circumference/tw.beta0/clight)

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