import numpy as np
import xobjects as xo

from .dress import dress

pmass = 938.2720813e6


reference_vars = (
    (xo.Float64, 'q0'),
    (xo.Float64, 'mass0'),
    (xo.Float64, 'beta0'),
    (xo.Float64, 'gamma0'),
    (xo.Float64, 'p0c',)
    )

per_particle_vars = [
    (xo.Float64, 's'),
    (xo.Float64, 'x'),
    (xo.Float64, 'y'),
    (xo.Float64, 'px'),
    (xo.Float64, 'py'),
    (xo.Float64, 'zeta'),
    (xo.Float64, 'psigma'),
    (xo.Float64, 'delta'),
    (xo.Float64, 'rpp'),
    (xo.Float64, 'rvv'),
    (xo.Float64, 'chi'),
    (xo.Float64, 'charge_ratio'),
    (xo.Int64, 'particle_id'),
    (xo.Int64, 'at_element'),
    (xo.Int64, 'at_turn'),
    (xo.Int64, 'state'),
    ]

fields = {'num_particles': xo.Int64}
for tt, nn in reference_vars:
    fields[nn] = tt

for tt, nn in per_particle_vars:
    fields[nn] = tt[:]

ParticlesData = type(
        'ParticlesData',
        (xo.Struct,),
        fields)

class Particles(dress(ParticlesData)):

    def __init__(self, pysixtrack_particles=None, num_particles=None, **kwargs):

        # Initalize array sizes
        if pysixtrack_particles is not None:
            # Assuming list of pysixtrack particles
            num_particles = len(pysixtrack_particles)
            kwargs.update(
                    {kk: np.arange(num_particles)+1 for tt, kk in per_particle_vars})
            kwargs['num_particles'] = num_particles
        else:
            assert num_particles is not None

        self.xoinitialize(**kwargs)

        # Initalize arrays
        if pysixtrack_particles is not None:
            for tt, vv in reference_vars:
                vv_first = getattr(pysixtrack_particles[0], vv)
                for ii in range(self.num_particles):
                    assert getattr(
                            pysixtrack_particles[ii], vv) == vv_first
                setattr(self, vv, vv_first)
            for tt, vv in per_particle_vars:
                if vv == 'mass_ratio':
                    vv_pyst = 'mratio'
                elif vv == 'charge_ratio':
                    vv_pyst = 'qratio'
                elif vv == 'particle_id':
                    vv_pyst = 'partid'
                elif vv == 'at_element':
                    vv_pyst = 'elemid'
                elif vv == 'at_turn':
                    vv_pyst = 'turn'
                else:
                    vv_pyst = vv
                for ii in range(num_particles):
                    getattr(self, vv)[ii] = getattr(
                            pysixtrack_particles[ii], vv_pyst)

    def _set_p0c(self):
        energy0 = np.sqrt(self.p0c ** 2 + self.mass0 ** 2)
        self.beta0 = self.p0c / energy0
        self.gamma0 = energy0 / self.mass0

    def _set_delta(self):
        rep = np.sqrt(self.delta ** 2 + 2 * self.delta + 1 / self.beta0 ** 2)
        irpp = 1 + self.delta
        self.rpp = 1 / irpp
        beta = irpp / rep
        self.rvv = beta / self.beta0
        self.psigma = (
            np.sqrt(self.delta ** 2 + 2 * self.delta + 1 / self.beta0 ** 2)
            / self.beta0
            - 1 / self.beta0 ** 2
        )

    @property
    def ptau(self):
        return (
            np.sqrt(self.delta ** 2 + 2 * self.delta + 1 / self.beta0 ** 2)
            - 1 / self.beta0
        )

    def set_reference(self, p0c=7e12, mass0=pmass, q0=1):
        self.q0 = q0
        self.mass0 = mass0
        self.p0c = p0c
        return self

