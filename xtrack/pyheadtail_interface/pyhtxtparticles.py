import numpy as np
from scipy.constants import e,c

from PyHEADTAIL.particles.particles import Particles as PyHtParticles
from ..particles import Particles as XtParticles


class PyHtXtParticles(XtParticles,PyHtParticles):

    def __init__(self,circumference=None,particlenumber_per_mp=1.0, **kwargs):
        super(PyHtXtParticles,self).__init__(**kwargs)
        self.circumference = circumference
        self.particlenumber_per_mp = particlenumber_per_mp
        self._slice_sets = {}
        self.coords_n_momenta = set(['x','xp','y','yp','z','dp'])

    @classmethod
    def from_pyheadtail(cls, particles):
        new  = cls(num_particles=particles.macroparticlenumber)

        new.particlenumber_per_mp = particles.particlenumber_per_mp
        new.charge = particles.charge
        new.mass = particles.mass
        new.circumference = particles.circumference
        new.gamma = particles.gamma
        new.x = particles.x
        new.xp = particles.xp
        new.y = particles.y
        new.yp = particles.yp
        new.z = particles.z
        new.dp = particles.dp

        return new

    @property
    def z(self):
        return self.zeta

    @z.setter
    def z(self, value):
        self.zeta = value

    @property
    def xp(self):
        return self.px

    @xp.setter
    def xp(self, value):
        self.px = value

    @property
    def yp(self):
        return self.py

    @yp.setter
    def yp(self, value):
        self.py = value

    @property
    def dp(self):
        return self.delta

    @dp.setter
    def dp(self, value):
        self._update_delta(value)

    @property
    def mass(self):
        return self.mass0/(c*c)*e

    @mass.setter
    def mass(self, value):
        self.mass0 = value/e*c*c

    @property
    def charge(self):
        return self.q0*e

    @charge.setter
    def charge(self, value):
        self.q0 = value/e

    @property
    def macroparticlenumber(self):
        return self.num_particles

    @property
    def particlenumber_per_mp(self):
        return self.weight[0] # I avoid checking that they are all the same
                         # not to compromise on performance

    @particlenumber_per_mp.setter
    def particlenumber_per_mp(self, value):
        self.weight[:] = value

    @property
    def _gamma(self):
	# I assume that they are all the same and take the first
        # An assert would be too expensive...
        return self.gamma0[0]

    @_gamma.setter
    def _gamma(self, value):
        self.gamma0[:] = value

    @property
    def _beta(self):
	# I assume that they are all the same and take the first
        # An assert would be too expensive...
        return self.beta0[0]

    @_beta.setter
    def _beta(self, value):
        self.beta0[:] = value

    @property
    def _p0(self):
	# I assume that they are all the same and take the first
        # An assert would be too expensive...
        return self.p0c[0]/c*e

    @_p0.setter
    def _p0(self, value):
        self.p0c[:] = value/e*c

    @property
    def id(self):
        return self.particle_id

