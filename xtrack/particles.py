import xobjects as xo

from .dress import dress

pmass = 938.2720813e6

class ParticlesData(xo.Struct):

    num_particles = xo.Int64
    mass0 = xo.Float64
    beta0 = xo.Float64
    gamma0 = xo.Float64
    p0c = xo.Float64
    s = xo.Float64[:]
    x = xo.Float64[:]
    y = xo.Float64[:]
    px = xo.Float64[:]
    py = xo.Float64[:]
    zeta = xo.Float64[:]
    psigma = xo.Float64[:]
    delta = xo.Float64[:]
    rpp = xo.Float64[:]
    rvv = xo.Float64[:]
    chi = xo.Float64[:]
    charge_ratio = xo.Float64[:]
    particle_id = xo.Int64[:]
    at_element =  xo.Int64[:]
    at_turn = xo.Int64[:]
    state = xo.Int64[:]


class Particles(dress(ParticlesData)):

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

    def from_pysixtrack(self, inp, particle_index):
        assert particle_index < self.num_particles
        self.q0[particle_index] = inp.q0
        self.mass0[particle_index] = inp.mass0
        self.beta0[particle_index] = inp.beta0
        self.gamma0[particle_index] = inp.gamma0
        self.p0c[particle_index] = inp.p0c
        self.s[particle_index] = inp.s
        self.x[particle_index] = inp.x
        self.y[particle_index] = inp.y
        self.px[particle_index] = inp.px
        self.py[particle_index] = inp.py
        self.zeta[particle_index] = inp.zeta
        self.psigma[particle_index] = inp.psigma
        self.delta[particle_index] = inp.delta
        self.rpp[particle_index] = inp.rpp
        self.rvv[particle_index] = inp.rvv
        self.chi[particle_index] = inp.chi
        self.charge_ratio[particle_index] = inp.qratio
        self.particle_id[particle_index] = (
            inp.partid is not None and inp.partid or particle_index
        )
        self.at_element[particle_index] = inp.elemid
        self.at_turn[particle_index] = inp.turn
        self.state[particle_index] = inp.state
        return

    def to_pysixtrack(self, other, particle_index):
        assert particle_index < self.num_particles
        other._update_coordinates = False
        other.q0 = self.q0[particle_index]
        other.mass0 = self.mass0[particle_index]
        other.beta0 = self.beta0[particle_index]
        other.gamma0 = self.gamma0[particle_index]
        other.p0c = self.p0c[particle_index]
        other.s = self.s[particle_index]
        other.x = self.x[particle_index]
        other.y = self.y[particle_index]
        other.px = self.px[particle_index]
        other.py = self.py[particle_index]
        other.zeta = self.zeta[particle_index]
        other.psigma = self.psigma[particle_index]
        other.delta = self.delta[particle_index]
        other.chi = self.chi[particle_index]
        other.qratio = self.charge_ratio[particle_index]
        other.partid = self.particle_id[particle_index]
        other.turn = self.at_turn[particle_index]
        other.elemid = self.at_element[particle_index]
        other.state = self.state[particle_index]
        other._update_coordinates = True

        return
