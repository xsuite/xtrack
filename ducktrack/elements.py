import numpy as np
from scipy.constants import c as clight
from scipy.constants import epsilon_0
from scipy.constants import m_e as me_kg
from scipy.constants import e as qe

me = me_kg*clight**2/qe

from .base_classes import Element
from .be_beamfields.beambeam import BeamBeam4D
from .be_beamfields.beambeam import BeamBeam6D
from .be_beamfields.spacecharge import SCCoasting
from .be_beamfields.spacecharge import SCQGaussProfile
from .be_beamfields.spacecharge import SCInterpolatedProfile

_factorial = np.array(
    [
        1,
        1,
        2,
        6,
        24,
        120,
        720,
        5040,
        40320,
        362880,
        3628800,
        39916800,
        479001600,
        6227020800,
        87178291200,
        1307674368000,
        20922789888000,
        355687428096000,
        6402373705728000,
        121645100408832000,
        2432902008176640000,
    ]
)


class Drift(Element):
    """Drift in expanded form"""

    _description = [("length", "m", "Length of the drift", 0)]

    def track(self, p):
        length = self.length
        rpp = p.rpp
        xp = p.px * rpp
        yp = p.py * rpp
        p.x += xp * length
        p.y += yp * length
        p.zeta += length * (p.rvv - (1 + (xp ** 2 + yp ** 2) / 2))
        p.s += length


class DriftExact(Drift):
    """Drift in exact form"""

    _description = [("length", "m", "Length of the drift", 0)]

    def track(self, p):
        sqrt = p._m.sqrt
        length = self.length
        opd = 1 + p.delta
        lpzi = length / sqrt(opd ** 2 - p.px ** 2 - p.py ** 2)
        p.x += p.px * lpzi
        p.y += p.py * lpzi
        p.zeta += p.rvv * length - opd * lpzi
        p.s += length


def _arrayofsize(ar, size):
    ar = np.array(ar)
    if len(ar) == 0:
        return np.zeros(size, dtype=ar.dtype)
    elif len(ar) < size:
        ar = np.hstack([ar, np.zeros(size - len(ar), dtype=ar.dtype)])
    return ar


class Multipole(Element):
    """ Multipole """

    _description = [
        (
            "knl",
            "m^-n",
            "Normalized integrated strength of normal components",
            lambda: [0],
        ),
        (
            "ksl",
            "m^-n",
            "Normalized integrated strength of skew components",
            lambda: [0],
        ),
        (
            "hxl",
            "rad",
            "Rotation angle of the reference trajectory"
            "in the horizzontal plane",
            0,
        ),
        (
            "hyl",
            "rad",
            "Rotation angle of the reference trajectory in the vertical plane",
            0,
        ),
        ("length", "m", "Length of the originating thick multipole", 0),
    ]

    @property
    def order(self):
        return max(len(self.knl), len(self.ksl)) - 1

    def track(self, p):
        order = self.order
        length = self.length
        knl = _arrayofsize(self.knl, order + 1)
        ksl = _arrayofsize(self.ksl, order + 1)
        x = p.x
        y = p.y
        chi = p.chi
        dpx = knl[order]
        dpy = ksl[order]
        for ii in range(order, 0, -1):
            zre = (dpx * x - dpy * y) / ii
            zim = (dpx * y + dpy * x) / ii
            dpx = knl[ii - 1] + zre
            dpy = ksl[ii - 1] + zim
        dpx = -chi * dpx
        dpy = chi * dpy
        # curvature effect kick
        hxl = self.hxl
        hyl = self.hyl
        delta = p.delta
        if hxl != 0 or hyl != 0:
            b1l = chi * knl[0]
            a1l = chi * ksl[0]
            hxlx = hxl * x
            hyly = hyl * y
            if length > 0:
                hxx = hxlx / length
                hyy = hyly / length
            else:  # non physical weak focusing disabled (SixTrack mode)
                hxx = 0
                hyy = 0
            dpx += hxl + hxl * delta - b1l * hxx
            dpy -= hyl + hyl * delta - a1l * hyy
            p.zeta -= chi * (hxlx - hyly)
        p.px += dpx
        p.py += dpy


class RFMultipole(Element):
    """
    H= -l sum   Re[ (kn[n](zeta) + i ks[n](zeta) ) (x+iy)**(n+1)/ n ]

    kn[n](z) = k_n cos(2pi w tau + pn/180*pi)
    ks[n](z) = k_n cos(2pi w tau + pn/180*pi)

    """

    _description = [
        ("voltage", "volt", "Voltage", 0),
        ("frequency", "hertz", "Frequency", 0),
        ("lag", "degree", "Delay in the cavity sin(lag - w tau)", 0),
        ("knl", "", "...", lambda: [0]),
        ("ksl", "", "...", lambda: [0]),
        ("pn", "", "...", lambda: [0]),
        ("ps", "", "...", lambda: [0]),
    ]

    @property
    def order(self):
        return max(len(self.knl), len(self.ksl)) - 1

    def track(self, p):
        sin = p._m.sin
        cos = p._m.cos
        pi = p._m.pi
        order = self.order
        k = 2 * pi * self.frequency / clight
        tau = p.zeta / p.rvv / p.beta0
        ktau = k * tau
        deg2rad = pi / 180
        knl = _arrayofsize(self.knl, order + 1)
        ksl = _arrayofsize(self.ksl, order + 1)
        pn = _arrayofsize(self.pn, order + 1) * deg2rad
        ps = _arrayofsize(self.ps, order + 1) * deg2rad
        x = p.x
        y = p.y
        dpx = 0
        dpy = 0
        dptr = 0
        zre = 1
        zim = 0
        for ii in range(order + 1):
            pn_ii = pn[ii] - ktau
            ps_ii = ps[ii] - ktau
            cn = cos(pn_ii)
            sn = sin(pn_ii)
            cs = cos(ps_ii)
            ss = sin(ps_ii)
            # transverse kick order i!
            dpx += cn * knl[ii] * zre - cs * ksl[ii] * zim
            dpy += cs * ksl[ii] * zre + cn * knl[ii] * zim
            # compute z**(i+1)/(i+1)!
            zret = (zre * x - zim * y) / (ii + 1)
            zim = (zim * x + zre * y) / (ii + 1)
            zre = zret
            fnr = knl[ii] * zre
            # fni = knl[ii] * zim
            # fsr = ksl[ii] * zre
            fsi = ksl[ii] * zim
            # energy kick order i+1
            dptr += sn * fnr - ss * fsi

        chi = p.chi
        p.px += -chi * dpx
        p.py += chi * dpy
        dv0 = self.voltage * sin(self.lag * deg2rad - ktau)
        p.add_to_energy(p.charge_ratio * p.q0 * (dv0 - p.p0c * k * dptr))


class Cavity(Element):
    """Radio-frequency cavity"""

    _description = [
        ("voltage", "V", "Integrated energy change", 0),
        ("frequency", "Hz", "Frequency of the cavity", 0),
        ("lag", "degree", "Delay in the cavity sin(lag - w tau)", 0),
    ]

    def track(self, p):
        sin = p._m.sin
        pi = p._m.pi
        k = 2 * pi * self.frequency / clight
        tau = p.zeta / p.rvv / p.beta0
        phase = self.lag * pi / 180 - k * tau
        p.add_to_energy(p.charge_ratio * p.q0 * self.voltage * sin(phase))


class SawtoothCavity(Element):
    """Radio-frequency cavity"""

    _description = [
        ("voltage", "V", "Integrated energy change", 0),
        ("frequency", "Hz", "Equivalent Frequency of the cavity", 0),
        ("lag", "degree", "Delay in the cavity `lag - w tau`", 0),
    ]

    def track(self, p):
        pi = p._m.pi
        k = 2 * pi * self.frequency / clight
        tau = p.zeta / p.rvv / p.beta0
        phase = self.lag * pi / 180 - k * tau
        phase = (phase + pi) % (2 * pi) - pi
        p.add_to_energy(p.charge_ratio * p.q0 * self.voltage * phase)


class XYShift(Element):
    """shift of the reference"""

    _description = [
        ("dx", "m", "Horizontal shift", 0),
        ("dy", "m", "Vertical shift", 0),
    ]

    def track(self, p):
        p.x -= self.dx
        p.y -= self.dy




class Elens(Element):
    """Hollow Electron Lens"""

    _description = [("voltage", "V", "Voltage of the electron lens", 0),
                    ("current", "A", "Current of the e-beam", 0),
                    ("inner_radius", "m", "Inner radius of the hollow e-beam", 0),
                    ("outer_radius", "m", "Outer radius of the hollow e-beam", 0),
                    ("ebeam_center_x", "m", "Center of the e-beam in x", 0),
                    ("ebeam_center_y", "m", "Center of the e-beam in y", 0),
                    ("elens_length", "m", "Length of the hollow electron lens", 0)
                    ]

    def track(self, p):

        # vacuum permittivity
        epsilon0 = epsilon_0
        pi       = np.pi             # pi
        e_mass   = me                # electron mass

        # get the transverse amplitude
        # TO DO: needs to be modified for off-centererd e-beam
        r = np.sqrt(p.x**2 + p.y**2)

        # magnetic rigidity

        if type(p.p0c) is float:
            Brho = p.p0c/(p.q0*clight)
        else:
            Brho = p.p0c[0]/(p.q0*clight)


        # Electron properties
        Ekin_e = self.voltage                         # kinetic energy
        Etot_e = Ekin_e + e_mass                      # total energy
        p_e    = np.sqrt(Etot_e**2 - e_mass**2)       # electron momentum
        beta_e = p_e/Etot_e                           # relativ. beta

        # relativistic beta  of protons
        beta_p = p.rvv*p.beta0

        # abbreviate for better readability
        r1 = self.inner_radius
        r2 = self.outer_radius
        I  = self.current

        # geometric factor frr
        frr = ((r**2 - r1**2)/(r2**2 - r1**2))  # uniform distribution

        try:
            frr = [max(0,iitem) for iitem in frr]
            frr = [min(1,iitem) for iitem in frr]
            frr = np.array(frr, dtype = float)

        except TypeError:
            frr = max(0,frr)
            frr = min(1,frr)
            frr = np.array([frr], dtype=float)


        #
        #
        # if len(frr)>0:
        #     frr[frr<0] = 0
        #     frr[frr>1] = 1

        # calculate the kick at r2 (maximum kick)
        theta_max = ((1/(4*pi*epsilon0))*(2*self.elens_length*I)*
                    (1+beta_e*beta_p)*(1/(r2*Brho*beta_e*beta_p*clight**2)))

        # calculate the kick of the particles
        # the (-1) stems from the attractive force of the E-field
        theta = (-1)*theta_max*r2*p.rpp*p.chi

        print("type frr", type(frr))
        print("type r", type(r))

        theta = theta*np.divide(frr, r, out=np.zeros_like(frr), where=r!=0)

        # convert px and py to x' and y'
        xp   = p.px * p.rpp
        yp   = p.py * p.rpp

        # update xp and yp with the HEL kick
        # use np.divide to not crash when r=0
        xp = xp + p.x*np.divide(theta, r, out=np.zeros_like(theta), where=r!=0)
        yp = yp + p.y*np.divide(theta, r, out=np.zeros_like(theta), where=r!=0)

        # update px and py.
        p.px = xp/p.rpp
        p.py = yp/p.rpp



class SRotation(Element):
    """anti-clockwise rotation of the reference frame"""

    _description = [("angle", "", "Rotation angle", 0)]

    def track(self, p):
        deg2rag = p._m.pi / 180
        cz = p._m.cos(self.angle * deg2rag)
        sz = p._m.sin(self.angle * deg2rag)
        xn = cz * p.x + sz * p.y
        yn = -sz * p.x + cz * p.y
        p.x = xn
        p.y = yn
        xn = cz * p.px + sz * p.py
        yn = -sz * p.px + cz * p.py
        p.px = xn
        p.py = yn


class LimitRect(Element):
    _description = [
        ("min_x", "m", "Minimum horizontal aperture", -1.0),
        ("max_x", "m", "Maximum horizontal aperture", 1.0),
        ("min_y", "m", "Minimum vertical aperture", -1.0),
        ("max_y", "m", "Minimum vertical aperture", 1.0),
    ]

    def track(self, particle):

        x = particle.x
        y = particle.y

        if not hasattr(particle.state, "__iter__"):
            particle.state = int(
                x >= self.min_x
                and x <= self.max_x
                and y >= self.min_y
                and y <= self.max_y
            )
        else:
            particle.state = np.int_(
                (x >= self.min_x)
                & (x <= self.max_x)
                & (y >= self.min_y)
                & (y <= self.max_y)
            )
            particle.remove_lost_particles()


class LimitEllipse(Element):
    _description = [
        ("a", "m", "Horizontal semiaxis", 1.0),
        ("b", "m", "Vertical semiaxis", 1.0),
    ]

    def track(self, particle):

        x = particle.x
        y = particle.y

        if not hasattr(particle.state, "__iter__"):
            particle.state = int(
                x * x / (self.a * self.a) + y * y / (self.b * self.b) <= 1.0
            )
        else:
            particle.state = np.int_(
                x * x / (self.a * self.a) + y * y / (self.b * self.b) <= 1.0
            )
            particle.remove_lost_particles()


class LimitRectEllipse(Element):
    _description = [
        ("max_x", "m", "Maximum horizontal aperture", 1.0),
        ("max_y", "m", "Maximum vertical aperture", 1.0),
        ("a", "m", "Horizontal semiaxis", 1.0),
        ("b", "m", "Vertical semiaxis", 1.0),
    ]

    def track(self, particle):

        x = particle.x
        y = particle.y

        if not hasattr(particle.state, "__iter__"):
            particle.state = int(
                x >= -self.max_x
                and x <= self.max_x
                and y >= -self.max_y
                and y <= self.max_y
                and x * x / (self.a * self.a) + y * y / (self.b * self.b) <= 1.0
            )
        else:
            particle.state = np.int_(
                (x >= -self.max_x)
                & (x <= self.max_x)
                & (y >= -self.max_y)
                & (y <= self.max_y)
                & (x * x / (self.a * self.a) + y * y / (self.b * self.b) <= 1.0)
            )
            particle.remove_lost_particles()

class LimitPolygon(Element):
    _description = [
        ("x_vertices", "m", "Horizontal vertices coordinates", ()),
        ("y_vertices", "m", "Vertical vertices coordinates", ()),
    ]

    def track(self, particle):
        raise NotImplementedError

class BeamMonitor(Element):
    _description = [
        ("num_stores", "", "...", 0),
        ("start", "", "...", 0),
        ("skip", "", "...", 1),
        ("max_particle_id", "", "", 0),
        ("min_particle_id", "", "", 0),
        ("is_rolling", "", "", False),
        ("is_turn_ordered", "", "", True),
        ("data", "", "...", lambda: []),
    ]

    def offset(self, particle):
        _offset = -1
        nn = (
            self.max_particle_id >= self.min_particle_id
            and (self.max_particle_id - self.min_particle_id + 1)
            or -1
        )
        assert self.is_turn_ordered

        if (
            particle.turn >= self.start
            and nn > 0
            and particle.particle_id >= self.min_particle_id
            and particle.particle_id <= self.max_particle_id
        ):
            turns_since_start = particle.turns - self.start
            store_index = turns_since_start // self.skip
            if store_index < self.num_stores:
                pass
            elif self.is_rolling:
                store_index = store_index % self.num_stores
            else:
                store_index = -1

            if store_index >= 0:
                _offset = store_index * nn + particle.particle_id

        return _offset

    def track(self, p):
        self.data.append(p.copy)


class DipoleEdge(Element):
    _description = [
        ("h", "1/m", "Curvature", 0),
        ("e1", "rad", "Face angle", 0),
        ("hgap", "m", "Equivalent gap", 0),
        ("fint", "", "Fringe integral", 0),
    ]

    def track(self, p):
        tan = p._m.tan
        sin = p._m.sin
        cos = p._m.cos
        corr = 2 * self.h * self.hgap * self.fint
        r21 = self.h * tan(self.e1)
        r43 = -self.h * tan(
            self.e1 - corr / cos(self.e1) * (1 + sin(self.e1) ** 2)
        )
        p.px += r21 * p.x
        p.py += r43 * p.y


__all__ = [
    "BeamBeam4D",
    "BeamBeam6D",
    "BeamMonitor",
    "Cavity",
    "DipoleEdge",
    "Drift",
    "DriftExact",
    "Element",
    "Elens",
    "LimitEllipse",
    "LimitRect",
    "Multipole",
    "RFMultipole",
    "SCCoasting",
    "SCInterpolatedProfile",
    "SCQGaussProfile",
    "SRotation",
    "XYShift",
]
