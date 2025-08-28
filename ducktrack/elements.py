# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xtrack as xt
from scipy.constants import c as clight
from scipy.constants import epsilon_0
from scipy.constants import mu_0
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


class Marker(Element):

    _description = []

    def track(self, p):
        pass


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
        p.zeta += length * (1. - 1. / p.rvv * (1 + (xp ** 2 + yp ** 2) / 2))
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
        p.zeta += length - 1 / p.rvv * opd * lpzi
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
            p.zeta -= 1. / p.rvv * chi * (hxlx - hyly)
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
        tau = p.zeta / p.beta0
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
        tau = p.zeta / p.beta0
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
        tau = p.zeta / p.beta0
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


class Misalignment(Element):
    _description = [
        ("dx", "m", "Misalignment in x", 0),
        ("dy", "m", "Misalignment in y", 0),
        ("ds", "m", "Misalignment in s", 0),
        ("theta", "rad", "Rotation around the y axis (positive s to x)", 0),
        ("phi", "rad", "Rotation around the x axis (positive s to y)", 0),
        ("psi", "rad", "Rotation around the s axis (positive y to x)", 0),
        ("anchor", "", "Reference point, anchor, of the misalignment "
                       "(0 for entry, 0.5 for middle, 1 for exit, etc.)", 0),
        ("length", "m", "Length of the element to which the misalignment applies", 0),
        ("angle", "rad", "Angle of bending if applicable (0 for straight)", 0),
        ("tilt", "rad", "Tilt of the element (0 for no tilt, positive for tilt towards x)", 0),
        ("is_exit", "", "Whether we are at the exit of the element", False),
    ]

    def track(self, particles):
        if not self.is_exit:
            if self.angle:
                self.track_bent_entry(particles)
            else:
                self.track_straight_entry(particles)
        else:
            if self.angle:
                self.track_bent_exit(particles)
            else:
                self.track_straight_exit(particles)

    def track_bent_entry(self, particles):
        dx, dy, ds = self.dx, self.dy, self.ds
        theta, phi, psi = self.theta, self.phi, self.psi
        f, length, angle, tilt = self.anchor, self.length, self.angle, self.tilt

        s_phi, c_phi = np.sin(phi), np.cos(phi)
        s_theta, c_theta = np.sin(theta), np.cos(theta)
        s_psi, c_psi = np.sin(psi), np.cos(psi)
        misalignment_matrix = np.array([
            [-s_phi * s_psi * s_theta + c_psi * c_theta, -c_psi * s_phi * s_theta - c_theta * s_psi, c_phi * s_theta, dx],
            [c_phi * s_psi, c_phi * c_psi, s_phi, dy],
            [-c_theta * s_phi * s_psi - c_psi * s_theta, -c_psi * c_theta * s_phi + s_psi * s_theta, c_phi * c_theta, ds],
            [0, 0, 0, 1],
        ])

        rho = length / angle
        matrix_first_half = np.array([
            [
                (np.cos(f * angle) - 1) * np.cos(tilt) ** 2 + 1,
                (np.cos(f * angle) - 1) * np.cos(tilt) * np.sin(tilt),
                -np.cos(tilt) * np.sin(f * angle),
                rho * (np.cos(f * angle) - 1) * np.cos(tilt),
            ],
            [
                (np.cos(f * angle) - 1) * np.cos(tilt) * np.sin(tilt),
                (np.cos(f * angle) - 1) * np.sin(tilt) ** 2 + 1,
                -np.sin(f * angle) * np.sin(tilt),
                rho * (np.cos(f * angle) - 1) * np.sin(tilt),
            ],
            [
                np.cos(tilt) * np.sin(f * angle),
                np.sin(f * angle) * np.sin(tilt),
                np.cos(f * angle),
                rho * np.sin(f * angle),
            ],
            [0, 0, 0, 1],
        ])

        inv_matrix_first_half = np.linalg.inv(matrix_first_half)

        temp = matrix_first_half @ misalignment_matrix

        misaligned_entry = temp @ inv_matrix_first_half

        mis_x, mis_y, mis_s = misaligned_entry[:3, 3]
        rot_theta = np.arctan2(misaligned_entry[0, 2], misaligned_entry[2, 2])
        rot_phi = np.arctan2(misaligned_entry[1, 2], np.sqrt(misaligned_entry[1, 0] ** 2 + misaligned_entry[1, 1] ** 2))
        rot_psi = np.arctan2(misaligned_entry[1, 0], misaligned_entry[1, 1])

        line = xt.Line(
            elements=[
                xt.XYShift(dx=mis_x, dy=mis_y),
                xt.DriftExact(length=mis_s),
                xt.YRotation(angle=np.rad2deg(rot_theta)),
                xt.XRotation(angle=np.rad2deg(-rot_phi)),  # flipped angle convention
                xt.SRotation(angle=np.rad2deg(rot_psi)),
            ]
        )

        particles.zeta -= mis_s
        particles.s -= mis_s
        line.track(particles)

    def track_straight_entry(self, particles):
        dx, dy, ds = self.dx, self.dy, self.ds
        theta, phi, psi = self.theta, self.phi, self.psi
        f, length, angle = self.anchor, self.length, self.angle

        mis_x = dx - f * length * np.cos(phi) * np.sin(theta)
        mis_y = dy - f * length * np.sin(phi)
        mis_s = ds - f * length * (np.cos(phi) * np.cos(theta) - 1)

        line = xt.Line(
            elements=[
                xt.XYShift(dx=mis_x, dy=mis_y),
                xt.Solenoid(length=mis_s),
                xt.YRotation(angle=np.rad2deg(theta)),
                xt.XRotation(angle=np.rad2deg(-phi)),  # flipped angle convention
                xt.SRotation(angle=np.rad2deg(psi)),
            ]
        )

        particles.zeta -= mis_s
        particles.s -= mis_s
        line.track(particles)

    def track_bent_exit(self, particles):
        dx, dy, ds = self.dx, self.dy, self.ds
        theta, phi, psi = self.theta, self.phi, self.psi
        f, length, angle, tilt = self.anchor, self.length, self.angle, self.tilt

        s_phi, c_phi = np.sin(phi), np.cos(phi)
        s_theta, c_theta = np.sin(theta), np.cos(theta)
        s_psi, c_psi = np.sin(psi), np.cos(psi)
        misalignment_matrix = np.array([
            [-s_phi * s_psi * s_theta + c_psi * c_theta, -c_psi * s_phi * s_theta - c_theta * s_psi, c_phi * s_theta, dx],
            [c_phi * s_psi, c_phi * c_psi, s_phi, dy],
            [-c_theta * s_phi * s_psi - c_psi * s_theta, -c_psi * c_theta * s_phi + s_psi * s_theta, c_phi * c_theta, ds],
            [0, 0, 0, 1],
        ])

        m00, m01, m02, m03 = misalignment_matrix[0, :]
        m10, m11, m12, m13 = misalignment_matrix[1, :]
        m20, m21, m22, m23 = misalignment_matrix[2, :]
        inv_misalignment_matrix = np.array([
            [m00, m10, m20, -m00 * m03 - m10 * m13 - m20 * m23],
            [m01, m11, m21, -m01 * m03 - m11 * m13 - m21 * m23],
            [m02, m12, m22, -m02 * m03 - m12 * m13 - m22 * m23],
            [0, 0, 0, 1],
        ])

        f_compl = 1 - f
        rho = length / angle
        matrix_second_half = np.array([
            [
                (np.cos(f_compl * angle) - 1) * np.cos(tilt) ** 2 + 1,
                (np.cos(f_compl * angle) - 1) * np.cos(tilt) * np.sin(tilt),
                np.cos(tilt) * -np.sin(f_compl * angle),
                rho * (np.cos(f_compl * angle) - 1) * np.cos(tilt),
            ],
            [
                (np.cos(f_compl * angle) - 1) * np.cos(tilt) * np.sin(tilt),
                (np.cos(f_compl * angle) - 1) * np.sin(tilt) ** 2 + 1,
                -np.sin(f_compl * angle) * np.sin(tilt),
                rho * (np.cos(f_compl * angle) - 1) * np.sin(tilt),
            ],
            [
                -np.cos(tilt) * -np.sin(f_compl * angle),
                np.sin(f_compl * angle) * np.sin(tilt),
                np.cos(f_compl * angle),
                rho * np.sin(f_compl * angle),
            ],
            [0, 0, 0, 1],
        ])

        inv_matrix_second_half = np.linalg.inv(matrix_second_half)

        realign = inv_matrix_second_half @ inv_misalignment_matrix @ matrix_second_half

        mis_x, mis_y, mis_s = realign[:3, 3]
        rot_theta = np.arctan2(realign[0, 2], realign[2, 2])
        rot_phi = np.arctan2(realign[1, 2], np.sqrt(realign[1, 0] ** 2 + realign[1, 1] ** 2))
        rot_psi = np.arctan2(realign[1, 0], realign[1, 1])

        line = xt.Line(
            elements=[
                xt.XYShift(dx=mis_x, dy=mis_y),
                xt.Solenoid(length=mis_s),
                xt.YRotation(angle=np.rad2deg(rot_theta)),
                xt.XRotation(angle=np.rad2deg(-rot_phi)),  # flipped angle convention
                xt.SRotation(angle=np.rad2deg(rot_psi)),
            ]
        )

        particles.zeta -= mis_s
        particles.s -= mis_s
        line.track(particles)

    def track_straight_exit(self, particles):
        dx, dy, ds = self.dx, self.dy, self.ds
        theta, phi, psi = self.theta, self.phi, self.psi
        f, length, angle = self.anchor, self.length, self.angle

        mis_x = (f - 1) * length * np.cos(phi) * np.sin(theta) - dx
        mis_y = (f - 1) * length * np.sin(phi) - dy
        mis_s = (f - 1) * length * (np.cos(phi) * np.cos(theta) - 1) - ds

        line = xt.Line(
            elements=[
                xt.SRotation(angle=np.rad2deg(-psi)),
                xt.XRotation(angle=np.rad2deg(phi)),  # flipped angle convention
                xt.YRotation(angle=np.rad2deg(-theta)),
                xt.Solenoid(length=mis_s),
                xt.XYShift(dx=mis_x, dy=mis_y),
            ]
        )

        particles.zeta -= mis_s
        particles.s -= mis_s
        line.track(particles)


class Wire(Element):
    """Current-carrying wire"""

    _description = [("L_phy"  ,"m"," Physical length of the wire ",0),
                    ("L_int"  ,"m"," Integration length (embedding drift)",0),
                    ("current","A"," Current in the wire",0),
                    ("xma"    ,"m"," x position of the wire from reference trajectory",0),
                    ("yma"    ,"m"," y position of the wire from reference trajectory",0)
                    ]

    def track(self, p):
        # Data from particle
        x      = p.x
        y      = p.y
        D_x    = x-self.xma
        D_y    = y-self.yma
        R2     = D_x*D_x + D_y*D_y


        # chi = q/q0 * m0/m
        # p0c : reference particle momentum
        # q0  : reference particle charge
        chi    = p.chi
        p0c    = p.p0c
        q0     = p.q0


        # Computing the kick
        L1   = self.L_int + self.L_phy
        L2   = self.L_int - self.L_phy
        N    = mu_0*self.current*q0/(4*np.pi*p0c/clight)

        dpx  =  -N*D_x*(np.sqrt(L1*L1 + 4.0*R2) - np.sqrt(L2*L2 + 4.0*R2))/R2
        dpy  =  -N*D_y*(np.sqrt(L1*L1 + 4.0*R2) - np.sqrt(L2*L2 + 4.0*R2))/R2


        # Update the particle properties
        p.px += dpx
        p.py += dpy



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

    @classmethod
    def from_dict(cls, dct):
        dct = dct.copy()
        for kk in list(dct.keys()):
            if kk == "h" or kk == "_h":
                dct["k"] = dct[kk]
                continue
            if kk.startswith("_"):
                dct[kk[1:]] = dct[kk]
        return super(DipoleEdge, cls).from_dict(dct)

    _description = [
        ("k", "1/m", "Curvature", 0),
        ("e1", "rad", "Face angle", 0),
        ("hgap", "m", "Equivalent gap", 0),
        ("fint", "", "Fringe integral", 0),
    ]

    def track(self, p):
        tan = p._m.tan
        sin = p._m.sin
        cos = p._m.cos
        corr = 2 * self.k * self.hgap * self.fint
        r21 = self.k * tan(self.e1)
        r43 = -self.k * tan(
            self.e1 - corr / cos(self.e1) * (1 + sin(self.e1) ** 2)
        )
        p.px += r21 * p.x
        p.py += r43 * p.y

class LinearTransferMatrix(Element):
    _description = [
        ("alpha_x_0","","",0.0),
        ("beta_x_0","","",0.0),
        ("disp_x_0","","",0.0),
        ("disp_px_0","","",0.0),
        ("alpha_x_1","","",0.0),
        ("beta_x_1","","",0.0),
        ("disp_x_1","","",0.0),
        ("disp_px_1","","",0.0),
        ("alpha_y_0","","",0.0),
        ("beta_y_0","","",0.0),
        ("disp_y_0","","",0.0),
        ("disp_py_0","","",0.0),
        ("alpha_y_1","","",0.0),
        ("beta_y_1","","",0.0),
        ("disp_y_1","","",0.0),
        ("disp_py_1","","",0.0),
        ("Q_x","","",0.0),
        ("Q_y","","",0.0),
        ("beta_s","","",0.0),
        ("Q_s","","",0.0),
        ("bucket_length","","",0.0),
        ("chroma_x","","",0.0),
        ("chroma_y","","",0.0),
        ("det_xx","","",0.0),
        ("det_xy","","",0.0),
        ("det_yy","","",0.0),
        ("det_yx","","",0.0),
        ("energy_ref_increment","","",0.0),
        ("energy_increment","","",0.0),
        ("x_ref_0","","",0.0),
        ("px_ref_0","","",0.0),
        ("x_ref_1","","",0.0),
        ("px_ref_1","","",0.0),
        ("y_ref_0","","",0.0),
        ("py_ref_0","","",0.0),
        ("y_ref_1","","",0.0),
        ("py_ref_1","","",0.0),
        ("damping_rate_x","","",0.0),
        ("damping_rate_px","","",0.0),
        ("damping_rate_y","","",0.0),
        ("damping_rate_py","","",0.0),
        ("damping_rate_zeta","","",0.0),
        ("damping_rate_pzeta","","",0.0),
        ("gauss_noise_ampl_x","","",0.0),
        ("gauss_noise_ampl_px","","",0.0),
        ("gauss_noise_ampl_y","","",0.0),
        ("gauss_noise_ampl_py","","",0.0),
        ("gauss_noise_ampl_zeta","","",0.0),
        ("gauss_noise_ampl_pzeta","","",0.0),
        ("damping_matrix","","",0.0),]

    def track(self,p):
        sin = p._m.sin
        cos = p._m.cos
        sqrt = p._m.sqrt

        cos_s = cos(2.0*np.pi*self.Q_s)
        sin_s = sin(2.0*np.pi*self.Q_s)
        beta_ratio_x =  sqrt(self.beta_x_1/self.beta_x_0)
        beta_prod_x = sqrt(self.beta_x_1*self.beta_x_0)
        beta_ratio_y =  sqrt(self.beta_y_1/self.beta_y_0)
        beta_prod_y = sqrt(self.beta_y_1*self.beta_y_0)

        #Transverse linear uncoupled matrix

        # removing dispersion and close orbit
        old_x=p.x
        old_px=p.px
        old_y=p.y
        old_py=p.py

        p.x -= self.disp_x_0 * p.delta + self.x_ref_0
        p.px -= self.disp_px_0 * p.delta + self.px_ref_0
        p.y -= self.disp_y_0 * p.delta + self.y_ref_0
        p.py -= self.disp_py_0 * p.delta + self.py_ref_0
        # p.zeta += (self.disp_px_0*old_x - self.disp_x_0*old_px + self.disp_py_0*old_y - self.disp_y_0*old_py)/p.rvv

        J_x = 0.5 * (
                (1.0 + self.alpha_x_0*self.alpha_x_0)/self.beta_x_0 * p.x*p.x
                + 2*self.alpha_x_0 * p.x*p.px
                + self.beta_x_0 * p.px*p.px)
        J_y = 0.5 * (
                (1.0 + self.alpha_y_0*self.alpha_y_0)/self.beta_y_0 * p.y*p.y
                + 2*self.alpha_y_0 * p.y*p.py
                + self.beta_y_0 * p.py*p.py)
        phase = 2*np.pi*(self.Q_x+self.chroma_x*p.delta+self.det_xx*J_x+self.det_xy*J_y)
        cos_x = cos(phase)
        sin_x = sin(phase)
        phase = 2*np.pi*(self.Q_y+self.chroma_y*p.delta+self.det_yy*J_y+self.det_yx*J_x)
        cos_y = cos(phase)
        sin_y = sin(phase)

        M00_x = beta_ratio_x*(cos_x+self.alpha_x_0*sin_x)
        M01_x = beta_prod_x*sin_x
        M10_x = ((self.alpha_x_0-self.alpha_x_1)*cos_x
                      -(1+self.alpha_x_0*self.alpha_x_1)*sin_x
                       )/beta_prod_x
        M11_x = (cos_x-self.alpha_x_1*sin_x)/beta_ratio_x
        M00_y = beta_ratio_y*(cos_y+self.alpha_y_0*sin_y)
        M01_y = beta_prod_y*sin_y
        M10_y = ((self.alpha_y_0-self.alpha_y_1)*cos_y
                      -(1+self.alpha_y_0*self.alpha_y_1)*sin_y
                      )/beta_prod_y
        M11_y = (cos_y-self.alpha_y_1*sin_y)/beta_ratio_y

        p.x,p.px = M00_x*p.x + M01_x*p.px, M10_x*p.x + M11_x*p.px
        p.y,p.py = M00_y*p.y + M01_y*p.py, M10_y*p.y + M11_y*p.py

        shift = 0.0
        if self.bucket_length > 0:
            bucket_length_zeta = self.bucket_length*p.beta0*clight
            shift = bucket_length_zeta*np.floor(p.zeta/bucket_length_zeta+0.5);
            p.zeta -= shift
            
        pzeta_new = sin_s*p.zeta/self.beta_s+cos_s*p.pzeta
        zeta_new = cos_s*p.zeta-self.beta_s*sin_s*p.pzeta

        p.zeta = zeta_new + shift
        p.pzeta = pzeta_new

        if self.energy_increment !=0:
            p.add_to_energy(self.energy_increment)

        #Change energy reference
        #In the transverse plane de change is smoothed, i.e. 
        #  both the position and the momentum are scaled,
        #  rather than only the momentum.
        if self.energy_ref_increment != 0:
            old_px = p.px.copy()
            old_py = p.py.copy()
            new_energy0 = p.mass0*p.gamma0 + self.energy_ref_increment
            new_p0c = sqrt(new_energy0*new_energy0-p.mass0*p.mass0)
            new_beta0 = new_p0c / new_energy0
            new_gamma0 = new_energy0 / p.mass0
            geo_emit_factor = sqrt(p.beta0*p.gamma0/new_beta0/new_gamma0)

            p.p0c = new_p0c

            p.x *= geo_emit_factor
            p.px = old_px * geo_emit_factor
            p.y *= geo_emit_factor
            p.py = old_py * geo_emit_factor

        if not hasattr(self.damping_matrix,'__iter__'):
            if (self.damping_rate_x < 0.0 or self.damping_rate_px < 0.0
                    or self.damping_rate_y < 0.0 or self.damping_rate_py < 0.0
                    or self.damping_rate_zeta < 0.0 or self.damping_rate_pzeta < 0.0):
                raise ValueError("Damping rates cannot be negative")
            if self.damping_rate_x > 0.0:
                p.x *= 1.0-self.damping_rate_x
            if self.damping_rate_px > 0.0:
                p.px *= 1.0-self.damping_rate_px
            if self.damping_rate_y > 0.0:
                p.y *= 1.0-self.damping_rate_y
            if self.damping_rate_py > 0.0:
                p.py *= 1.0-self.damping_rate_py
            if self.damping_rate_zeta > 0.0:
                p.zeta *= 1.0-self.damping_rate_zeta
            if self.damping_rate_pzeta > 0.0:
                p.pzeta *= 1.0-self.damping_rate_pzeta
        else:
            assert np.shape(self.damping_matrix) == (6,6)
            transformation = self.damping_matrix + np.identity(6)
            p.x,p.px,p.y,p.py,p.zeta,p.pzeta = transformation[0,0]*p.x+transformation[0,1]*p.px+transformation[0,2]*p.y+transformation[0,3]*p.py+transformation[0,4]*p.zeta+transformation[0,5]*p.pzeta,\
                                               transformation[1,0]*p.x+transformation[1,1]*p.px+transformation[1,2]*p.y+transformation[1,3]*p.py+transformation[1,4]*p.zeta+transformation[1,5]*p.pzeta,\
                                               transformation[2,0]*p.x+transformation[2,1]*p.px+transformation[2,2]*p.y+transformation[2,3]*p.py+transformation[2,4]*p.zeta+transformation[2,5]*p.pzeta,\
                                               transformation[3,0]*p.x+transformation[3,1]*p.px+transformation[3,2]*p.y+transformation[3,3]*p.py+transformation[3,4]*p.zeta+transformation[3,5]*p.pzeta,\
                                               transformation[4,0]*p.x+transformation[4,1]*p.px+transformation[4,2]*p.y+transformation[4,3]*p.py+transformation[4,4]*p.zeta+transformation[4,5]*p.pzeta,\
                                               transformation[5,0]*p.x+transformation[5,1]*p.px+transformation[5,2]*p.y+transformation[5,3]*p.py+transformation[5,4]*p.zeta+transformation[5,5]*p.pzeta


        if (self.gauss_noise_ampl_x < 0.0 or self.gauss_noise_ampl_px < 0.0 
                or self.gauss_noise_ampl_y < 0.0 or self.gauss_noise_ampl_py < 0.0
                or self.gauss_noise_ampl_zeta < 0.0 or self.gauss_noise_ampl_pzeta < 0.0):
            raise ValueError("Noise amplitude cannot be negative")
        if self.gauss_noise_ampl_x > 0.0:
            p.x += self.gauss_noise_ampl_x*np.random.randn(len(p.x))
        if self.gauss_noise_ampl_x > 0.0:
            p.px += self.gauss_noise_ampl_x*np.random.randn(len(p.px))
        if self.gauss_noise_ampl_y > 0.0:
            p.y += self.gauss_noise_ampl_y*np.random.randn(len(p.y))
        if self.gauss_noise_ampl_py > 0.0:
            p.py += self.gauss_noise_ampl_py*np.random.randn(len(p.py))
        if self.gauss_noise_ampl_zeta > 0.0:
            p.zeta += self.gauss_noise_ampl_zeta*np.random.randn(len(p.zeta))
        if self.gauss_noise_ampl_pzeta > 0.0:
            p.pzeta += self.gauss_noise_ampl_pzeta*np.random.randn(len(p.pzeta))

        # re-adding dispersion and closed orbit
        old_x=p.x
        old_px=p.px
        old_y=p.y
        old_py=p.py

        p.x += self.disp_x_1 * p.delta + self.x_ref_1
        p.px += self.disp_px_1 * p.delta + self.px_ref_1 
        p.y += self.disp_y_1 * p.delta + self.y_ref_1
        p.py += self.disp_py_1 * p.delta + self.py_ref_1
        # p.zeta-= (self.disp_px_1*old_x - self.disp_x_1*old_px + self.disp_py_1*old_y - self.disp_y_1*old_py)/p.rvv


class FirstOrderTaylorMap(Element):
    _description = [
        ("length","","",0.0),
        ("m0","","",0.0),
        ("m1","","",0.0)]

    def track(self,p):
        if self.m0 is None:
            self.m0 = np.zeros(6,dtype=np.float64)
        else:
            if len(np.shape(self.m0)) != 1 or np.shape(self.m0)[0] != 6:
                raise ValueError(f'Wrong shape for m0: {np.shape(m0)}')
        if self.m1 is None:
            self.m1 = np.zeros((6,6),dtype=np.float64)
        else:
            if len(np.shape(self.m1)) != 2 or np.shape(self.m1)[0] != 6 or np.shape(self.m1)[1] != 6:
                raise ValueError(f'Wrong shape for m1: {np.shape(m1)}')

        beta0 = p.beta0
        coords0 = np.array([p.x,p.px,p.y,p.py,p.zeta/beta0,p.ptau])
        p.x = self.m0[0] + self.m1[0,0]*coords0[0] + self.m1[0,1]*coords0[1] + self.m1[0,2]*coords0[2] + self.m1[0,3]*coords0[3] + self.m1[0,4]*coords0[4] + self.m1[0,5]*coords0[5]
        p.px = self.m0[1] + self.m1[1,0]*coords0[0] + self.m1[1,1]*coords0[1] + self.m1[1,2]*coords0[2] + self.m1[1,3]*coords0[3] + self.m1[1,4]*coords0[4] + self.m1[1,5]*coords0[5]
        p.y = self.m0[2] + self.m1[2,0]*coords0[0] + self.m1[2,1]*coords0[1] + self.m1[2,2]*coords0[2] + self.m1[2,3]*coords0[3] + self.m1[2,4]*coords0[4] + self.m1[2,5]*coords0[5]
        p.py = self.m0[3] + self.m1[3,0]*coords0[0] + self.m1[3,1]*coords0[1] + self.m1[3,2]*coords0[2] + self.m1[3,3]*coords0[3] + self.m1[3,4]*coords0[4] + self.m1[3,5]*coords0[5]
        tau = self.m0[4] + self.m1[4,0]*coords0[0] + self.m1[4,1]*coords0[1] + self.m1[4,2]*coords0[2] + self.m1[4,3]*coords0[3] + self.m1[4,4]*coords0[4] + self.m1[4,5]*coords0[5]
        ptau = self.m0[5] + self.m1[5,0]*coords0[0] + self.m1[5,1]*coords0[1] + self.m1[5,2]*coords0[2] + self.m1[5,3]*coords0[3] + self.m1[5,4]*coords0[4] + self.m1[5,5]*coords0[5]
        p.delta = np.sqrt(ptau*ptau + 2.0*ptau/p.beta0+1.0)-1.0
        p.zeta = tau * beta0

        if self.length > 0.0:
            raise NotImplementedError('Radiation is not implemented')

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
    "LinearTransferMatrix",
    "FirstOrderTaylorMap"
]
