# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import json

import numpy as np
from scipy.special import wofz
from scipy.special import gamma as tgamma

from scipy.constants import m_p as pmass_kg
from scipy.constants import e as qe
from scipy.constants import c as clight
pmass = pmass_kg*clight**2/qe

from xobjects import JEncoder

def count_not_none(*lst):
    return len(lst) - sum(p is None for p in lst)

class MathlibDefault(object):

    from numpy import sqrt, exp, sin, cos, abs, pi, tan, interp, linspace
    from numpy import power as pow

    @classmethod
    def wfun(cls, z_re, z_im):
        w = wofz(z_re + 1j * z_im)
        return w.real, w.imag

    @classmethod
    def gamma(cls, arg):
        assert arg > 0.0
        return tgamma(arg)

class Pyparticles:


    def _g1(self, mass0, p0c, energy0):
        beta0 = p0c / energy0
        gamma0 = energy0 / mass0
        return mass0, beta0, gamma0, p0c, energy0

    def _g2(self, mass0, beta0, gamma0):
        energy0 = mass0 * gamma0
        p0c = energy0 * beta0
        return mass0, beta0, gamma0, p0c, energy0

    def _f1(self, mass0, p0c):
        sqrt = self._m.sqrt
        energy0 = sqrt(p0c ** 2 + mass0 ** 2)
        return self._g1(mass0, p0c, energy0)

    def _f2(self, mass0, energy0):
        sqrt = self._m.sqrt
        p0c = sqrt(energy0 ** 2 - mass0 ** 2)
        return self._g1(mass0, p0c, energy0)

    def _f3(self, mass0, beta0):
        sqrt = self._m.sqrt
        gamma0 = 1 / sqrt(1 - beta0 ** 2)
        return self._g2(mass0, beta0, gamma0)

    def _f4(self, mass0, gamma0):
        sqrt = self._m.sqrt
        beta0 = sqrt(1 - 1 / gamma0 ** 2)
        return self._g2(mass0, beta0, gamma0)

    def copy(self, index=None):
        p = self.__class__()
        for k, v in list(self.__dict__.items()):
            if type(v) in [np.ndarray, dict]:
                if index is None:
                    v = v.copy()
                else:
                    v = v[index]
            p.__dict__[k] = v
        return p

    def __init__ref(self, p0c, energy0, gamma0, beta0, mask_check=None):
        not_none = count_not_none(beta0, gamma0, p0c, energy0)
        if not_none == 0:
            p0c = 1e9
            not_none = 1
            # raise ValueError("Particles defined without energy reference")
        if not_none > 0:
            if p0c is not None:
                new = self._f1(self.mass0, p0c)
                self._update_ref(*new)
            elif energy0 is not None:
                new = self._f2(self.mass0, energy0)
                self._update_ref(*new)
            elif gamma0 is not None:
                new = self._f4(self.mass0, gamma0)
                self._update_ref(*new)
            elif beta0 is not None:
                new = self._f3(self.mass0, beta0)
                self._update_ref(*new)

            if not_none>1:
                ddd = {'beta0': beta0, 'gamma0': gamma0, 'energy0': energy0,
                       'p0c': p0c}
                for nn, vv in ddd.items():
                    if vv is None:
                        continue

                    vv_ref = getattr(self, nn)

                    if mask_check is not None:
                        vv = vv[mask_check]
                        vv_ref = vv_ref[mask_check]

                    if not np.allclose(vv, vv_ref, atol=1e-13):
                        raise ValueError(
                            f"""\
                        Provided energy reference is inconsistent:
                        p0c    = {p0c},
                        energy0     = {energy0},
                        gamma0 = {gamma0},
                        beta0  = {beta0}"""
                        )

    def __init__delta(self, delta, ptau, pzeta, mask_check=None):
        not_none = count_not_none(delta, ptau, pzeta)
        if not_none == 0:
            self.delta = 0.0
        elif not_none >= 1:
            if delta is not None:
                self.delta = delta
            elif ptau is not None:
                self.ptau = ptau
            elif pzeta is not None:
                self.pzeta = pzeta

            if not_none>1:
                ddd = {'delta': delta, 'ptau': ptau, 'pzeta': pzeta}
                for nn, vv in ddd.items():
                    if vv is None:
                        continue

                    vv_ref = getattr(self, nn)

                    if mask_check is not None:
                        vv = vv[mask_check]
                        vv_ref = vv_ref[mask_check]

                    if not np.allclose(vv, vv_ref, atol=1e-13):
                        raise ValueError(
                            f"""
                        Particles defined with inconsistent energy deviations:
                        delta  = {delta},
                        ptau     = {ptau},
                        pzeta = {pzeta}"""
                        )

    def __init__zeta(self, zeta, tau):
        not_none = count_not_none(zeta, tau)
        if not_none == 0:
            self.zeta = 0.0
        elif not_none == 1:
            if zeta is not None:
                self.zeta = zeta
            elif tau is not None:
                self.tau = tau
        else:
            raise ValueError(
                f"""\
            Particles defined with multiple time deviations:
            zeta  = {zeta},
            tau   = {tau}"""
            )

    def __init__chi(self, mass_ratio, charge_ratio, chi):
        not_none = count_not_none(mass_ratio, charge_ratio, chi)
        if not_none == 0:
            self._chi = 1.0
            self._mass_ratio = 1.0
            self._charge_ratio = 1.0
        elif not_none == 1:
            raise ValueError(
                f"""\
            Particles defined with insufficient mass/charge information:
            chi    = {chi},
            mass_ratio = {mass_ratio},
            charge_ratio = {charge_ratio}"""
            )
        elif not_none == 2:
            if chi is None:
                self._mass_ratio = mass_ratio
                self._charge_ratio = charge_ratio
                self._chi = charge_ratio / mass_ratio
            elif mass_ratio is None:
                self._chi = chi
                self._charge_ratio = charge_ratio
                self._mass_ratio = charge_ratio / chi
            elif charge_ratio is None:
                self._chi = chi
                self._mass_ratio = mass_ratio
                self._charge_ratio = chi * mass_ratio
        else:
            self._chi = chi
            self._mass_ratio = mass_ratio
            self._charge_ratio = chi * mass_ratio
            if np.allclose(self._chi, charge_ratio / mass_ratio):
                raise ValueError(
                    f"""
            Particles defined with multiple mass/charge information:
            chi    = {chi},
            mass_ratio = {mass_ratio},
            charge_ratio = {charge_ratio}"""
                )

    def __init__(
        self,
        mathlib=None,
        **kwargs,
    ):

        for kk, vv in kwargs.items():
            if isinstance(vv, list):
                kwargs[kk] = np.array(vv)


        s = kwargs.get('s', 0.0)
        x = kwargs.get('x', 0.0)
        px = kwargs.get('px', 0.0)
        y = kwargs.get('y', 0.0)
        py = kwargs.get('py', 0.0)
        delta = kwargs.get('delta', None)
        ptau = kwargs.get('ptau', None)
        pzeta = kwargs.get('pzeta', None)
        rvv = kwargs.get('rvv', None)
        zeta = kwargs.get('zeta', None)
        tau = kwargs.get('tau', None)
        mass0 = kwargs.get('mass0', pmass)
        q0 = kwargs.get('q0', 1.0)
        p0c = kwargs.get('p0c', None)
        energy0 = kwargs.get('energy0', None)
        gamma0 = kwargs.get('gamma0', None)
        beta0 = kwargs.get('beta0', None)
        chi = kwargs.get('chi', None)
        mass_ratio = kwargs.get('mass_ratio', None)
        charge_ratio = kwargs.get('charge_ratio', None)
        particle_id = kwargs.get('particle_id', None)
        parent_particle_id = kwargs.get('parent_particle_id', None)
        at_turn = kwargs.get('at_turn', None)
        state = kwargs.get('state', None)  # == 0 particle lost, == 1 particle active
        weight = kwargs.get('weight', None)
        at_element = kwargs.get('at_element', None)

        if mathlib is None:
            mathlib=MathlibDefault

        self._m = mathlib
        self.s = s
        self.x = x
        self.px = px
        self.y = y
        self.py = py
        self.zeta = zeta
        self._mass0 = mass0
        self.q0 = q0
        self._update_coordinates = False

        if state is not None and not(np.isscalar(state)):
            mask_check = state > -1e8 # Unallocated particles
        else:
            mask_check = None
        self.__init__ref(p0c, energy0, gamma0, beta0, mask_check=mask_check)
        self.__init__delta(delta, ptau, pzeta, mask_check=mask_check)
        self.__init__zeta(zeta, tau)
        self.__init__chi(chi=chi, mass_ratio=mass_ratio, charge_ratio=charge_ratio)
        self._update_coordinates = True
        length = self._check_array_length()

        if particle_id is None:
            particle_id = np.arange(length) if length is not None else 0
        self.particle_id = particle_id

        if parent_particle_id is None:
            if np.isscalar(self.particle_id):
                parent_particle_id = self.particle_id
            else:
                parent_particle_id = self.particle_id.copy()
        self.parent_particle_id = parent_particle_id

        if at_turn is None:
            at_turn = np.zeros(length) if length is not None else 0
        self.at_turn = at_turn

        if at_element is None:
            at_element = np.zeros(length) if length is not None else 0
        self.at_element = at_element

        if state is None:
            state = np.ones(length) if length is not None else 1
        self.state = state

        if weight is None:
            weight = np.ones(length, dtype=np.float64) if length is not None else 1
        self.weight = weight

        self.lost_particles = []

    def _check_array_length(self):
        names = ["x", "px", "y", "py", "zeta", "delta", "_mass0", "q0", "p0c"]
        length = None
        for nn in names:
            xx = getattr(self, nn)
            if hasattr(xx, "__iter__") and len(xx)>1:
                if length is None:
                    length = len(xx)
                else:
                    if length != len(xx):
                        raise ValueError(f"invalid length len({nn})={len(xx)}")
        return length

    Px = property(lambda p: p.px * p.p0c * p.mass_ratio)
    Py = property(lambda p: p.py * p.p0c * p.mass_ratio)
    energy = property(lambda p: (p.ptau * p.p0c + p.energy0) * p.mass_ratio)
    pc = property(lambda p: (p.delta * p.p0c + p.p0c) * p.mass_ratio)
    mass = property(lambda p: p.mass0 * p.mass_ratio)
    beta = property(lambda p: (1 + p.delta) / (1 / p.beta0 + p.ptau))
    # rvv = property(lambda self: self.beta/self.beta0)
    # rpp = property(lambda self: 1/(1+self.delta))

    rvv = property(lambda self: self._rvv)
    rpp = property(lambda self: self._rpp)

    def add_to_energy(self, energy):
        sqrt = self._m.sqrt
        deltabeta0 = self.delta * self.beta0
        ptaubeta0 = sqrt(deltabeta0 ** 2 + 2 * deltabeta0 * self.beta0 + 1) - 1
        ptaubeta0 += energy / self.energy0
        ptau = ptaubeta0 / self.beta0
        self._delta = sqrt(ptau ** 2 + 2 * ptau / self.beta0 + 1) - 1
        self._rvv = (1 + self.delta) / (1 + ptaubeta0)
        self._rpp = 1 / (1 + self.delta)

    delta = property(lambda self: self._delta)

    @delta.setter
    def delta(self, delta):
        sqrt = self._m.sqrt
        self._delta = delta
        deltabeta0 = delta * self.beta0
        ptaubeta0 = sqrt(deltabeta0 ** 2 + 2 * deltabeta0 * self.beta0 + 1) - 1
        self._rvv = (1 + self.delta) / (1 + ptaubeta0)
        self._rpp = 1 / (1 + self.delta)

    pzeta = property(lambda self: self.ptau / self.beta0)

    @pzeta.setter
    def pzeta(self, pzeta):
        self.ptau = pzeta * self.beta0

    tau = property(lambda self: self.zeta / self.beta0)

    @tau.setter
    def tau(self, tau):
        self.zeta = self.beta0 * tau

    @property
    def ptau(self):
        sqrt = self._m.sqrt
        return (
            sqrt(self.delta ** 2 + 2 * self.delta + 1 / self.beta0 ** 2)
            - 1 / self.beta0
        )

    @ptau.setter
    def ptau(self, ptau):
        sqrt = self._m.sqrt
        self.delta = sqrt(ptau ** 2 + 2 * ptau / self.beta0 + 1) - 1

    mass0 = property(lambda self: self._mass0)

    @mass0.setter
    def mass0(self, mass0):
        new = self._f1(mass0, self.p0c)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    beta0 = property(lambda self: self._beta0)

    @beta0.setter
    def beta0(self, beta0):
        new = self._f3(self.mass0, beta0)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    gamma0 = property(lambda self: self._gamma0)

    @gamma0.setter
    def gamma0(self, gamma0):
        new = self._f4(self.mass0, gamma0)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    p0c = property(lambda self: self._p0c)

    @p0c.setter
    def p0c(self, p0c):
        new = self._f1(self.mass0, p0c)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    energy0 = property(lambda self: self._energy0)

    @energy0.setter
    def energy0(self, energy0):
        new = self._f2(self.mass0, energy0)
        _abs = self._get_absolute()
        self._update_ref(*new)
        self._update_particles_from_absolute(*_abs)

    mass_ratio = property(lambda self: self._mass_ratio)

    @mass_ratio.setter
    def mass_ratio(self, mass_ratio):
        self._mass_ratio = mass_ratio
        self._chi = self._charge_ratio / self._mass_ratio

    charge_ratio = property(lambda self: self._charge_ratio)

    @charge_ratio.setter
    def charge_ratio(self, charge_ratio):
        self._charge_ratio = charge_ratio
        self._chi = charge_ratio / self._mass_ratio

    chi = property(lambda self: self._chi)

    @chi.setter
    def chi(self, chi):
        self._charge_ratio = self._chi * self._mass_ratio
        self._chi = chi

    def _get_absolute(self):
        return self.Px, self.Py, self.pc, self.energy, self.tau

    def _update_ref(self, mass0, beta0, gamma0, p0c, energy0):
        self._mass0 = mass0
        self._beta0 = beta0
        self._gamma0 = gamma0
        self._p0c = p0c
        self._energy0 = energy0

    def _update_particles_from_absolute(self, Px, Py, pc, energy, tau):
        if self._update_coordinates:
            mass_ratio = self.mass / self.mass0
            norm = mass_ratio * self.p0c
            self._mass_ratio = mass_ratio
            self._chi = self._charge_ratio / mass_ratio
            self._ptau = energy / norm - 1
            self._delta = pc / norm - 1
            self.px = Px / norm
            self.py = Py / norm
            self.zeta = tau * self.beta0

    def __repr__(self):
        out = f"""\
        mass0   = {self.mass0}
        p0c     = {self.p0c}
        energy0 = {self.energy0}
        beta0   = {self.beta0}
        gamma0  = {self.gamma0}
        s       = {self.s}
        x       = {self.x}
        px      = {self.px}
        y       = {self.y}
        py      = {self.py}
        zeta    = {self.zeta}
        delta   = {self.delta}
        ptau    = {self.ptau}
        mass_ratio  = {self.mass_ratio}
        charge_ratio  = {self.charge_ratio}
        chi     = {self.chi}
        state   = {self.state}
        weight  = {self.weight}"""
        return out

    _dict_vars = (
        "s",
        "x",
        "px",
        "y",
        "py",
        "zeta",
        "delta",
        "mass0",
        "q0",
        "p0c",
        "chi",
        "mass_ratio",
        "particle_id",
        "parent_particle_id",
        "at_turn",
        "state",
        "weight",
    )

    def remove_lost_particles(self, keep_memory=True):

        if hasattr(self.state, "__iter__"):
            mask_valid = self.state == 1

            if np.any(~mask_valid):
                if keep_memory:
                    to_trash = self.copy()  # Not exactly efficient (but robust)
                    for ff in ('_rvv',) + self._dict_vars: # _rvv needs to be updated
                                                           # before delta
                        if hasattr(getattr(self, ff), "__iter__"):
                            setattr(to_trash, ff, getattr(self, ff)[~mask_valid])
                    self.lost_particles.append(to_trash)

            for ff in ('_rvv',) + self._dict_vars:
                if hasattr(getattr(self, ff), "__iter__"):
                    setattr(self, ff, getattr(self, ff)[mask_valid])

    def to_dict(self):
        return {kk: getattr(self, kk) for kk in self._dict_vars}

    @classmethod
    def from_dict(cls, dct):
        return cls(**dct)

    def to_json(self, filename):
        with open(filename, 'w') as fid:
            json.dump(self.to_dict(), fid, cls=JEncoder)

    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as fid:
            return cls.from_dict(json.load(fid))

    def compare(self, particle, rel_tol=1e-6, abs_tol=1e-15):
        res = True
        for kk in self._dict_vars:
            v1 = getattr(self, kk)
            v2 = getattr(particle, kk)
            if v1 is not None and v2 is not None:
                diff = v1 - v2
                if hasattr(diff, "__iter__"):
                    for nn in range(len(diff)):
                        vv1 = v1[nn] if hasattr(v1, "__iter__") else v1
                        vv2 = v2[nn] if hasattr(v2, "__iter__") else v2
                        if abs(diff[nn]) > abs_tol:
                            print(f"{kk}[{nn}] {vv1} {vv2}  diff:{diff[nn]}")
                            res = False
                        if abs(vv1) > 0 and abs(diff[nn]) / vv1 > rel_tol:
                            print(f"{kk}[{nn}] {vv1} {vv2} rdiff:{diff[nn]/vv1}")
                            res = False
                else:
                    if abs(diff) > abs_tol:
                        print(f"{kk} {v1} {v2}  diff:{diff}")
                        res = False
                    if abs(v1) > 0 and abs(diff) / v1 > rel_tol:
                        print(f"{kk} {v1} {v2} rdiff:{diff/v1}")
                        res = False
        return res

    @classmethod
    def from_madx_twiss(cls, twiss):
        out = cls(
            p0c=twiss.summary.pc * 1e6,
            mass0=twiss.summary.mass * 1e6,
            q0=twiss.summary.charge,
            s=twiss.s[:],
            x=twiss.x[:],
            px=twiss.px[:],
            y=twiss.y[:],
            py=twiss.py[:],
            tau=twiss.t[:],
            ptau=twiss.pt[:],
        )
        return out

    @classmethod
    def from_madx_track(cls, mad):
        tracksumm = mad.table.tracksumm
        mad_beam = mad.sequence().beam
        out = cls(
            p0c=mad_beam.pc * 1e6,
            mass0=mad_beam.mass * 1e6,
            q0=mad_beam.charge,
            s=tracksumm.s[:],
            x=tracksumm.x[:],
            px=tracksumm.px[:],
            y=tracksumm.y[:],
            py=tracksumm.py[:],
            tau=tracksumm.t[:],
            ptau=tracksumm.pt[:],
        )
        return out

    @classmethod
    def from_list(cls, lst):
        ll = len(lst)
        dct = {nn: np.zeros(ll) for nn in cls._dict_vars}
        for ii, pp in enumerate(lst):
            for nn in cls._dict_vars:
                dct[nn][ii] = getattr(pp, nn, 0)
        return cls(**dct)