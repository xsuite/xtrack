# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from scipy.constants import e as echarge

from ..base_classes import Element
from .gaussian_fields import get_Ex_Ey_Gx_Gy_gauss
from .qgauss import QGauss
from scipy.interpolate import CubicSpline


class SCCoasting(Element):
    """Space charge for a coasting beam."""

    _description = [
        ("number_of_particles", "", "Number of particles in the beam", 0.0),
        ("circumference", "m", "Machine circumference", 1.0),
        ("sigma_x", "m", "Horizontal size of the beam (r.m.s.)", 1.0),
        ("sigma_y", "m", "Vertical size of the beam (r.m.s.)", 1.0),
        ("length", "m", "Integration length of space charge kick", 0.0),
        ("x_co", "m", "Horizontal closed orbit offset", 0.0),
        ("y_co", "m", "Vertical closed orbit offset", 0.0),
    ]
    _extra = [
        ("min_sigma_diff", "m", "Threshold to detect round beam", 1e-8),
        ("enabled", "", "Switch to disable space charge effect", True),
    ]

    def track(self, p):
        if self.enabled:
            charge = p.q0 * echarge

            Ex, Ey = get_Ex_Ey_Gx_Gy_gauss(
                p.x - self.x_co,
                p.y - self.y_co,
                self.sigma_x,
                self.sigma_y,
                min_sigma_diff=self.min_sigma_diff,
                skip_Gs=True,
                mathlib=p._m,
            )

            fact_kick = (
                p.chi
                * self.number_of_particles
                / self.circumference
                * (charge * p.charge_ratio)
                * charge
                * (1 - p.beta0 * p.beta0)
                / (p.p0c * echarge * p.beta0)
                * self.length
            )

            p.px += fact_kick * Ex
            p.py += fact_kick * Ey


class SCQGaussProfile(Element):
    """Space charge for a bunched beam with generalised
    Gaussian profile.
    """

    _description = [
        ("number_of_particles", "", "Number of particles in the bunch", 0.0),
        ("bunchlength_rms", "m", "Length of the bunch (r.m.s.)", 1.0),
        ("sigma_x", "m", "Horizontal size of the beam (r.m.s.)", 1.0),
        ("sigma_y", "m", "Vertical size of the beam (r.m.s.)", 1.0),
        ("length", "m", "Integration length of space charge kick", 0.0),
        ("x_co", "m", "Horizontal closed orbit offset", 0.0),
        ("y_co", "m", "Vertical closed orbit offset", 0.0),
    ]
    _extra = [
        ("min_sigma_diff", "m", "Threshold to detect round beam", 1e-8),
        ("enabled", "", "Switch to disable space charge effect", True),
        (
            "q_parameter",
            "",
            "q parameter of generalised Gaussian distribution (q=1 for standard Gaussian)",
            1.0,
        ),
    ]

    def track(self, p):
        if self.enabled:
            distr = QGauss(self.q_parameter, mathlib=p._m)
            sigma = p.zeta / p.rvv
            fact_kick = self.number_of_particles * distr.eval(
                sigma, QGauss.sqrt_beta(self.bunchlength_rms)
            )

            charge = p.q0 * echarge
            fact_kick *= p.chi * p.charge_ratio * self.length * charge * charge
            fact_kick *= 1 - p.beta0 * p.beta0
            fact_kick /= p.p0c * echarge * p.beta0

            Ex, Ey = get_Ex_Ey_Gx_Gy_gauss(
                p.x - self.x_co,
                p.y - self.y_co,
                self.sigma_x,
                self.sigma_y,
                min_sigma_diff=self.min_sigma_diff,
                skip_Gs=True,
                mathlib=p._m,
            )

            p.px += fact_kick * Ex
            p.py += fact_kick * Ey


class SCInterpolatedProfile(Element):
    """Space charge for a bunched beam with discretised profile."""

    _description = [
        ("number_of_particles", "", "Number of particles in the bunch", 0.0),
        (
            "line_density_profile",
            "1/m",
            "Discretised list of density values with integral normalised to 1",
            lambda: [1.0, 1.0],
        ),
        ("dz", "m", "Unit distance in zeta between profile points", 1.0),
        ("z0", "m", "Start zeta position of line density profile", -0.5),
        ("sigma_x", "m", "Horizontal size of the beam (r.m.s.)", 1.0),
        ("sigma_y", "m", "Vertical size of the beam (r.m.s.)", 1.0),
        ("length", "m", "Integration length of space charge kick", 0.0),
        ("x_co", "m", "Horizontal closed orbit offset", 0.0),
        ("y_co", "m", "Vertical closed orbit offset", 0.0),
    ]
    _extra = [
        (
            "method",
            "",
            "Interpolation method; 0 == linear (default), 1 == cubic spline",
            0,
        ),
        ("min_sigma_diff", "m", "Threshold to detect round beam", 1e-8),
        ("enabled", "", "Switch to disable space charge effect", True),
    ]

    def track(self, p):
        if self.enabled:
            n_prof_points = len(self.line_density_profile)
            charge = p.q0 * echarge

            Ex, Ey = get_Ex_Ey_Gx_Gy_gauss(
                p.x - self.x_co,
                p.y - self.y_co,
                self.sigma_x,
                self.sigma_y,
                min_sigma_diff=self.min_sigma_diff,
                skip_Gs=True,
                mathlib=p._m,
            )

            fact_kick = (
                p.chi
                * (charge * p.charge_ratio)
                * charge
                * (1 - p.beta0 * p.beta0)
                / (p.p0c * echarge * p.beta0)
                * self.length
            )

            absc_values = p._m.linspace(
                self.z0, self.z0 + self.dz * (n_prof_points - 1), n_prof_points
            )

            if self.method == 0:
                ld_factor = p._m.interp(
                    p.zeta, absc_values, self.line_density_profile
                )
            elif self.method == 1:
                cs = CubicSpline(absc_values, self.line_density_profile)
                ld_factor = cs(p.zeta)
            else:
                ld_factor = 1

            fact_kick *= self.number_of_particles * ld_factor
            p.px += fact_kick * Ex
            p.py += fact_kick * Ey
