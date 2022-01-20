import numpy as np
from scipy.constants import e as qe
from scipy.constants import c as clight

from ..base_classes import Element
from .gaussian_fields import get_Ex_Ey_Gx_Gy_gauss
from . import BB6D
from . import BB6Ddata


class BeamBeam4D(Element):
    """Interaction with a transverse-Gaussian strong beam (4D modelling)."""

    _description = [
        (
            "charge",
            "e",
            "Charge of the interacting distribution (strong beam)",
            0,
        ),
        ("sigma_x", "m", "Horizontal size of the strong beam (r.m.s.)", 1.0),
        ("sigma_y", "m", "Vertical size of the strong beam (r.m.s.)", 1.0),
        ("beta_r", "", "Relativistic beta of the stron beam", 1.0),
        (
            "x_bb",
            "m",
            "H. position of the strong beam w.r.t. the reference trajectory",
            0,
        ),
        (
            "y_bb",
            "m",
            "V. position of the strong beam w.r.t. the reference trajectory",
            0,
        ),
        ("d_px", "", "H. kick subtracted after the interaction.", 0),
        ("d_py", "", "V. kick subtracted after the interaction.", 0),
    ]

    _extra = [
        ("min_sigma_diff", "m", "Threshold to detect round beam", 1e-28),
        ("enabled", "", "Switch for closed orbit search", True),
    ]

    def track(self, p):
        if self.enabled:
            charge = p.charge_ratio * p.q0
            x = p.x - self.x_bb
            px = p.px
            y = p.y - self.y_bb
            py = p.py

            chi = p.chi

            beta = p.beta0 / p.rvv
            p0c = p.p0c * qe

            Ex, Ey = get_Ex_Ey_Gx_Gy_gauss(
                x,
                y,
                self.sigma_x,
                self.sigma_y,
                min_sigma_diff=1e-10,
                skip_Gs=True,
                mathlib=p._m,
            )

            fact_kick = (
                chi
                * self.charge
                * qe
                * charge
                * qe
                * (1.0 + beta * self.beta_r)
                / (p0c * (beta + self.beta_r))
            )

            px += fact_kick * Ex - self.d_px
            py += fact_kick * Ey - self.d_py

            p.px = px
            p.py = py


class BeamBeam6D(Element):
    """Interaction with a transverse-Gaussian strong beam (6D modelling).

    http://cds.cern.ch/record/2306400
    """

    _description = [
        (
            "phi",
            "rad",
            "Crossing angle (>0 weak beam increases"
            "x in the direction motion)",
            0,
        ),
        ("alpha", "rad", "Crossing plane tilt angle (>0 x tends to y)", 0),
        (
            "x_bb_co",
            "m",
            "H. position of the strong beam w.r.t. the closed orbit",
            0,
        ),
        (
            "y_bb_co",
            "m",
            "V. position of the strong beam w.r.t. the closed orbit",
            0,
        ),
        (
            "charge_slices",
            "qe",
            "Charge of the interacting slices (strong beam)",
            (0.0),
        ),
        (
            "zeta_slices",
            "m",
            "Longitudinal position of the interacting"
            "slices (>0 head of the strong).",
            (0.0),
        ),
        (
            "sigma_11",
            "m^2",
            "Sigma_11 element of the sigma matrix of the strong beam",
            1.0,
        ),
        (
            "sigma_12",
            "m",
            "Sigma_12 element of the sigma matrix of the strong beam",
            0,
        ),
        (
            "sigma_13",
            "m^2",
            "Sigma_13 element of the sigma matrix of the strong beam",
            0,
        ),
        (
            "sigma_14",
            "m",
            "Sigma_14 element of the sigma matrix of the strong beam",
            0,
        ),
        (
            "sigma_22",
            "",
            "Sigma_22 element of the sigma matrix of the strong beam",
            0,
        ),
        (
            "sigma_23",
            "m",
            "Sigma_23 element of the sigma matrix of the strong beam",
            0,
        ),
        (
            "sigma_24",
            "",
            "Sigma_24 element of the sigma matrix of the strong beam",
            0,
        ),
        (
            "sigma_33",
            "m^2",
            "Sigma_33 element of the sigma matrix of the strong beam",
            1.0,
        ),
        (
            "sigma_34",
            "m",
            "Sigma_34 element of the sigma matrix of the strong beam",
            0,
        ),
        (
            "sigma_44",
            "",
            "Sigma_44 element of the sigma matrix of the strong beam",
            0,
        ),
        ("x_co", "m", "x coordinate the closed orbit (weak beam).", 0),
        ("px_co", "", "px coordinate the closed orbit (weak beam).", 0),
        ("y_co", "m", "y coordinate the closed orbit (weaek beam).", 0),
        ("py_co", "", "py coordinate the closed orbit (weaek beam).", 0),
        ("zeta_co", "m", "zeta coordinate the closed orbit (weaek beam).", 0),
        ("delta_co", "", "delta coordinate the closed orbit (weaek beam).", 0),
        ("d_x", "m", "Quantity subtracted from x after the interaction.", 0),
        ("d_px", "", "Quantity subtracted from px after the interaction.", 0),
        ("d_y", "m", "Quantity subtracted from y after the interaction.", 0),
        ("d_py", "", "Quantity subtracted from py after the interaction.", 0),
        (
            "d_zeta",
            "m",
            "Quantity subtracted from sigma after the interaction.",
            0,
        ),
        (
            "d_delta",
            "",
            "Quantity subtracted from delta after the interaction.",
            0,
        ),
    ]
    _extra = [
        ("min_sigma_diff", "m", "Threshold to detect round beam", 1e-28),
        (
            "threshold_singular",
            "",
            "Threshold to detect small denominator",
            1e-28,
        ),
        ("enabled", "", "Switch for closed orbit search", True),
    ]

    def track(self, p):
        if self.enabled:
            bb6data = BB6Ddata.BB6D_init(
                qe,
                self.phi,
                self.alpha,
                self.x_bb_co,
                self.y_bb_co,
                np.atleast_1d(self.charge_slices),
                np.atleast_1d(self.zeta_slices),
                self.sigma_11,
                self.sigma_12,
                self.sigma_13,
                self.sigma_14,
                self.sigma_22,
                self.sigma_23,
                self.sigma_24,
                self.sigma_33,
                self.sigma_34,
                self.sigma_44,
                self.x_co,
                self.px_co,
                self.y_co,
                self.py_co,
                self.zeta_co,
                self.delta_co,
                self.min_sigma_diff,
                self.threshold_singular,
                self.d_x,
                self.d_px,
                self.d_y,
                self.d_py,
                self.d_zeta,
                self.d_delta,
                self.enabled,
            )
            (
                x_ret,
                px_ret,
                y_ret,
                py_ret,
                zeta_ret,
                delta_ret,
            ) = BB6D.BB6D_track(
                p.x,
                p.px,
                p.y,
                p.py,
                p.zeta,
                p.delta,
                p.q0 * qe,
                p.p0c / clight * qe,
                bb6data,
                mathlib=p._m,
            )
            self._last_bb6data = bb6data
            p.x = x_ret
            p.px = px_ret
            p.y = y_ret
            p.py = py_ret
            p.zeta = zeta_ret
            if hasattr(p, 'update_delta'):
                p._update_delta(delta_ret)
            else:
                p.delta = delta_ret
