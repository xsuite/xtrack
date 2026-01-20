import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

import xobjects as xo
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

from xtrack.beam_elements.spline_param_schema import SplineParameterSchema

def test_splineboris_homogeneous():
    """
    Test the SplineBoris element with a homogeneous field, which has an analytic solution.
    """

    s_start = 0
    s_end = 1
    length = s_end - s_start
    n_steps = 100

    # Field strength and orientation in the transverse plane
    B_0 = 1.0
    field_angle = np.pi / 7
    B_x = B_0 * np.cos(field_angle)
    B_y = B_0 * np.sin(field_angle)

    # Homogeneous transverse field coefficients on [s_start, s_end]
    # c1 = f(s0), c2 = f'(s0), c3 = f(s1), c4 = f'(s1), c5 = ∫ f(s) ds
    ks_0 = np.array([B_x, 0.0, B_x, 0.0, B_x * length])
    kn_0 = np.array([B_y, 0.0, B_y, 0.0, B_y * length])
    bs = np.zeros_like(ks_0)

    param_table = SplineParameterSchema.build_param_table_from_spline_coeffs(
        ks_0=ks_0,
        kn_0=kn_0,
        bs=bs,
        n_steps=n_steps,
    )

    splineboris = xt.SplineBoris.from_parameter_table(
        par_table=param_table,
        s_start=s_start,
        s_end=s_end,
        multipole_order=1,
        n_steps=n_steps,
    )

    # Reference and test particle
    line = xt.Line(elements=[splineboris])
    line.particle_ref = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1.0,
        energy0=1e9,
    )

    p = line.particle_ref.copy()
    p.x = 1e-3  # 1 mm offset
    p.px = 1e-3  # small transverse momentum to create a visible helix

    line.build_tracker()

    

def test_splineboris_solenoid_map():
    return

