import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe
import pytest
import xobjects as xo
import pandas as pd
from pathlib import Path
import sys

import xtrack as xt
from xtrack.beam_elements.spline_param_schema import (
    SplineParameterSchema,
    build_parameter_table_from_df,
)

# Make the auto-generated spline field evaluator importable without requiring
# an installed xtrack package.
elements_src_path = (
    Path(__file__).parent.parent / "xtrack" / "beam_elements" / "elements_src"
)
if str(elements_src_path) not in sys.path:
    sys.path.insert(0, str(elements_src_path))
from _spline_B_field_eval_python import evaluate_B

# Test some common field angles, as well as some unusual ones
@pytest.mark.parametrize('field_angle', [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 4*np.pi/9, np.pi/7])
def test_splineboris_homogeneous_analytic(field_angle):
    """
    Test the SplineBoris element with a homogeneous field, which has an analytic solution.
    Knowing the angle of the field, we can rotate our coordinates such that the field points in the y-direction.
    We then use helix geometry to calculate the end-point/angle of the particle.
    We then rotate back to the original coordinates and check this solution against the SplineBoris.
    """

    s_start = 0
    s_end = 1
    length = s_end - s_start
    n_steps = 100

    # Field strength and orientation in the transverse plane
    B_0 = 0.1
    B_x = B_0 * np.cos(field_angle)
    B_y = B_0 * np.sin(field_angle)

    # Homogeneous transverse field coefficients on [s_start, s_end]
    # c1 = f(s0), c2 = f'(s0), c3 = f(s1), c4 = f'(s1), c5 = ∫ f(s) ds
    Bx_0_coeffs = np.array([B_x, 0.0, B_x, 0.0, B_x * length])
    By_0_coeffs = np.array([B_y, 0.0, B_y, 0.0, B_y * length])
    Bs_coeffs = np.zeros_like(Bx_0_coeffs)

    import sys
    from pathlib import Path

    # Add examples directory to path to import FieldFitter
    examples_path = Path(__file__).parent.parent / "examples" / "splineboris" / "spline_fitter"
    if str(examples_path) not in sys.path:
        sys.path.insert(0, str(examples_path))
    from field_fitter import FieldFitter

    # Convert to the basis that the field evaluator uses.
    Bx_poly = FieldFitter._poly(s_start, s_end, Bx_0_coeffs)
    By_poly = FieldFitter._poly(s_start, s_end, By_0_coeffs)
    Bs_poly = FieldFitter._poly(s_start, s_end, Bs_coeffs)

    Bx_values = Bx_poly(np.linspace(s_start, s_end, 100))
    By_values = By_poly(np.linspace(s_start, s_end, 100))

    degree = 4

    ks_0 = np.zeros(degree + 1)
    ks_0[:len(Bx_poly.coef)] = Bx_poly.coef
    kn_0 = np.zeros(degree + 1)
    kn_0[:len(By_poly.coef)] = By_poly.coef
    bs = np.zeros(degree + 1)
    bs[:len(Bs_poly.coef)] = Bs_poly.coef

    # Assert that the field is constant (homogeneous) over the region
    # This validates that the polynomial representation correctly represents a constant field
    np.testing.assert_allclose(Bx_values, B_x, rtol=1e-12, atol=1e-12,
                                err_msg="B_x field should be constant (homogeneous)")
    np.testing.assert_allclose(By_values, B_y, rtol=1e-12, atol=1e-12,
                                err_msg="B_y field should be constant (homogeneous)")

    param_table = SplineParameterSchema.build_param_table_from_spline_coeffs(
        ks_0=ks_0,
        kn_0=kn_0,
        bs=bs,
        n_steps=n_steps,
    )

    splineboris = xt.SplineBoris(
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

    # Analytic solution for the helix angle
    kin_xp = p.kin_xprime[0]
    kin_yp = p.kin_yprime[0]
    x = p.x[0]
    y = p.y[0]

    # Transform the coordinates to a frame where the field points in the y-direction:
    # In this frame, x' = dx/dz (where z is along the field direction)
    R = np.array([[np.sin(field_angle), -np.cos(field_angle)],
                  [np.cos(field_angle), np.sin(field_angle)]])
    x_rot, y_rot = R @ np.array([x, y])
    xp_rot, yp_rot = R @ np.array([kin_xp, kin_yp])
    
    q_C = abs(p.q0) * qe  # Coulomb
    p0_SI = p.p0c[0] * qe / clight  # (eV/c) -> kg m/s
    px_SI = p.kin_px[0] * p0_SI
    ps_SI = p.kin_ps[0] * p0_SI

    # In the rotated frame where B || y, p_perp is in (x,s)
    p_perp_SI = np.sqrt(px_SI**2 + ps_SI**2)
    rho = p_perp_SI / (q_C * B_0)

    assert rho > (s_end - s_start) * 2 / np.pi


    sqrt_term = np.sqrt(1.0 + xp_rot**2)

    # Two candidate centers (from helix geometry):
    # x_c = x0 ∓ rho/sqrt(1+xp0^2)
    # s_c = s0 ± rho*xp0/sqrt(1+xp0^2)
    x_c_plus  = x_rot - rho / sqrt_term
    s_c_plus  = s_start + rho * xp_rot / sqrt_term

    x_c_minus = x_rot + rho / sqrt_term
    s_c_minus = s_start - rho * xp_rot / sqrt_term

    def xp_from_center(xc, sc, x0, s0):
        # x' = dx/ds = -(s - s_c)/(x - x_c)
        return -(s0 - sc) / (x0 - xc)

    # Pick the center whose implied slope matches xp_rot best
    xp_pred_plus  = xp_from_center(x_c_plus,  s_c_plus,  x_rot, s_start)
    xp_pred_minus = xp_from_center(x_c_minus, s_c_minus, x_rot, s_start)

    if abs(xp_pred_plus - xp_rot) <= abs(xp_pred_minus - xp_rot):
        x_c, s_c = x_c_plus, s_c_plus
    else:
        x_c, s_c = x_c_minus, s_c_minus

    # Determine the correct branch for x(s) from the initial point (NOT from the center sign)
    x0_diff = x_rot - x_c
    s0_diff = s_start - s_c

    # Guard numerical noise
    rad0 = rho**2 - s0_diff**2
    if rad0 < -1e-15 * rho**2:
        raise ValueError(f"Initial point not on circle: rho^2-(s0-s_c)^2 = {rad0}")
    rad0 = max(0.0, rad0)
    sqrt0 = np.sqrt(rad0)

    sigma_branch = np.sign(x0_diff)
    if sigma_branch == 0:
        sigma_branch = 1.0

    # Sanity: reconstruct x0
    x0_recon = x_c + sigma_branch * sqrt0
    if abs(x0_recon - x_rot) > 1e-12:
        raise ValueError(
            f"Branch selection failed: x0_recon={x0_recon}, x0={x_rot}, "
            f"diff={x0_recon - x_rot}"
        )

    # Now compute end-point x(s_end), x'(s_end) consistently on the same branch
    s_diff = s_end - s_c
    rad_end = rho**2 - s_diff**2
    if rad_end < -1e-15 * rho**2:
        raise ValueError(f"s_end outside circle: rho^2-(s_end-s_c)^2 = {rad_end}")
    rad_end = max(0.0, rad_end)
    sqrt_end = np.sqrt(rad_end)

    x_end_rot = x_c + sigma_branch * sqrt_end
    x_diff = x_end_rot - x_c

    # x' = -(s-s_c)/(x-x_c)
    xp_end_rot = -s_diff / x_diff

    # --- Optional: robust phase change (wrapped to [-pi, pi]) ---
    initial_phase = np.arctan2(s0_diff, x0_diff)
    final_phase   = np.arctan2(s_diff,  x_diff)
    phase_change  = np.arctan2(np.sin(final_phase - initial_phase),
                            np.cos(final_phase - initial_phase))

    # y_end_rot and yp_end_rot (your existing formulas)
    y_end_rot  = y_rot + yp_rot * x0_diff * phase_change
    yp_end_rot = yp_rot * x0_diff / x_diff

    
    # Transform back to original (x, y) coordinates
    R_inv = np.linalg.inv(R)  # Inverse rotation (transpose of orthogonal matrix)
    x_final, y_final = R_inv @ np.array([x_end_rot, y_end_rot])
    xp_final, yp_final = R_inv @ np.array([xp_end_rot, yp_end_rot])
    
    # Track the particle with SplineBoris
    line.track(p)
    x_end_splineboris = p.x[0]
    y_end_splineboris = p.y[0]
    xp_final_splineboris = p.kin_xprime[0]
    yp_final_splineboris = p.kin_yprime[0]

    assert np.allclose(x_final, x_end_splineboris, atol=1e-12)
    assert np.allclose(y_final, y_end_splineboris, atol=1e-12)
    assert np.allclose(xp_final, xp_final_splineboris, atol=1e-12)
    assert np.allclose(yp_final, yp_final_splineboris, atol=1e-12)



# Test some common field angles, as well as some unusual ones
@pytest.mark.parametrize('field_angle', [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 4*np.pi/9, np.pi/7])
def test_splineboris_homogeneous_rbend(field_angle):
    """
    Test the SplineBoris element with a homogeneous field against the RBend.
    """

    s_start = 0
    s_end = 1
    length = s_end - s_start
    n_steps = 100

    # Field strength and orientation in the transverse plane
    B_0 = 0.1
    B_x = B_0 * np.cos(field_angle)
    B_y = B_0 * np.sin(field_angle)

    # Homogeneous transverse field coefficients on [s_start, s_end]
    # c1 = f(s0), c2 = f'(s0), c3 = f(s1), c4 = f'(s1), c5 = ∫ f(s) ds
    Bx_0_coeffs = np.array([B_x, 0.0, B_x, 0.0, B_x * length])
    By_0_coeffs = np.array([B_y, 0.0, B_y, 0.0, B_y * length])
    Bs_coeffs = np.zeros_like(Bx_0_coeffs)

    import sys
    from pathlib import Path

    # Add examples directory to path to import FieldFitter
    examples_path = Path(__file__).parent.parent / "examples" / "splineboris" / "spline_fitter"
    if str(examples_path) not in sys.path:
        sys.path.insert(0, str(examples_path))
    from field_fitter import FieldFitter

    # Convert to the basis that the field evaluator uses.
    Bx_poly = FieldFitter._poly(s_start, s_end, Bx_0_coeffs)
    By_poly = FieldFitter._poly(s_start, s_end, By_0_coeffs)
    Bs_poly = FieldFitter._poly(s_start, s_end, Bs_coeffs)

    Bx_values = Bx_poly(np.linspace(s_start, s_end, 100))
    By_values = By_poly(np.linspace(s_start, s_end, 100))

    degree = 4

    ks_0 = np.zeros(degree + 1)
    ks_0[:len(Bx_poly.coef)] = Bx_poly.coef
    kn_0 = np.zeros(degree + 1)
    kn_0[:len(By_poly.coef)] = By_poly.coef
    bs = np.zeros(degree + 1)
    bs[:len(Bs_poly.coef)] = Bs_poly.coef

    # Assert that the field is constant (homogeneous) over the region
    # This validates that the polynomial representation correctly represents a constant field
    np.testing.assert_allclose(Bx_values, B_x, rtol=1e-12, atol=1e-12,
                                err_msg="B_x field should be constant (homogeneous)")
    np.testing.assert_allclose(By_values, B_y, rtol=1e-12, atol=1e-12,
                                err_msg="B_y field should be constant (homogeneous)")

    param_table = SplineParameterSchema.build_param_table_from_spline_coeffs(
        ks_0=ks_0,
        kn_0=kn_0,
        bs=bs,
        n_steps=n_steps,
    )

    splineboris = xt.SplineBoris(
        par_table=param_table,
        s_start=s_start,
        s_end=s_end,
        multipole_order=1,
        n_steps=n_steps,
    )

    # Reference and test particle
    line_splineboris = xt.Line(elements=[splineboris])
    line_splineboris.particle_ref = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1.0,
        energy0=1e9,
    )

    p_splineboris = line_splineboris.particle_ref.copy()
    p_splineboris.x = 1e-3
    p_splineboris.px = 1e-3

    p_rbend = p_splineboris.copy()
    p_rbend.x = 1e-3
    p_rbend.px = 1e-3

    k0 =  B_0 * clight / p_rbend.p0c[0]

    edge_model = 'suppressed'       # Ignore the edge effects
    rot_s_rad = field_angle-np.pi/2 # For field_angle = 0, the field is in the x-direction. To have the bend reflect this, we need to rotate the coordinate system in the opposite direction.
    b_rbend = xt.RBend(k0=k0, k0_from_h=False, length_straight=length, angle=0, edge_entry_angle=0, edge_exit_angle=0, rot_s_rad=rot_s_rad)
    b_rbend.edge_entry_model = edge_model
    b_rbend.edge_exit_model = edge_model
    b_rbend.model = 'bend-kick-bend'

    line_rbend = xt.Line(elements=[b_rbend])
    line_rbend.particle_ref = p_rbend

    line_rbend.track(p_rbend)
    x_end_rbend = p_rbend.x[0]
    y_end_rbend = p_rbend.y[0]
    px_end_rbend = p_rbend.kin_px[0]
    py_end_rbend = p_rbend.kin_py[0]

    # Track the particle with SplineBoris
    line_splineboris.track(p_splineboris)
    x_end_splineboris = p_splineboris.x[0]
    y_end_splineboris = p_splineboris.y[0]
    px_final_splineboris = p_splineboris.kin_px[0]
    py_final_splineboris = p_splineboris.kin_py[0]

    assert np.allclose(x_end_rbend, x_end_splineboris, atol=1e-12)
    assert np.allclose(y_end_rbend, y_end_splineboris, atol=1e-12)
    assert np.allclose(px_end_rbend, px_final_splineboris, atol=1e-12)
    assert np.allclose(py_end_rbend, py_final_splineboris, atol=1e-12)



def test_splineboris_undulator_vs_boris_spatial():
    """
    Build a lightweight undulator from spline-fit parameters and check that
    tracking with SplineBoris and BorisSpatialIntegrator gives consistent
    end coordinates.
    """

    # ------------------------------------------------------------------
    # Load fit parameters and build a reduced parameter table
    # ------------------------------------------------------------------
    base_dir = (
        Path(__file__).parent.parent
        / "examples"
        / "splineboris"
        / "spline_fitter"
        / "field_maps"
    )
    filepath = base_dir / "field_fit_pars.csv"

    df = pd.read_csv(
        filepath,
        index_col=[
            "field_component",
            "derivative_x",
            "region_name",
            "s_start",
            "s_end",
            "idx_start",
            "idx_end",
            "param_index",
        ],
    )

    multipole_order = 3
    n_steps_test = 1000

    par_table, s_start, s_end = build_parameter_table_from_df(
        df_fit_pars=df,
        n_steps=n_steps_test,
        multipole_order=multipole_order,
    )

    s_vals = np.linspace(s_start, s_end, n_steps_test)
    ds = (s_end - s_start) / n_steps_test

    n_seg = n_steps_test

    # ------------------------------------------------------------------
    # Build a short undulator line using SplineBoris slices
    # ------------------------------------------------------------------
    spline_elems = []
    for i in range(n_seg):
        params_i = [par_table[i].tolist()]  # shape (1, n_params) for n_steps=1
        s_val_i = s_vals[i]

        elem_s_start = s_val_i - ds / 2
        elem_s_end = s_val_i + ds / 2

        spline_elems.append(
            xt.SplineBoris(
                par_table=params_i,
                multipole_order=multipole_order,
                s_start=elem_s_start,
                s_end=elem_s_end,
                n_steps=1,
            )
        )

    line_spline = xt.Line(elements=spline_elems)

    # This undulator is part of the SLS, so we use the nominal energy of the SLS.
    p_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)
    line_spline.particle_ref = p_ref.copy()

    p_spline = line_spline.particle_ref.copy()
    p_spline.x = 1e-3
    p_spline.px = 1e-4
    p_spline.y = 0.5e-3
    p_spline.py = -0.5e-4

    line_spline.track(p_spline)

    # ------------------------------------------------------------------
    # Wrap the spline-based field evaluator into (x, y, z) callables
    # ------------------------------------------------------------------
    def make_segment_field(params_1d, multipole_order_local):
        params_arr = np.asarray(params_1d, dtype=float)

        def field(x, y, z):
            Bx, By, Bs = evaluate_B(x, y, z, params_arr, multipole_order_local)
            return Bx, By, Bs

        return field

    # ------------------------------------------------------------------
    # Build a parallel undulator line using BorisSpatialIntegrator
    # ------------------------------------------------------------------
    boris_elems = []
    for i in range(n_seg):
        s_val_i = s_vals[i]
        elem_s_start = s_val_i - ds / 2
        elem_s_end = s_val_i + ds / 2

        field_i = make_segment_field(par_table[i], multipole_order)

        boris_elems.append(
            xt.BorisSpatialIntegrator(
                fieldmap_callable=field_i,
                s_start=elem_s_start,
                s_end=elem_s_end,
                n_steps=1,
            )
        )

    line_boris = xt.Line(elements=boris_elems)
    line_boris.particle_ref = p_ref.copy()

    p_boris = line_boris.particle_ref.copy()
    p_boris.x = 1e-3
    p_boris.px = 1e-4
    p_boris.y = 0.5e-3
    p_boris.py = -0.5e-4

    line_boris.track(p_boris)

    # ------------------------------------------------------------------
    # Compare end coordinates
    # ------------------------------------------------------------------
    # The two integrators are not bitwise identical; use tight but robust tolerances.
    xo.assert_allclose(p_spline.x, p_boris.x, rtol=1e-7, atol=5e-7)
    xo.assert_allclose(p_spline.px, p_boris.px, rtol=1e-7, atol=5e-7)
    xo.assert_allclose(p_spline.y, p_boris.y, rtol=1e-7, atol=5e-7)
    xo.assert_allclose(p_spline.py, p_boris.py, rtol=1e-7, atol=5e-7)
    xo.assert_allclose(p_spline.zeta, p_boris.zeta, rtol=1e-7, atol=5e-7)
    xo.assert_allclose(p_spline.delta, p_boris.delta, rtol=1e-7, atol=5e-7)

test_splineboris_undulator_vs_boris_spatial()

# def test_splineboris_quadrupole_analytic():
#     import numpy as np
#     from scipy import special, optimize

#     def duffing_cn_solution(t, x0, v0, p, q):
#         # Energy from ICs
#         E = 0.5*v0**2 + 0.5*p*x0**2 + 0.25*q*x0**4

#         # Amplitude A from turning-point energy
#         A2 = (-p + np.sqrt(p**2 + 4*q*E)) / q
#         A = np.sqrt(A2)

#         # Frequency and parameter m
#         Omega2 = p + q*A2
#         Omega = np.sqrt(Omega2)
#         m = (q*A2) / (2*Omega2)

#         # Phase u0 via incomplete elliptic integral F(phi|m)
#         phi0 = np.arccos(np.clip(x0 / A, -1.0, 1.0))
#         u0 = special.ellipkinc(phi0, m)

#         # Fix branch to match v0
#         sn, cn, dn, _ = special.ellipj(u0, m)
#         v0_model = -A * Omega * sn * dn
#         if np.sign(v0_model) != np.sign(v0) and abs(v0) > 0:
#             u0 = -u0

#         # Evaluate x(t)
#         u = Omega*t + u0
#         sn, cn, dn, _ = special.ellipj(u, m)
#         x = A * cn
#         xdot = -A * Omega * sn * dn
#         return x, xdot, (A, Omega, m, u0)

#     def z_of_t(t, z0, C, k1, A, Omega, m, u0):
#         u = Omega*t + u0
#         # amplitude ph = am(u|m)
#         sn, cn, dn, ph = special.ellipj(u, m)
#         sn0, cn0, dn0, ph0 = special.ellipj(u0, m)

#         eps  = special.ellipeinc(ph,  m)   # ε(u|m) = E(am(u)|m)
#         eps0 = special.ellipeinc(ph0, m)

#         return (z0 + C*t + 0.5 * (k1*A*A) / (Omega*m) * ((eps - eps0) + (m-1)*(u - u0)))


#     def x_of_z(z_target, t_lo, t_hi, x0, v0, p, q, z0, C, k1, A, Omega, m, u0):
#         """
#         Find the time t such that z_of_t(t, ...) = z_target.
        
#         To know beforehand if f(a) and f(b) have different signs:
#         1. Evaluate f(a) = z_of_t(a, ...) - z_target
#         2. Evaluate f(b) = z_of_t(b, ...) - z_target  
#         3. Check: np.sign(f(a)) != np.sign(f(b))
        
#         If signs are the same, the root is outside [a, b] and we need to expand the bracket.
#         Since z(t) is monotone, we can expand the interval until we find opposite signs.
#         """
#         f = lambda t: z_of_t(t, z0, C, k1, A, Omega, m, u0) - z_target
        
#         # Check if f(t_lo) and f(t_hi) have opposite signs (required for brentq)
#         f_lo = f(t_lo)
#         f_hi = f(t_hi)
        
#         # If signs are the same, try to find a valid bracket by expanding the interval
#         if np.sign(f_lo) == np.sign(f_hi):
#             # Try expanding the interval
#             t_range = t_hi - t_lo
#             max_expansions = 10
#             for i in range(max_expansions):
#                 # Expand symmetrically
#                 t_lo_expanded = t_lo - t_range * (i + 1)
#                 t_hi_expanded = t_hi + t_range * (i + 1)
#                 f_lo_expanded = f(t_lo_expanded)
#                 f_hi_expanded = f(t_hi_expanded)
#                 if np.sign(f_lo_expanded) != np.sign(f_hi_expanded):
#                     t_lo, t_hi = t_lo_expanded, t_hi_expanded
#                     break
#             else:
#                 # If we couldn't find a bracket, use secant method which doesn't require bracketing
#                 result = optimize.root_scalar(f, x0=t_lo, x1=t_hi, method='secant')
#                 if not result.converged:
#                     raise ValueError(f"Could not find root: f({t_lo})={f_lo}, f({t_hi})={f_hi}")
#                 t = result.root
#         else:
#             # Signs are opposite, can use brentq directly
#             t = optimize.brentq(f, t_lo, t_hi)   # requires z(t) monotone over [t_lo,t_hi]
        
#         return duffing_cn_solution(t, x0, v0, p, q), t

#     p = xt.Particles(
#         mass0=xt.ELECTRON_MASS_EV,
#         q0=1.0,
#         energy0=1e9,
#     )
#     p.x = 1e-3
#     p.px = 1e-3

#     g = 1000        # Gradient of the quadrupole field in T/m
#     length = 2      # Length of the quadrupole in meters
#     q_C = abs(p.q0) * qe  # Coulomb
#     p0_SI = float(p.p0c[0]) * qe / clight  # (eV/c) -> kg m/s
#     px_SI = float(p.kin_px[0]) * p0_SI
#     ps_SI = float(p.kin_ps[0]) * p0_SI

#     # In the rotated frame where B || y, p_perp is in (x,s)
#     p_perp_SI = np.sqrt(px_SI**2 + ps_SI**2)
#     gamma0_scalar = float(p.gamma0[0])
#     mass0_scalar = float(p.mass0)
#     g_norm = g / gamma0_scalar / mass0_scalar

#     vx_0 = px_SI / gamma0_scalar / mass0_scalar
#     vs_0 = ps_SI / gamma0_scalar / mass0_scalar

#     x0_scalar = float(p.x[0])  # Save initial x value before tracking
#     C_0 = vs_0 - 1/2 * g_norm * x0_scalar**2

#     s_start = 0
#     s_end = length
#     n_steps = 100

#     # Homogeneous transverse field coefficients on [s_start, s_end]
#     # c1 = f(s0), c2 = f'(s0), c3 = f(s1), c4 = f'(s1), c5 = ∫ f(s) ds
#     Bx_0_coeffs = np.array([0, 0.0, 0, 0.0, 0])
#     By_0_coeffs = np.array([0, 0.0, 0, 0.0, 0])
#     Bx_1_coeffs = np.array([g, 0.0, g, 0.0, g * length])
#     By_1_coeffs = np.array([0, 0.0, 0, 0.0, 0])
#     Bs_coeffs = np.zeros_like(Bx_0_coeffs)

#     import sys
#     from pathlib import Path

#     # Add examples directory to path to import FieldFitter
#     examples_path = Path(__file__).parent.parent / "examples" / "splineboris" / "spline_fitter"
#     if str(examples_path) not in sys.path:
#         sys.path.insert(0, str(examples_path))
#     from field_fitter import FieldFitter

#     # Convert to the basis that the field evaluator uses.
#     Bx_0_poly = FieldFitter._poly(s_start, s_end, Bx_0_coeffs)
#     By_0_poly = FieldFitter._poly(s_start, s_end, By_0_coeffs)
#     Bx_1_poly = FieldFitter._poly(s_start, s_end, Bx_1_coeffs)
#     By_1_poly = FieldFitter._poly(s_start, s_end, By_1_coeffs)
#     Bs_poly = FieldFitter._poly(s_start, s_end, Bs_coeffs)

#     degree = 4

#     ks_0 = np.zeros(degree + 1)
#     ks_0[:len(Bx_0_poly.coef)] = Bx_0_poly.coef
#     kn_0 = np.zeros(degree + 1)
#     kn_0[:len(By_0_poly.coef)] = By_0_poly.coef
#     ks_1 = np.zeros(degree + 1)
#     ks_1[:len(Bx_1_poly.coef)] = Bx_1_poly.coef
#     kn_1 = np.zeros(degree + 1)
#     kn_1[:len(By_1_poly.coef)] = By_1_poly.coef
#     bs = np.zeros(degree + 1)
#     bs[:len(Bs_poly.coef)] = Bs_poly.coef

#     param_table = SplineParameterSchema.build_param_table_from_spline_coeffs(
#         ks_0=ks_0,
#         kn_0=kn_0,
#         ks_1=ks_1,
#         kn_1=kn_1,
#         bs=bs,
#         n_steps=n_steps,
#     )

#     splineboris = xt.SplineBoris(
#         par_table=param_table,
#         s_start=s_start,
#         s_end=s_end,
#         multipole_order=1,
#         n_steps=n_steps,
#     )

#     line_splineboris = xt.Line(elements=[splineboris])
#     line_splineboris.particle_ref = p

#     line_splineboris.track(p)
#     x_end_splineboris = float(p.x[0])
#     _, _, (A_0, Omega_0, m_0, u0_0) = duffing_cn_solution(0, x0_scalar, vx_0, g_norm * C_0, g_norm/2)
#     # Convert tuple values to scalars
#     A_0 = float(A_0)
#     Omega_0 = float(Omega_0)
#     m_0 = float(m_0)
#     u0_0 = float(u0_0)

#     result, _ = x_of_z(s_end, -1e-7, 1e-7, x0_scalar, vx_0, g_norm * C_0, g_norm/2, s_start, C_0, g_norm, A_0, Omega_0, m_0, u0_0)
#     x_end_analytic = float(result[0])  # Extract x from (x, xdot, (A, Omega, m, u0))

#     assert np.allclose(x_end_analytic, x_end_splineboris, atol=1e-12)