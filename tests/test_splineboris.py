import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe

import xobjects as xo
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

from xtrack.beam_elements.spline_param_schema import SplineParameterSchema

"""
Tests for tracking without radiation with SplineBoris.
"""

def test_splineboris_homogeneous():
    """
    Test the SplineBoris element with a homogeneous field, which has an analytic solution.
    """

    s_start = 0
    s_end = 1
    length = s_end - s_start
    n_steps = 100

    # Field strength and orientation in the transverse plane
    B_0 = 0.1
    field_angle = np.pi / 2
    B_x = B_0 * np.cos(field_angle)
    B_y = B_0 * np.sin(field_angle)

    # Homogeneous transverse field coefficients on [s_start, s_end]
    # c1 = f(s0), c2 = f'(s0), c3 = f(s1), c4 = f'(s1), c5 = ∫ f(s) ds
    ks_0 = np.array([B_x, 0.0, B_x, 0.0, B_x * length])
    kn_0 = np.array([B_y, 0.0, B_y, 0.0, B_y * length])
    bs = np.zeros_like(ks_0)

    # Test that the field is constant (homogeneous) over the region
    # Use _poly in fieldfitter to plot the graph on [s_start, s_end]
    import sys
    from pathlib import Path

    # Add examples directory to path to import FieldFitter
    examples_path = Path(__file__).parent.parent / "examples" / "splineboris" / "spline_fitter"
    if str(examples_path) not in sys.path:
        sys.path.insert(0, str(examples_path))
    from field_fitter import FieldFitter

    s_vals = np.linspace(s_start, s_end, 200)
    # _poly takes (s0, s1, coeffs) and returns a polynomial that can be evaluated
    Bx_poly_obj = FieldFitter._poly(s_start, s_end, ks_0)
    By_poly_obj = FieldFitter._poly(s_start, s_end, kn_0)
    Bx_poly = Bx_poly_obj(s_vals)
    By_poly = By_poly_obj(s_vals)

    # import matplotlib.pyplot as plt

    # plt.plot(s_vals, Bx_poly, label='Bx_poly')
    # plt.plot(s_vals, By_poly, label='By_poly')
    # plt.legend()
    # plt.show()

    # Assert that the field is constant (homogeneous) over the region
    # This validates that the polynomial representation correctly represents a constant field
    np.testing.assert_allclose(Bx_poly, B_x, rtol=1e-12, atol=1e-12,
                                err_msg="B_x field should be constant (homogeneous)")
    np.testing.assert_allclose(By_poly, B_y, rtol=1e-12, atol=1e-12,
                                err_msg="B_y field should be constant (homogeneous)")

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
    
    # Calculate velocities from kinetic momenta
    # In xtrack: kin_px, kin_py, kin_ps are normalized by p0c
    # Actual momenta: p_x = kin_px * p0c (in eV/c)
    # Velocity: v_x = p_x / (γm) = (kin_px * p0c * qe / c) / (γ * mass0 * qe / c^2)
    #          = kin_px * p0c * c / (γ * mass0)
    gamma = p.gamma0[0]
    mass0_kg = p.mass0 * qe / clight**2  # Convert eV to kg
    p0c_SI = p.p0c[0] * qe / clight  # Convert eV/c to kg m/s
    
    # Velocity magnitude along s (longitudinal direction in original frame)
    vs = p.kin_ps[0] * p0c_SI / (gamma * mass0_kg)  # m/s
    
    # Transverse velocities in original frame (m/s)
    vx = p.kin_px[0] * p0c_SI / (gamma * mass0_kg)
    vy = p.kin_py[0] * p0c_SI / (gamma * mass0_kg)
    
    # Transform velocities to the rotated frame (where B points in y-direction)
    # The rotation applies to the transverse (x,y) plane
    vx_rot, vy_rot = R @ np.array([vx, vy])
    
    # Cyclotron frequency: ω_c = qB / (γm) [rad/s]
    # mass0 is in eV, convert to kg: mass_kg = mass0 * qe / c^2
    # So: ω_c = qe * B_0 / (gamma * (mass0 * qe / c^2)) = B_0 * c^2 / (gamma * mass0)
    q_C = abs(p.q0) * qe                 # Coulomb
    mc2_J = p.mass0 * qe                 # Joule (since mass0 is eV = mc^2 in eV)
    ps_SI  = p.kin_ps[0] * p0c_SI
    px_SI  = p.kin_px[0] * p0c_SI
    # with B along y, p_perp is in the x-s plane:
    pperp_SI = np.sqrt(px_SI**2 + ps_SI**2)

    p0_SI = p.p0c[0] * qe / clight   # (eV/c) -> kg m/s

    px_SI = p.kin_px[0] * p0_SI
    ps_SI = p.kin_ps[0] * p0_SI

    # In the rotated frame where B || y, p_perp is in (x,s)
    p_perp_SI = np.sqrt(px_SI**2 + ps_SI**2)
    rho = p_perp_SI / (q_C * B_0)

    assert rho > (s_end - s_start) * 2 / np.pi

    sign = 1
    
    # Center coordinates according to equations (1.26) and (1.27)
    # In the rotated frame (where B points in y-direction):
    # - x_new: coordinate perpendicular to B (this is "x" in the manual)
    # - z coordinate: along the direction of motion (this is "z" in the manual, corresponds to s)
    # - y_B: coordinate along B direction (this is "y" in the manual)
    # 
    # x'_0 = v_x,0 / v_z,0 = xp_new (already calculated, this is dx/dz)
    # x_c = x_0 ∓ ρ / sqrt(1 + x'_0^2)  [equation 1.26]
    # z_c = z_0 ± ρ * x'_0 / sqrt(1 + x'_0^2)  [equation 1.27]
    # 
    # The initial z position is s_start (path length coordinate)
    z_0 = s_start  # Initial z position along motion direction
    s_end = s_end
    
    # --- Drop-in: choose the correct (x_c, s_c) and the correct x(s) branch non-circularly ---

    sqrt_term = np.sqrt(1.0 + xp_rot**2)

    # Two candidate centers (from your analytic formulas)
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
        print("plus is better")
        x_c, s_c = x_c_plus, s_c_plus
    else:
        print("minus is better")
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
    
    print(f"Analytic solution at s_end:")
    print(f"  x_final = {x_final:.6e} m")
    print(f"  y_final = {y_final:.6e} m")
    print(f"  xp_final = {xp_final:.6e}")
    print(f"  yp_final = {yp_final:.6e}")
    
    # Track the particle with SplineBoris
    line.track(p)
    x_end_splineboris = p.x[0]
    y_end_splineboris = p.y[0]
    xp_final_splineboris = p.kin_xprime[0]
    yp_final_splineboris = p.kin_yprime[0]
    print(f"SplineBoris solution at s_end:")
    print(f"  x_final = {x_end_splineboris:.6e} m")
    print(f"  y_final = {y_end_splineboris:.6e} m")
    print(f"  xp_final = {xp_final_splineboris:.6e}")
    print(f"  yp_final = {yp_final_splineboris:.6e}")

    # Print ratio
    print(f"  x_final / x_end_splineboris = {x_final / x_end_splineboris:.6e}")
    print(f"  y_final / y_end_splineboris = {y_final / y_end_splineboris:.6e}")
    print(f"  xp_final / xp_final_splineboris = {xp_final / xp_final_splineboris:.6e}")
    print(f"  yp_final / yp_final_splineboris = {yp_final / yp_final_splineboris:.6e}")

    rho_analytic_est = length / abs(xp_end_rot)
    rho_spline_est   = length / abs(xp_final_splineboris)

    print(f"rho_est (analytic) = {rho_analytic_est:.6e}")
    print(f"rho_est (spline)   = {rho_spline_est:.6e}")
    print(f"ratio rho_spline/rho_analytic = {rho_spline_est / rho_analytic_est:.6e}")

    L = length
    h_spline = abs(xp_final_splineboris - kin_xp) / L  # better than abs(xp) if kin_xp not zero

    p0_SI = p.p0c[0] * qe / clight
    px_SI = p.kin_px[0] * p0_SI
    ps_SI = p.kin_ps[0] * p0_SI
    p_perp_SI = np.sqrt(px_SI**2 + ps_SI**2)

    B_eff_spline = h_spline * p_perp_SI / (abs(p.q0) * qe)
    print("B_eff_spline =", B_eff_spline, "T")
    print("B_eff_spline/B0 =", B_eff_spline / B_0)

    assert np.allclose(x_final, x_end_splineboris, atol=1e-12)
    assert np.allclose(y_final, y_end_splineboris, atol=1e-12)
    assert np.allclose(xp_final, xp_final_splineboris, atol=1e-12)
    assert np.allclose(yp_final, yp_final_splineboris, atol=1e-12)
    line.build_tracker()

test_splineboris_homogeneous()

def test_splineboris_solenoid_map():
    return

"""
Tests for spin tracking with SplineBoris.
"""

"""
Tests for tracking with radiation with SplineBoris.
"""