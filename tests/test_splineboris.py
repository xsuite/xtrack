import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe
import pytest
import xobjects as xo
import pandas as pd
from pathlib import Path
import sys

import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

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

    # Convert to the basis that the field evaluator uses.
    Bx_poly = xt.SplineBoris.spline_poly(s_start, s_end, Bx_0_coeffs)
    By_poly = xt.SplineBoris.spline_poly(s_start, s_end, By_0_coeffs)
    Bs_poly = xt.SplineBoris.spline_poly(s_start, s_end, Bs_coeffs)

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

    param_table = xt.SplineBoris.build_param_table_from_spline_coeffs(
        bs=bs,
        kn={0: kn_0},
        ks={0: ks_0},
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

    # Convert to the basis that the field evaluator uses.
    Bx_poly = xt.SplineBoris.spline_poly(s_start, s_end, Bx_0_coeffs)
    By_poly = xt.SplineBoris.spline_poly(s_start, s_end, By_0_coeffs)
    Bs_poly = xt.SplineBoris.spline_poly(s_start, s_end, Bs_coeffs)

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

    param_table = xt.SplineBoris.build_param_table_from_spline_coeffs(
        bs=bs,
        kn={0: kn_0},
        ks={0: ks_0},
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



# @pytest.mark.parametrize(
#     'case,atol',
#     zip(
#         [case['case'].copy() for case in COMMON_TEST_CASES],
#         [3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 3e-8, 2e-5, 1e-5, 2e-8, 1e-5, 2e-5],
#     ),
#     ids=[case['id'] for case in COMMON_TEST_CASES],
# )
def test_uniform_solenoid():

    atol = 3e-8
    case = {
    'x': 0.001,
    'px': 1e-05,
    'y': 0.002,
    'py': 2e-05,
    'delta': 0.001,
    'spin_x': 0.1,
    'spin_z': 0.2,
    }
    case['spin_y'] = np.sqrt(1 - case['spin_x']**2 - case['spin_z']**2)

    

    p = xt.Particles(
        p0c=700e9, mass0=xt.ELECTRON_MASS_EV,
        anomalous_magnetic_moment=0.00115965218128,
        **case,
    )

    p_splineboris = p.copy()
    p_ref = p.copy()

    Bz_T = 0.05
    ks = Bz_T / (p.p0c[0] / clight / p.q0)
    env = xt.Environment()

    length = 0.25
    s_start = 0
    s_end = length
    n_steps = 100

        # Homogeneous transverse field coefficients on [s_start, s_end]
    # c1 = f(s0), c2 = f'(s0), c3 = f(s1), c4 = f'(s1), c5 = ∫ f(s) ds
    Bx_0_coeffs = np.array([0, 0.0, 0.0, 0.0, 0])
    By_0_coeffs = np.array([0, 0.0, 0.0, 0.0, 0])
    Bs_coeffs = np.array([Bz_T, 0.0, Bz_T, 0.0, Bz_T * length])

    # Convert to the basis that the field evaluator uses.
    Bx_poly = xt.SplineBoris.spline_poly(s_start, s_end, Bx_0_coeffs)
    By_poly = xt.SplineBoris.spline_poly(s_start, s_end, By_0_coeffs)
    Bs_poly = xt.SplineBoris.spline_poly(s_start, s_end, Bs_coeffs)

    Bs_values = Bs_poly(np.linspace(s_start, s_end, 100))


    degree = 4

    ks_0 = np.zeros(degree + 1)
    ks_0[:len(Bx_poly.coef)] = Bx_poly.coef
    kn_0 = np.zeros(degree + 1)
    kn_0[:len(By_poly.coef)] = By_poly.coef
    bs = np.zeros(degree + 1)
    bs[:len(Bs_poly.coef)] = Bs_poly.coef

    # Assert that the field is constant (homogeneous) over the region
    # This validates that the polynomial representation correctly represents a constant field
    np.testing.assert_allclose(Bs_values, Bz_T, rtol=1e-12, atol=1e-12,
                                err_msg="Bs field should be constant (homogeneous)")

    param_table = xt.SplineBoris.build_param_table_from_spline_coeffs(
        bs=bs,
        kn={0: kn_0},
        ks={0: ks_0},
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
    line_splineboris.particle_ref = p_splineboris


    line = env.new_line(
        components=[
            env.new('mysolenoid', xt.UniformSolenoid, length=length, ks=ks),
            env.new('mymarker', xt.Marker),
        ]
    )

    line.configure_spin(spin_model='auto')
    line_splineboris.configure_spin(spin_model='auto')

    line.track(p_ref)
    line_splineboris.track(p_splineboris)

    # print(p_ref.spin_x[0], p_splineboris.spin_x[0])
    # print(p_ref.spin_y[0], p_splineboris.spin_y[0])
    # print(p_ref.spin_z[0], p_splineboris.spin_z[0])

    # xo.assert_allclose(p_ref.spin_x[0], p_splineboris.spin_x[0], atol=atol, rtol=1e-12)
    # xo.assert_allclose(p_ref.spin_y[0], p_splineboris.spin_y[0], atol=atol, rtol=1e-12)
    # xo.assert_allclose(p_ref.spin_z[0], p_splineboris.spin_z[0], atol=atol, rtol=1e-12)

    print(p_ref.s, p_splineboris.s)
    print(p_ref.x, p_splineboris.x)
    print(p_ref.y, p_splineboris.y)
    print(p_ref.px, p_splineboris.px)
    print(p_ref.py, p_splineboris.py)
    print(p_ref.delta, p_splineboris.delta)

    xo.assert_allclose(p_ref.s, p_splineboris.s, atol=atol, rtol=1e-12)
    xo.assert_allclose(p_ref.x, p_splineboris.x, atol=atol, rtol=1e-12)
    xo.assert_allclose(p_ref.y, p_splineboris.y, atol=atol, rtol=1e-12)
    xo.assert_allclose(p_ref.px, p_splineboris.px, atol=atol, rtol=1e-12)
    xo.assert_allclose(p_ref.py, p_splineboris.py, atol=atol, rtol=1e-12)
    xo.assert_allclose(p_ref.delta, p_splineboris.delta, atol=atol, rtol=1e-12)




def test_splineboris_solenoid_vs_variable_solenoid():
    """
    Test SplineBoris element against VariableSolenoid for a solenoid field.

    This test creates a solenoid field, fits it with polynomial splines, tracks particles
    through both SplineBoris and VariableSolenoid, and compares the trajectories.
    """
    # Set basic parameters
    interval = 30
    multipole_order = 4
    n_steps = 5000

    # Make initial particles
    delta = np.array([0, 4])
    p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                    energy0=45.6e6,  # 45.6 MeV
                    x=1e-3,  # Start slightly off-axis
                    px=-1e-3*(1+delta),
                    y=1e-3,
                    delta=delta)

    # Make solenoid field instance
    sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=20)

    # # Prepare test data directory
    test_data_dir = Path(__file__).parent.parent / "test_data" / "splineboris"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    fit_pars_path = test_data_dir / "test_solenoid_vs_varsol_fit_pars.csv"

    # NOTE: If the fit parameters need to be updated, uncomment the following code.

    # # Add examples directory to path to import FieldFitter and StandardFieldMapParser
    # examples_path = Path(__file__).parent.parent / "examples" / "splineboris" / "spline_fitter"
    # if str(examples_path) not in sys.path:
    #     sys.path.insert(0, str(examples_path))
    # from field_fitter import FieldFitter
    # from fieldmap_parsers import StandardFieldMapParser

    # # Construct field map
    # dx = 0.001
    # dy = 0.001
    # x_axis = np.linspace(-multipole_order*dx/2, multipole_order*dx/2, multipole_order+1)
    # y_axis = np.linspace(-multipole_order*dy/2, multipole_order*dy/2, multipole_order+1)
    # z_axis = np.linspace(0, interval, n_steps+1)
    # X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    # Bx, By, Bz = sf.get_field(X.ravel(), Y.ravel(), Z.ravel())
    # Bx = Bx.reshape(X.shape)
    # By = By.reshape(X.shape)
    # Bz = Bz.reshape(X.shape)
    # data = np.column_stack([
    #     X.ravel(), Y.ravel(), Z.ravel(),
    #     Bx.ravel(), By.ravel(), Bz.ravel(),
    # ])
    # np.savetxt(test_data_dir / "test_solenoid_vs_varsol_fieldmap.dat", data)
    # parser = StandardFieldMapParser()
    # df_raw_data = parser.parse(test_data_dir / "test_solenoid_vs_varsol_fieldmap.dat")

    # # Fit the field map data
    # fitter = FieldFitter(df_raw_data=df_raw_data,
    #     xy_point=(0, 0),
    #     dx=dx,
    #     dy=dy,
    #     ds=1,
    #     min_region_size=10,
    #     deg=multipole_order-1,
    # )
    # # Use lower field_tol to include transverse field gradients needed for solenoid focusing
    # fitter.field_tol = 1e-4
    # fitter.set()
    # fitter.save_fit_pars(fit_pars_path)

    # Build solenoid using SplineBorisSequence - automatically creates one SplineBoris
    # element per polynomial piece with n_steps based on the data point count
    df_fit_pars = pd.read_csv(fit_pars_path)
    seq = xt.SplineBorisSequence(
        df_fit_pars=df_fit_pars,
        multipole_order=multipole_order,
        steps_per_point=1,  # one integration step per data point
    )

    # Get the Line of SplineBoris elements and track
    line_splineboris = seq.to_line()
    line_splineboris.build_tracker()
    p_splineboris = p0.copy()
    line_splineboris.track(p_splineboris, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_splineboris = line_splineboris.record_last_track

    # --- VariableSolenoid reference (paraxial approximation, on-axis Bz only) ---
    z_axis_ref = np.linspace(0, interval, n_steps)
    # Get on-axis Bz
    Bz_axis = sf.get_field(0 * z_axis_ref, 0 * z_axis_ref, z_axis_ref)[2]
    P0_J = p0.p0c[0] * qe / clight
    brho = P0_J / qe / p0.q0
    ks = Bz_axis / brho
    ks_entry = ks[:-1]
    ks_exit = ks[1:]
    dz = z_axis_ref[1] - z_axis_ref[0]
    line_varsol = xt.Line(elements=[
        xt.VariableSolenoid(length=dz, ks_profile=[ks_entry[ii], ks_exit[ii]])
        for ii in range(len(z_axis_ref) - 1)
    ])
    line_varsol.build_tracker()
    p_varsol = p0.copy()
    line_varsol.track(p_varsol, turn_by_turn_monitor='ONE_TURN_EBE')
    mon_varsol = line_varsol.record_last_track

    # Define check points in the solenoid region
    z_check = sf.z0 + sf.L * np.linspace(-2, 2, 1001)

    # Compare trajectories for each particle
    n_part = mon_splineboris.x.shape[0]
    for i_part in range(n_part):

        # VariableSolenoid trajectory derivatives
        s_varsol = 0.5 * (mon_varsol.s[i_part, :-1] + mon_varsol.s[i_part, 1:])
        dx_ds_varsol = np.diff(mon_varsol.x[i_part, :]) / np.diff(mon_varsol.s[i_part, :])
        dy_ds_varsol = np.diff(mon_varsol.y[i_part, :]) / np.diff(mon_varsol.s[i_part, :])

        # SplineBoris trajectory derivatives
        s_splineboris = 0.5 * (mon_splineboris.s[i_part, :-1] + mon_splineboris.s[i_part, 1:])
        dx_ds_splineboris = np.diff(mon_splineboris.x[i_part, :]) / np.diff(mon_splineboris.s[i_part, :])
        dy_ds_splineboris = np.diff(mon_splineboris.y[i_part, :]) / np.diff(mon_splineboris.s[i_part, :])

        # Interpolate at check points
        dx_ds_splineboris_check = np.interp(z_check, s_splineboris, dx_ds_splineboris)
        dy_ds_splineboris_check = np.interp(z_check, s_splineboris, dy_ds_splineboris)

        dx_ds_varsol_check = np.interp(z_check, s_varsol, dx_ds_varsol)
        dy_ds_varsol_check = np.interp(z_check, s_varsol, dy_ds_varsol)

        # Assert that SplineBoris matches the VariableSolenoid reference
        xo.assert_allclose(dx_ds_splineboris_check, dx_ds_varsol_check, rtol=0,
                atol=2.8e-2 * (np.max(dx_ds_varsol_check) - np.min(dx_ds_varsol_check)))
        xo.assert_allclose(dy_ds_splineboris_check, dy_ds_varsol_check, rtol=0,
                atol=2.8e-2 * (np.max(dy_ds_varsol_check) - np.min(dy_ds_varsol_check)))



def test_splineboris_undulator_vs_boris_spatial():
    """
    Build a lightweight undulator from spline-fit parameters using SplineBorisSequence
    and check that tracking with SplineBoris and BorisSpatialIntegrator gives consistent
    end coordinates.
    """

    # ------------------------------------------------------------------
    # Load fit parameters and build undulator using SplineBorisSequence
    # ------------------------------------------------------------------
    base_dir = (
        Path(__file__).parent.parent
        / "test_data"
        / "splineboris"
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

    # NOTE: If the fit parameters need to be updated, uncomment the following code.
    # # Add examples directory to path to import FieldFitter and StandardFieldMapParser
    # examples_path = Path(__file__).parent.parent / "examples" / "splineboris" / "spline_fitter"
    # if str(examples_path) not in sys.path:
    #     sys.path.insert(0, str(examples_path))
    # from field_fitter import FieldFitter
    # from fieldmap_parsers import StandardFieldMapParser
    
    # # Load the undulator field map from examples
    # fieldmap_path = (
    #     Path(__file__).parent.parent
    #     / "examples"
    #     / "splineboris"
    #     / "spline_fitter"
    #     / "field_maps"
        
    # )
    # parser = StandardFieldMapParser()
    # df_raw_data = parser.parse(filepath + "knot_map_test.txt")
    
    # # Fit the field map data
    # fitter = FieldFitter(
    #     df_raw_data=df_raw_data,
    #     xy_point=(0, 0),
    #     dx=0.001,
    #     dy=0.001,
    #     ds=0.001,
    #     min_region_size=10,
    #     deg=multipole_order-1,
    # )
    # fitter.set()
    # fitter.save_fit_pars(filepath)

    # Build undulator using SplineBorisSequence
    seq = xt.SplineBorisSequence(
        df_fit_pars=df,
        multipole_order=multipole_order,
        steps_per_point=1,
    )

    line_spline = seq.to_line()

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
    # Extract parameters from SplineBorisSequence elements
    # ------------------------------------------------------------------
    boris_elems = []
    for elem in seq.elements:
        # Each element has n_steps rows in par_table, all identical for a single piece
        # Use the first row to create the field callable
        params_i = np.asarray(elem.par_table[0], dtype=float)
        field_i = make_segment_field(params_i, multipole_order)

        boris_elems.append(
            xt.BorisSpatialIntegrator(
                fieldmap_callable=field_i,
                s_start=float(elem.s_start),
                s_end=float(elem.s_end),
                n_steps=int(elem.n_steps),
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
    xo.assert_allclose(p_spline.x, p_boris.x, rtol=1e-12, atol=1e-11)
    xo.assert_allclose(p_spline.px, p_boris.px, rtol=1e-12, atol=1e-11)
    xo.assert_allclose(p_spline.y, p_boris.y, rtol=1e-12, atol=1e-11)
    xo.assert_allclose(p_spline.py, p_boris.py, rtol=1e-12, atol=1e-11)
    xo.assert_allclose(p_spline.zeta, p_boris.zeta, rtol=1e-12, atol=1e-11)
    xo.assert_allclose(p_spline.delta, p_boris.delta, rtol=1e-12, atol=1e-11)