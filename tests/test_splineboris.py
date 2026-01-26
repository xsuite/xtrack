import numpy as np
from scipy.constants import c as clight
from scipy.constants import e as qe
import pytest

import xtrack as xt
from xtrack.beam_elements.spline_param_schema import SplineParameterSchema

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


def test_splineboris_solenoid_boris_integrator():
    """
    Test SplineBoris against BorisSpatialIntegrator with SolenoidField.
    
    This test:
    1. Generates field map data from SolenoidField
    2. Fits the field using FieldFitter (with caching)
    3. Builds SplineBoris element from fit parameters
    4. Compares tracking results with BorisSpatialIntegrator
    """
    import sys
    import pandas as pd
    from pathlib import Path
    
    # Add examples directory to path to import FieldFitter
    examples_path = Path(__file__).parent.parent / "examples" / "undulator" / "spline_fitter"
    if str(examples_path) not in sys.path:
        sys.path.insert(0, str(examples_path))
    from field_fitter import FieldFitter
    
    from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
    from xtrack.beam_elements.spline_param_schema import (
        build_parameter_table_from_df,
        FIELD_FIT_INDEX_COLUMNS,
    )
    
    # 1. Setup SolenoidField and particle configuration (matching test_boris_spatial.py)
    sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=20)
    
    delta = np.array([0, 4])
    p0 = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1,
        energy0=45.6e9/1000,
        x=[-1e-3, -1e-3],
        px=-1e-3*(1+delta),
        y=1e-3,
        delta=delta
    )
    
    # 2. Generate field data on grid (X, Y, Z) and convert to DataFrame
    # Define grid via dx, dy, ds and counts; FieldFitter expects X,Z as indices (s = Z*ds, x_phys = X*dx)
    dx, dy, ds = 0.001, 0.001, 0.03
    n_x, n_z = 21, 1001
    n_x_half = (n_x - 1) // 2
    x_idx = np.arange(-n_x_half, n_x_half + 1)
    z_idx = np.arange(n_z)
    y_vals = np.array([0.0])  # On-axis, FieldFitter computes derivatives from X variation

    x_phys = x_idx.astype(float) * dx
    z_phys = z_idx.astype(float) * ds

    Zp_grid, Xp_grid, Y_grid = np.meshgrid(z_phys, x_phys, y_vals, indexing='ij')
    Zi_grid, Xi_grid, _ = np.meshgrid(z_idx, x_idx, y_vals, indexing='ij')

    x_flat = Xp_grid.flatten()
    y_flat = Y_grid.flatten()
    z_flat = Zp_grid.flatten()
    xi_flat = Xi_grid.flatten()
    zi_flat = Zi_grid.flatten()

    Bx_flat, By_flat, Bs_flat = sf.get_field(x_flat, y_flat, z_flat)

    rows = []
    for i in range(len(x_flat)):
        rows.append({
            'X': float(xi_flat[i]),
            'Y': float(y_flat[i]),
            'Z': float(zi_flat[i]),
            'Bx': float(Bx_flat[i]),
            'By': float(By_flat[i]),
            'Bs': float(Bs_flat[i])
        })
    
    df_raw_data = pd.DataFrame(rows)
    df_raw_data.set_index(['X', 'Y', 'Z'], inplace=True)
    
    # Force DataFrame to be fully writable by recreating with explicit array copies
    # This ensures all underlying arrays are independent and writable
    # Use to_numpy(copy=True) and ensure arrays are writable
    bx_arr = df_raw_data['Bx'].to_numpy(copy=True)
    by_arr = df_raw_data['By'].to_numpy(copy=True)
    bs_arr = df_raw_data['Bs'].to_numpy(copy=True)
    x_arr = df_raw_data.index.get_level_values('X').to_numpy(copy=True)
    y_arr = df_raw_data.index.get_level_values('Y').to_numpy(copy=True)
    z_arr = df_raw_data.index.get_level_values('Z').to_numpy(copy=True)
    
    # Ensure arrays are writable (setflags)
    bx_arr.setflags(write=True)
    by_arr.setflags(write=True)
    bs_arr.setflags(write=True)
    x_arr.setflags(write=True)
    y_arr.setflags(write=True)
    z_arr.setflags(write=True)
    
    df_raw_data = pd.DataFrame({
        'Bx': bx_arr,
        'By': by_arr,
        'Bs': bs_arr
    }, index=pd.MultiIndex.from_arrays([x_arr, y_arr, z_arr], names=['X', 'Y', 'Z']))
    
    # 3. Fit parameter caching: check file existence, load or fit and save
    test_data_dir = Path(__file__).parent.parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)
    fit_pars_path = test_data_dir / "solenoid_field_fit_pars.csv"
    
    if fit_pars_path.exists():
        # Load existing fit parameters
        df_fit_pars = pd.read_csv(fit_pars_path, index_col=list(FIELD_FIT_INDEX_COLUMNS))
    else:
        # Create FieldFitter and perform fitting (same dx, dy, ds as grid)
        fitter = FieldFitter(
            df_raw_data=df_raw_data,
            xy_point=(0.0, 0.0),
            dx=dx,
            dy=dy,
            ds=ds,
            min_region_size=10,
            deg=2,  # For derivatives
        )
        fitter.set()
        fitter.save_fit_pars(fit_pars_path)
        df_fit_pars = fitter.df_fit_pars
    
    # 4. Build SplineBoris elements from fit parameters (series of elements, one per step)
    n_steps = 1000
    multipole_order = 2  # To include derivatives
    
    par_table, s_start_fit, s_end_fit = build_parameter_table_from_df(
        df_fit_pars=df_fit_pars,
        n_steps=n_steps,
        multipole_order=multipole_order,
    )
    
    # Create a series of SplineBoris elements, one for each step
    # This matches the pattern used in examples/undulator/sls_with_undulators_closed_spin_radiation.py
    splineboris_list = []
    s_vals = np.linspace(s_start_fit, s_end_fit, n_steps)
    ds_step = (s_end_fit - s_start_fit) / n_steps

    for i in range(n_steps):
        # Each element covers a small slice of the field map
        # params should be a 2D array: [[param1, param2, ...]] for n_steps=1
        params_i = [par_table[i].tolist()]
        s_val_i = s_vals[i]

        # For each single-step element, s_start and s_end define the range
        # in the field map that this step covers. Use a small interval around s_val_i.
        elem_s_start = s_val_i - ds_step / 2
        elem_s_end = s_val_i + ds_step / 2
        
        splineboris_i = xt.SplineBoris(
            par_table=params_i,
            multipole_order=multipole_order,
            s_start=elem_s_start,
            s_end=elem_s_end,
            n_steps=1,
        )
        splineboris_list.append(splineboris_i)
    
    # 5. Create BorisSpatialIntegrator element and track particles
    integrator = xt.BorisSpatialIntegrator(
        fieldmap_callable=sf.get_field,
        s_start=0,
        s_end=30,
        n_steps=n_steps
    )
    
    # Track with both elements
    p_splineboris = p0.copy()
    p_boris = p0.copy()
    
    line_splineboris = xt.Line(elements=splineboris_list)
    line_splineboris.particle_ref = p_splineboris
    line_splineboris.track(p_splineboris)
    
    integrator.track(p_boris)
    
    # 6. Compare results with appropriate tolerances
    # Use similar tolerances as in test_boris_spatial.py (relative tolerances around 2-3%)
    rtol = 0.03  # 3% relative tolerance
    atol_x = 1e-6  # Absolute tolerance for positions (1 micron)
    atol_p = 1e-6  # Absolute tolerance for momenta
    
    np.testing.assert_allclose(p_splineboris.x, p_boris.x, rtol=rtol, atol=atol_x)
    np.testing.assert_allclose(p_splineboris.y, p_boris.y, rtol=rtol, atol=atol_x)
    np.testing.assert_allclose(p_splineboris.px, p_boris.px, rtol=rtol, atol=atol_p)
    np.testing.assert_allclose(p_splineboris.py, p_boris.py, rtol=rtol, atol=atol_p)