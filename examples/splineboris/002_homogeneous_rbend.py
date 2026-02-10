"""
SplineBoris vs RBend comparison for a homogeneous dipole field.

This example demonstrates that a SplineBoris element with a constant
transverse magnetic field reproduces the same tracking results as the
standard RBend element. Several field orientations (field_angle) are
tested and the end-of-element coordinates are compared.
"""

import numpy as np
from scipy.constants import c as clight

import xobjects as xo
import xtrack as xt
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------
s_start = 0
s_end = 1
length = s_end - s_start
n_steps = 100

B_0 = 0.1  # [T] field strength

# Test several field orientations in the transverse plane
field_angles = [0, np.pi / 7, np.pi / 4, 4 * np.pi / 9, np.pi / 2,
                3 * np.pi / 4, np.pi]

# Store results for plotting
results = []

for field_angle in field_angles:
    # Field components in the transverse plane
    B_x = B_0 * np.cos(field_angle)
    B_y = B_0 * np.sin(field_angle)

    # ------------------------------------------------------------------
    # Build SplineBoris element with homogeneous field
    # ------------------------------------------------------------------
    # Spline coefficients: [f(s0), f'(s0), f(s1), f'(s1), integral]
    Bx_0_coeffs = np.array([B_x, 0.0, B_x, 0.0, B_x * length])
    By_0_coeffs = np.array([B_y, 0.0, B_y, 0.0, B_y * length])
    Bs_coeffs = np.zeros_like(Bx_0_coeffs)

    Bx_poly = xt.SplineBoris.spline_poly(s_start, s_end, Bx_0_coeffs)
    By_poly = xt.SplineBoris.spline_poly(s_start, s_end, By_0_coeffs)
    Bs_poly = xt.SplineBoris.spline_poly(s_start, s_end, Bs_coeffs)

    degree = 4
    ks_0 = np.zeros(degree + 1)
    ks_0[: len(Bx_poly.coef)] = Bx_poly.coef
    kn_0 = np.zeros(degree + 1)
    kn_0[: len(By_poly.coef)] = By_poly.coef
    bs = np.zeros(degree + 1)
    bs[: len(Bs_poly.coef)] = Bs_poly.coef

    param_table = xt.SplineBoris.build_param_table_from_spline_coeffs(
        bs=bs,
        kn={0: kn_0},
        ks={0: ks_0},
    )

    splineboris = xt.SplineBoris(
        par_table=param_table,
        s_start=s_start,
        s_end=s_end,
        multipole_order=1,
        n_steps=n_steps,
    )

    line_splineboris = xt.Line(elements=[splineboris])
    line_splineboris.particle_ref = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV,
        q0=1.0,
        energy0=1e9,
    )

    p_splineboris = line_splineboris.particle_ref.copy()
    p_splineboris.x = 1e-3
    p_splineboris.px = 1e-3

    # ------------------------------------------------------------------
    # Build equivalent RBend element
    # ------------------------------------------------------------------
    p_rbend = line_splineboris.particle_ref.copy()
    p_rbend.x = 1e-3
    p_rbend.px = 1e-3

    k0 = B_0 * clight / p_rbend.p0c[0]

    # rot_s_rad rotates the coordinate system so that the bend field
    # matches the desired field_angle orientation.
    rot_s_rad = field_angle - np.pi / 2
    b_rbend = xt.RBend(
        k0=k0,
        k0_from_h=False,
        length_straight=length,
        angle=0,
        edge_entry_angle=0,
        edge_exit_angle=0,
        rot_s_rad=rot_s_rad,
    )
    b_rbend.edge_entry_model = "suppressed"
    b_rbend.edge_exit_model = "suppressed"
    b_rbend.model = "bend-kick-bend"

    line_rbend = xt.Line(elements=[b_rbend])
    line_rbend.particle_ref = p_rbend

    # ------------------------------------------------------------------
    # Track
    # ------------------------------------------------------------------
    line_rbend.track(p_rbend)
    line_splineboris.track(p_splineboris)

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    angle_deg = np.degrees(field_angle)

    res = {
        "angle_deg": angle_deg,
        "x_rbend": p_rbend.x[0],
        "y_rbend": p_rbend.y[0],
        "px_rbend": p_rbend.kin_px[0],
        "py_rbend": p_rbend.kin_py[0],
        "x_sb": p_splineboris.x[0],
        "y_sb": p_splineboris.y[0],
        "px_sb": p_splineboris.kin_px[0],
        "py_sb": p_splineboris.kin_py[0],
    }
    results.append(res)

    print(f"\nfield_angle = {angle_deg:7.2f} deg")
    print(f"  x   : RBend = {res['x_rbend']:+.10e}  SplineBoris = {res['x_sb']:+.10e}"
          f"  diff = {abs(res['x_rbend'] - res['x_sb']):.2e}")
    print(f"  y   : RBend = {res['y_rbend']:+.10e}  SplineBoris = {res['y_sb']:+.10e}"
          f"  diff = {abs(res['y_rbend'] - res['y_sb']):.2e}")
    print(f"  px  : RBend = {res['px_rbend']:+.10e}  SplineBoris = {res['px_sb']:+.10e}"
          f"  diff = {abs(res['px_rbend'] - res['px_sb']):.2e}")
    print(f"  py  : RBend = {res['py_rbend']:+.10e}  SplineBoris = {res['py_sb']:+.10e}"
          f"  diff = {abs(res['py_rbend'] - res['py_sb']):.2e}")

# ---------------------------------------------------------------
# Plot comparison
# ---------------------------------------------------------------
angles = [r["angle_deg"] for r in results]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

for ax, key, label in zip(
    axes.flat,
    ["x", "y", "px", "py"],
    ["x [m]", "y [m]", "kin_px", "kin_py"],
):
    vals_rb = [r[f"{key}_rbend"] for r in results]
    vals_sb = [r[f"{key}_sb"] for r in results]
    ax.plot(angles, vals_rb, "o-", label="RBend")
    ax.plot(angles, vals_sb, "x--", label="SplineBoris")
    ax.set_ylabel(label)
    ax.legend()
    ax.grid(True, alpha=0.3)

axes[1, 0].set_xlabel("Field angle [deg]")
axes[1, 1].set_xlabel("Field angle [deg]")
fig.suptitle("SplineBoris vs RBend — homogeneous dipole field")
fig.tight_layout()
plt.show()
