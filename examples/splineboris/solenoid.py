import numpy as np
from pathlib import Path
from scipy.constants import c as clight
from scipy.constants import e as qe

import xobjects as xo
import xtrack as xt
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from spline_fitter.field_fitter import FieldFitter
from spline_fitter.fieldmap_parsers import StandardFieldMapParser
import matplotlib.pyplot as plt
import pandas as pd

n_steps = 1500
dx = 0.001
dy = 0.001
ds = 0.001

delta=np.array([0, 4])
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                energy0=45.6e9/1000,
                x=[-1e-3, -1e-3],
                px=-1e-3*(1+delta),
                y=1e-3,
                delta=delta)

sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=20)

field_maps_dir = Path(__file__).parent / "field_maps"
field_maps_dir.mkdir(exist_ok=True)
fit_pars_path = field_maps_dir / "solenoid_fit_pars.csv"

if not fit_pars_path.exists():
    # Construct field map and fit
    x_axis = np.linspace(-dx, dx, 5)
    y_axis = np.linspace(-dy, dy, 5)
    z_axis = np.linspace(0, 30, 1001)
    X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    Bx, By, Bz = sf.get_field(X.ravel(), Y.ravel(), Z.ravel())
    Bx = Bx.reshape(X.shape)
    By = By.reshape(X.shape)
    Bz = Bz.reshape(X.shape)
    data = np.column_stack([
        X.ravel(), Y.ravel(), Z.ravel(),
        Bx.ravel(), By.ravel(), Bz.ravel(),
    ])
    np.savetxt(field_maps_dir / "solenoid_field.dat", data)
    parser = StandardFieldMapParser()
    df_raw_data = parser.parse(field_maps_dir / "solenoid_field.dat")
    fitter = FieldFitter(df_raw_data=df_raw_data,
        xy_point=(0, 0),
        dx=dx,
        dy=dy,
        ds=ds,
        min_region_size=10,
        deg=4,
    )
    fitter.set()
    fitter.save_fit_pars(fit_pars_path)

z_axis = np.linspace(0, 30, 1001)  # for analytical reference below
df_fit_pars = pd.read_csv(fit_pars_path)
par_table, s_start, s_end = xt.SplineBoris.build_parameter_table_from_df(
    df_fit_pars=df_fit_pars,
    n_steps=n_steps,
    multipole_order=5,
)

# Build solenoid as many successive SplineBoris elements (like undulator_open and friends)
ds_spline = (s_end - s_start) / n_steps
s_vals = np.linspace(s_start, s_end, n_steps)
solenoid_elements = []
for i in range(n_steps):
    params_i = [par_table[i].tolist()]
    s_val_i = s_vals[i]
    elem_s_start = s_val_i - ds_spline / 2
    elem_s_end = s_val_i + ds_spline / 2
    elem_i = xt.SplineBoris(
        par_table=params_i,
        multipole_order=5,
        s_start=elem_s_start,
        s_end=elem_s_end,
        n_steps=1,
    )
    solenoid_elements.append(elem_i)

line_splineboris = xt.Line(elements=solenoid_elements)
line_splineboris.build_tracker()
p_splineboris = p0.copy()
p_ref = p0.copy()
line_splineboris.track(p_splineboris, turn_by_turn_monitor='ONE_TURN_EBE')
mon_splineboris = line_splineboris.record_last_track

# --- Analytical reference: VariableSolenoid line from on-axis Bz(s) ---
# Build thin-element representation of the same solenoid (Wolsky / MAD-X).
Bz_axis = sf.get_field(0 * z_axis, 0 * z_axis, z_axis)[2]
P0_J = p0.p0c[0] * qe / clight
brho = P0_J / qe / p0.q0
ks = Bz_axis / brho
ks_entry = ks[:-1]
ks_exit = ks[1:]
dz = z_axis[1] - z_axis[0]
line_ref = xt.Line(elements=[
    xt.VariableSolenoid(length=dz, ks_profile=[ks_entry[ii], ks_exit[ii]])
    for ii in range(len(z_axis) - 1)
])
line_ref.build_tracker()
line_ref.track(p_ref, turn_by_turn_monitor='ONE_TURN_EBE')
mon_ref = line_ref.record_last_track

# Plot particle tracks (x, y vs s): SplineBoris vs analytical reference
n_part = mon_splineboris.x.shape[0]
fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
for i in range(n_part):
    axes[0].plot(mon_splineboris.s[i, :], mon_splineboris.x[i, :] * 1e3, '-', label=f'SplineBoris p{i}' if i < 2 else '')
    axes[0].plot(mon_ref.s[i, :], mon_ref.x[i, :] * 1e3, '--', alpha=0.8, label=f'Ref p{i}' if i < 2 else '')
    axes[1].plot(mon_splineboris.s[i, :], mon_splineboris.y[i, :] * 1e3, '-', label=f'SplineBoris p{i}' if i < 2 else '')
    axes[1].plot(mon_ref.s[i, :], mon_ref.y[i, :] * 1e3, '--', alpha=0.8, label=f'Ref p{i}' if i < 2 else '')
axes[0].set_ylabel('x [mm]')
axes[1].set_ylabel('y [mm]')
axes[1].set_xlabel('s [m]')
axes[0].legend(ncol=2)
axes[1].legend(ncol=2)
axes[0].set_title('Particle tracks: SplineBoris (solid) vs analytical ref (dashed)')
fig.tight_layout()
plt.show()

# SplineBoris (fitted field) vs analytical reference (allow tol from fit)
xo.assert_allclose(mon_splineboris.x[:, -1], mon_ref.x[:, -1], rtol=0, atol=5e-3)
xo.assert_allclose(mon_splineboris.px[:, -1], mon_ref.px[:, -1], rtol=0, atol=5e-3)
xo.assert_allclose(mon_splineboris.y[:, -1], mon_ref.y[:, -1], rtol=0, atol=5e-3)
xo.assert_allclose(mon_splineboris.py[:, -1], mon_ref.py[:, -1], rtol=0, atol=5e-3)
xo.assert_allclose(mon_splineboris.s[:, -1], mon_ref.s[:, -1], rtol=0, atol=5e-3)
xo.assert_allclose(mon_splineboris.delta[:, -1], mon_ref.delta[:, -1], rtol=0, atol=5e-3)

