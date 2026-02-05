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

interval = 30
dx = 0.001
dy = 0.001
multipole_order = 4
n_steps = 5000

delta = np.array([0, 0.001])
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1,
                energy0=45.6e6,  # 45.6 GeV (e.g. FCC-ee Z-pole)
                x=1e-3,  # Start slightly off-axis
                px=-1e-3*(1+delta),
                y=1e-3,
                delta=delta)

sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=20)

def get_field(x, y, z):
    return sf.get_field(x, y, z)

field_maps_dir = Path(__file__).parent / "field_maps"
field_maps_dir.mkdir(exist_ok=True)
fit_pars_path = field_maps_dir / "solenoid_fit_pars.csv"


# Construct field map and fit
x_axis = np.linspace(-multipole_order*dx/2, multipole_order*dx/2, multipole_order+1)
y_axis = np.linspace(-multipole_order*dy/2, multipole_order*dy/2, multipole_order+1)
z_axis = np.linspace(0, interval, n_steps+1)
X, Y, Z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
Bx, By, Bz = get_field(X.ravel(), Y.ravel(), Z.ravel())
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
# ds=1 so that s_full = Z * ds = Z (meters). Raw Z from the field map is in meters.
fitter = FieldFitter(df_raw_data=df_raw_data,
    xy_point=(0, 0),
    dx=dx,
    dy=dy,
    ds=1,
    min_region_size=10,
    deg=multipole_order-1,
)
# Use lower field_tol to include transverse field gradients needed for solenoid focusing
fitter.field_tol = 1e-4
fitter.set()
fitter.save_fit_pars(fit_pars_path)

# Plot the fit results
# NOTE: Fit corresponds well with data.
for i in range(multipole_order):
    fitter.plot_fields(der=i)
    plt.show()

# Build solenoid using SplineBorisSequence - automatically creates one SplineBoris
# element per polynomial piece with n_steps based on the data point count
df_fit_pars = pd.read_csv(fit_pars_path)
seq = xt.SplineBorisSequence(
    df_fit_pars=df_fit_pars,
    multipole_order=multipole_order,
    steps_per_point=5,  # one integration step per data point
)

# Get the Line of SplineBoris elements
line_splineboris = seq.to_line()
line_splineboris.build_tracker()
p_splineboris = p0.copy()
line_splineboris.track(p_splineboris, turn_by_turn_monitor='ONE_TURN_EBE')
mon_splineboris = line_splineboris.record_last_track

# --- TRUE REFERENCE: BorisSpatialIntegrator with same analytical field ---
# This is the gold standard - uses the same full 3D field directly
p_boris = p0.copy()
boris_integrator = xt.BorisSpatialIntegrator(
    fieldmap_callable=get_field,  # Same field function as used for fitting
    s_start=0,
    s_end=interval,
    n_steps=n_steps,
)
boris_integrator.track(p_boris)

# --- VariableSolenoid reference (paraxial approximation, on-axis Bz only) ---
n_ref_steps = n_steps
z_axis_ref = np.linspace(0, interval, n_ref_steps)
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

# === COMPARISON: Quantify agreement between trackers ===
print("\n" + "="*60)
print("TRACKING COMPARISON")
print("="*60)

# Final state comparison
print(f"\n--- Final particle states (delta={delta.tolist()}) ---")
print(f"{'':20s} {'SplineBoris':>15s} {'Boris(ref)':>15s} {'VarSol':>15s}")
for i in range(len(delta)):
    print(f"Particle {i} (delta={delta[i]}):")
    print(f"{'  x [mm]':20s} {p_splineboris.x[i]*1e3:15.6f} {p_boris.x[i]*1e3:15.6f} {p_varsol.x[i]*1e3:15.6f}")
    print(f"{'  y [mm]':20s} {p_splineboris.y[i]*1e3:15.6f} {p_boris.y[i]*1e3:15.6f} {p_varsol.y[i]*1e3:15.6f}")
    print(f"{'  px [mrad]':20s} {p_splineboris.px[i]*1e3:15.6f} {p_boris.px[i]*1e3:15.6f} {p_varsol.px[i]*1e3:15.6f}")
    print(f"{'  py [mrad]':20s} {p_splineboris.py[i]*1e3:15.6f} {p_boris.py[i]*1e3:15.6f} {p_varsol.py[i]*1e3:15.6f}")

# Differences
print("\n--- SplineBoris vs Boris (fitted spline vs analytical field) ---")
for i in range(len(delta)):
    dx = (p_splineboris.x[i] - p_boris.x[i]) * 1e6  # in um
    dy = (p_splineboris.y[i] - p_boris.y[i]) * 1e6
    dpx = (p_splineboris.px[i] - p_boris.px[i]) * 1e6  # in urad
    dpy = (p_splineboris.py[i] - p_boris.py[i]) * 1e6
    print(f"  particle {i} (delta={delta[i]}): Δx={dx:.2f}um, Δy={dy:.2f}um, Δpx={dpx:.2f}urad, Δpy={dpy:.2f}urad")

print("\n--- Boris vs VariableSolenoid (full 3D field vs paraxial model) ---")
for i in range(len(delta)):
    dx = (p_boris.x[i] - p_varsol.x[i]) * 1e3
    dy = (p_boris.y[i] - p_varsol.y[i]) * 1e3
    dpx = (p_boris.px[i] - p_varsol.px[i]) * 1e3
    dpy = (p_boris.py[i] - p_varsol.py[i]) * 1e3
    print(f"  particle {i} (delta={delta[i]}): Δx={dx:.4f}mm, Δy={dy:.4f}mm, Δpx={dpx:.4f}mrad, Δpy={dpy:.4f}mrad")

# Use mon_varsol as mon_ref for plotting
mon_ref = mon_varsol

# Plot particle tracks in 3D: x horizontal, y vertical, s longitudinal
n_part = mon_splineboris.x.shape[0]
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Boris integrator logs
x_boris = np.array(boris_integrator.x_log)
y_boris = np.array(boris_integrator.y_log)
z_boris = np.array(boris_integrator.z_log)

colors = plt.cm.tab10.colors
for i in range(n_part):
    # SplineBoris tracks (solid lines)
    ax.plot(mon_splineboris.s[i, :], 
            mon_splineboris.x[i, :] * 1e3, 
            mon_splineboris.y[i, :] * 1e3, 
            '-', color=colors[i], linewidth=2,
            label=f'SplineBoris p{i}')
    # Boris integrator (dotted - this is the TRUE reference)
    ax.plot(z_boris[:, i], 
            x_boris[:, i] * 1e3, 
            y_boris[:, i] * 1e3, 
            ':', color=colors[i], linewidth=2,
            label=f'Boris p{i}')
    # VariableSolenoid tracks (dashed lines)
    ax.plot(mon_ref.s[i, :], 
            mon_ref.x[i, :] * 1e3, 
            mon_ref.y[i, :] * 1e3, 
            '--', color=colors[i], alpha=0.5, linewidth=1.5,
            label=f'VarSol p{i}')

ax.set_xlabel('s [m]')
ax.set_ylabel('x [mm]')
ax.set_zlabel('y [mm]')
ax.set_title('SplineBoris (solid) vs Boris ref (dotted) vs VariableSolenoid (dashed)')
ax.legend(loc='upper left')
ax.view_init(elev=20, azim=-60)
fig.tight_layout()
plt.show()

# Also plot 2D comparison in x and y vs s (easier to see differences)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for i in range(n_part):
    # x vs s
    axes[0, i].plot(mon_splineboris.s[i, :], mon_splineboris.x[i, :] * 1e3, '-', 
                    label='SplineBoris', linewidth=2)
    axes[0, i].plot(z_boris[:, i], x_boris[:, i] * 1e3, ':', 
                    label='Boris (ref)', linewidth=2)
    axes[0, i].plot(mon_ref.s[i, :], mon_ref.x[i, :] * 1e3, '--', 
                    label='VarSol', alpha=0.7)
    axes[0, i].set_xlabel('s [m]')
    axes[0, i].set_ylabel('x [mm]')
    axes[0, i].set_title(f'Particle {i} (delta={delta[i]}): x vs s')
    axes[0, i].legend()
    axes[0, i].grid(True, alpha=0.3)
    
    # y vs s
    axes[1, i].plot(mon_splineboris.s[i, :], mon_splineboris.y[i, :] * 1e3, '-', 
                    label='SplineBoris', linewidth=2)
    axes[1, i].plot(z_boris[:, i], y_boris[:, i] * 1e3, ':', 
                    label='Boris (ref)', linewidth=2)
    axes[1, i].plot(mon_ref.s[i, :], mon_ref.y[i, :] * 1e3, '--', 
                    label='VarSol', alpha=0.7)
    axes[1, i].set_xlabel('s [m]')
    axes[1, i].set_ylabel('y [mm]')
    axes[1, i].set_title(f'Particle {i} (delta={delta[i]}): y vs s')
    axes[1, i].legend()
    axes[1, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# SplineBoris (fitted spline) vs Boris (analytical field) - should agree within fit accuracy
xo.assert_allclose(p_splineboris.x, p_boris.x, rtol=0, atol=10e-6)  # 10 um
xo.assert_allclose(p_splineboris.y, p_boris.y, rtol=0, atol=10e-6)
xo.assert_allclose(p_splineboris.px, p_boris.px, rtol=0, atol=10e-6)  # 10 urad
xo.assert_allclose(p_splineboris.py, p_boris.py, rtol=0, atol=10e-6)
print("\nAssertions passed: SplineBoris matches Boris reference within 10um/10urad tolerance")
