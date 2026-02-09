"""
Spin tracking with undulators and radiation.

This script loads the SLS MADX file, builds undulator using SplineBorisSequence,
computes twiss with spin tracking and radiation, and displays results.
"""

import xtrack as xt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from spline_fitter.field_fitter import FieldFitter


multipole_order = 3

# Particle reference
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)

# Load SLS MADX file
madx_file = Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls' / 'b075_2024.09.25.madx'
env = xt.load(str(madx_file))
line_sls = env.ring

# Configure bend model
line_sls.configure_bend_model(core='mat-kick-mat')

# Set particle reference
line_sls.particle_ref = p0.copy()

BASE_DIR = Path(__file__).resolve().parent

# Load the raw field map data from knot_map_test.txt
field_map_path = BASE_DIR / "field_maps" / "knot_map_test.txt"
df_raw_data = pd.read_csv(
    field_map_path,
    sep='\t',
    header=None,
    names=['X', 'Y', 'Z', 'Bx', 'By', 'Bs'],
)
df_raw_data = df_raw_data.set_index(['X', 'Y', 'Z'])

# Grid spacing in meters (the dataset uses mm, so 1 mm = 0.001 m)
dx = 0.001
dy = 0.001
ds = 0.001

field_fitter = FieldFitter(
    df_raw_data=df_raw_data,
    xy_point=(0, 0),
    dx=dx,
    dy=dy,
    ds=ds,
    min_region_size=10,
    deg=multipole_order-1,
)

field_fitter.fit()
field_fitter.save_fit_pars(
    BASE_DIR
    / "field_maps"
    / "field_fit_pars.csv"
)

# Build undulator using SplineBorisSequence - automatically creates one SplineBoris
# element per polynomial piece with n_steps based on the data point count
seq = xt.SplineBorisSequence(
    df_fit_pars=field_fitter.df_fit_pars,
    multipole_order=multipole_order,
    steps_per_point=1,
)

# Get the Line of SplineBoris elements (pass env for insert support)
piecewise_undulator = seq.to_line(env=env)
l_wig = seq.length

piecewise_undulator.build_tracker()

piecewise_undulator.particle_ref = p0.copy()

# The issue: When you use betx=1, bety=1, twiss4d treats the line as OPEN (non-periodic).
# For an open line, the orbit is computed from initial conditions in particle_on_co.
# If particle_on_co has zero initial conditions (x=0, px=0, y=0, py=0), the orbit will
# be zero unless there are kicks from the wiggler.
#
# Solution: Use only_orbit=True to explicitly compute the orbit, which should properly
# propagate through the wiggler and show any kicks/deviations.

# Create env variables for corrector strengths (needed for matching)
env['k0l_corr1'] = 0.
env['k0l_corr2'] = 0.
env['k0l_corr3'] = 0.
env['k0l_corr4'] = 0.
env['k0sl_corr1'] = 0.
env['k0sl_corr2'] = 0.
env['k0sl_corr3'] = 0.
env['k0sl_corr4'] = 0.

# Create corrector elements with expressions referencing env variables
env.new('corr1', xt.Multipole, knl=['k0l_corr1'], ksl=['k0sl_corr1'])
env.new('corr2', xt.Multipole, knl=['k0l_corr2'], ksl=['k0sl_corr2'])
env.new('corr3', xt.Multipole, knl=['k0l_corr3'], ksl=['k0sl_corr3'])
env.new('corr4', xt.Multipole, knl=['k0l_corr4'], ksl=['k0sl_corr4'])

# Insert correctors at nearest element boundary (s_tol avoids slicing)
piecewise_undulator.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_wig - 0.1),
    env.place('corr4', at=l_wig - 0.02),
], s_tol=5e-3)

# Matching targets use corrector element names for intermediate positions
opt = piecewise_undulator.match(
    solve=False,
    betx=0, bety=0,
    only_orbit=True,
    include_collective=True,
    vary=xt.VaryList(['k0l_corr1', 'k0sl_corr1',
                      'k0l_corr2', 'k0sl_corr2',
                      'k0l_corr3', 'k0sl_corr3',
                      'k0l_corr4', 'k0sl_corr4',
                      ], step=1e-6),
    targets=[
        xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
        xt.TargetSet(x=0., y=0, at='corr2'),
        xt.TargetSet(x=0., y=0, at='corr3')
        ],
)
opt.step(2)


piecewise_undulator.particle_ref.anomalous_magnetic_moment = 0.00115965218128

tw_undulator_corr_spin = piecewise_undulator.twiss4d(
    betx=1, bety=1, 
    include_collective=True, 
    spin=True,
    spin_x=0.5, spin_y=0.25, spin_z=0.25
    )
tw_undulator_corr_spin.plot('x y')
tw_undulator_corr_spin.plot('betx bety', 'dx dy')
tw_undulator_corr_spin.plot('spin_x spin_y spin_z')

plt.show()


# Enable radiation on all elements (this sets radiation_flag on elements)
piecewise_undulator.configure_radiation(model='mean')

# Verify radiation is enabled on SplineBoris elements
tt = piecewise_undulator.get_table()
spline_boris_elements = tt.rows[tt.element_type == 'SplineBoris']
if len(spline_boris_elements) > 0:
    first_elem = piecewise_undulator[spline_boris_elements.name[0]]
    print(f"Radiation flag on SplineBoris element: {first_elem.radiation_flag}")

# Run twiss4d including radiation effects (average energy loss)
tw_undulator_corr_spin_rad = piecewise_undulator.twiss4d(
    betx=1, bety=1,
    include_collective=True,
    spin=True,
    spin_x=0.5, spin_y=0.25, spin_z=0.25,
    radiation_method='full'  # Use 'kick_as_co' for average energy loss, or 'full' for full computation
)

# Check if there's any energy loss (delta change)
delta_no_rad = tw_undulator_corr_spin.delta
delta_with_rad = tw_undulator_corr_spin_rad.delta
delta_diff = delta_with_rad - delta_no_rad
print(f"\nEnergy loss check:")
print(f"  Max |delta| without radiation: {np.max(np.abs(delta_no_rad)):.2e}")
print(f"  Max |delta| with radiation: {np.max(np.abs(delta_with_rad)):.2e}")
print(f"  Max |delta difference|: {np.max(np.abs(delta_diff)):.2e}")
if np.max(np.abs(delta_diff)) < 1e-10:
    print("  WARNING: No measurable energy loss detected! Radiation may not be working.")
else:
    print(f"  Energy loss detected: {np.max(np.abs(delta_diff)) * piecewise_undulator.particle_ref.energy0[0] / 1e6:.4f} MeV")

# Plot results to compare with/without radiation
fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

# Plot closed orbit x
axs[0].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.x, label='no rad')
axs[0].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.x, label='with rad')
axs[0].set_ylabel('x [m]')
axs[0].legend()
axs[0].grid(True)

# Plot energy loss (delta)
axs[1].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.delta, label='delta (no rad)', linestyle='-')
axs[1].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.delta, label='delta (with rad)', linestyle='--')
axs[1].set_ylabel('delta (relative energy deviation)')
axs[1].legend()
axs[1].grid(True)

# Plot spin components
axs[2].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.spin_x, label='spin_x (no rad)', linestyle='-')
axs[2].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.spin_y, label='spin_y (no rad)', linestyle='-')
axs[2].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.spin_z, label='spin_z (no rad)', linestyle='-')
axs[2].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.spin_x, label='spin_x (with rad)', linestyle='--')
axs[2].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.spin_y, label='spin_y (with rad)', linestyle='--')
axs[2].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.spin_z, label='spin_z (with rad)', linestyle='--')
axs[2].set_ylabel('spin')
axs[2].set_xlabel('s [m]')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()