"""
Spin tracking with undulators.

This script loads the SLS MADX file, inserts wigglers at 11 locations,
computes twiss with radiation integrals, and prints results.
"""

import xtrack as xt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from xtrack.beam_elements.spline_param_schema import build_parameter_table_from_df


multipole_order = 3

n_steps = 1000

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
# Use the field-fit parameters produced by the spline fitter example
filepath = (
    BASE_DIR
    / "spline_fitter"
    / "field_maps"
    / "field_fit_pars.csv"
)

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

# Build the canonical parameter table directly from the fit-parameter DataFrame.
par_table, s_start, s_end = build_parameter_table_from_df(
    df_fit_pars=df,
    n_steps=n_steps,
    multipole_order=multipole_order,
)

# # Configure pandas to display full DataFrame without truncation
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)

# # Print the full DataFrame
# print(df_reset)

# list of n_steps wigglers;
wiggler_list = []
name_list = ['wiggler_'+str(i) for i in range(1, n_steps+1)]

s_vals = np.linspace(s_start, s_end, n_steps)
l_wig = s_end - s_start
ds = (s_end - s_start) / n_steps

for i in range(n_steps):
    # params should be a 2D array: [[param1, param2, ...]] for n_steps=1
    params_i = [par_table[i].tolist()]
    s_val_i = s_vals[i]
        
    # For each single-step element, s_start and s_end should define the range
    # in the field map that this step covers. Use a small interval around s_val_i.
    # The s coordinates here are in the field map coordkinate system (can be negative).
    elem_s_start = s_val_i - ds/2
    elem_s_end = s_val_i + ds/2
    
    wiggler_i = xt.SplineBoris.from_parameter_table(
        par_table=params_i,
        multipole_order=multipole_order,
        s_start=elem_s_start,
        s_end=elem_s_end,
        n_steps=1,
    )
    wiggler_list.append(wiggler_i)
    env.elements[name_list[i]] = wiggler_i

piecewise_undulator = env.new_line(components=name_list)

piecewise_undulator.build_tracker()

piecewise_undulator.particle_ref = p0.copy()

# tw_undulator = piecewise_undulator.twiss4d(betx=1, bety=1, include_collective=True)
# tw_undulator.plot('x y')
# tw_undulator.plot('betx bety', 'dx dy')
# plt.show()

# The issue: When you use betx=1, bety=1, twiss4d treats the line as OPEN (non-periodic).
# For an open line, the orbit is computed from initial conditions in particle_on_co.
# If particle_on_co has zero initial conditions (x=0, px=0, y=0, py=0), the orbit will
# be zero unless there are kicks from the wiggler.
#
# Solution: Use only_orbit=True to explicitly compute the orbit, which should properly
# propagate through the wiggler and show any kicks/deviations.

env['k0l_corr1'] = 0.
env['k0l_corr2'] = 0.
env['k0l_corr3'] = 0.
env['k0l_corr4'] = 0.
env['k0sl_corr1'] = 0.
env['k0sl_corr2'] = 0.
env['k0sl_corr3'] = 0.
env['k0sl_corr4'] = 0.
env['on_wig_corr'] = 1.0

env.new('corr1', xt.Multipole, knl=['on_wig_corr * k0l_corr1'], ksl=['on_wig_corr * k0sl_corr1'])
env.new('corr2', xt.Multipole, knl=['on_wig_corr * k0l_corr2'], ksl=['on_wig_corr * k0sl_corr2'])
env.new('corr3', xt.Multipole, knl=['on_wig_corr * k0l_corr3'], ksl=['on_wig_corr * k0sl_corr3'])
env.new('corr4', xt.Multipole, knl=['on_wig_corr * k0l_corr4'], ksl=['on_wig_corr * k0sl_corr4'])

piecewise_undulator.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_wig - 0.1),
    env.place('corr4', at=l_wig - 0.02),
    ], s_tol=5e-3
)

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
        xt.TargetSet(x=0., y=0, at='wiggler_167'),
        xt.TargetSet(x=0., y=0, at='wiggler_833')
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
    radiation_method='kick_as_co'  # Use 'kick_as_co' for average energy loss, or 'full' for full computation
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