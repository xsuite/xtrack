"""
Spin tracking with undulators and radiation.

This script builds and corrects an SLS undulator from fitted Spline4 field
data, saves it to JSON, computes twiss with spin tracking and radiation, and
displays results.
"""

import xtrack as xt
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xtrack._temp.splineboris.field_fitter import FieldFitter


multipole_order = 3

# Particle reference
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)

env = xt.Environment()

BASE_DIR = Path(__file__).resolve().parent
UNDULATOR_JSON = BASE_DIR / 'sls_undulator.json'

# Load the raw field map data from shared test_data
field_map_path = BASE_DIR.parent.parent / "test_data" / "sls" / "undulator_field_map.txt"
df_raw_data = pd.read_csv(
    field_map_path,
    sep=r"\s+",
    header=None,
    names=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
).set_index(["X", "Y", "Z"])

# Distance unit in meters (the dataset uses mm, so 1 mm = 0.001 m)
distance_unit = 0.001

field_fitter = FieldFitter(
    raw_data=df_raw_data,
    xy_point=(0, 0),
    distance_unit=distance_unit,
    min_region_size=10,
    deg=multipole_order-1,
)

# Get the fitted field data for each longitudinal interval. Tracking settings
# are intentionally applied below when constructing the SplineBoris elements.
spline_data = field_fitter.get_spline_data(
    multipole_order=multipole_order,
)

# Build and register the SplineBoris elements explicitly.
undulator_element_names = []
for ii, piece in enumerate(spline_data):
    element_name = f'undulator_splineboris_{ii}'

    # Match the field-map resolution: one Boris step per interval
    # between adjacent data points in this piece.
    nn_steps = max(1, piece['idx_end'] - piece['idx_start'])

    env.elements[element_name] = xt.SplineBoris(
        length=piece['s_end'] - piece['s_start'],
        n_steps=nn_steps,
        bs=piece['bs'],
        bx=piece['bx'],
        by=piece['by'],
    )
    undulator_element_names.append(element_name)

undulator = env.new_line(components=undulator_element_names)
l_wig = undulator.get_length()

undulator.particle_ref = p0.copy()

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
undulator.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_wig - 0.1),
    env.place('corr4', at=l_wig - 0.02),
], s_tol=5e-3)

# Matching targets use corrector element names for intermediate positions
opt = undulator.match(
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

undulator.particle_ref.anomalous_magnetic_moment = 0.00115965218128

# Save the corrected undulator before enabling radiation below. Example 005
# reloads this file and inserts the undulator into the SLS ring.
undulator.to_json(UNDULATOR_JSON)

tw_undulator_corr_spin = undulator.twiss4d(
    betx=1, bety=1,
    spin=True,
    spin_x=0.5, spin_y=0.25, spin_z=0.25
    )
tw_undulator_corr_spin.plot('x y')
tw_undulator_corr_spin.plot('betx bety', 'dx dy')
tw_undulator_corr_spin.plot('spin_x')
tw_undulator_corr_spin.plot('spin_y')
tw_undulator_corr_spin.plot('spin_z')

plt.show()

# Enable radiation on all elements
undulator.configure_radiation(model='mean')

# Run twiss4d including radiation effects (average energy loss)
tw_undulator_corr_spin_rad = undulator.twiss4d(
    betx=1, bety=1,
    include_collective=True,
    spin=True,
    spin_x=0.5, spin_y=0.25, spin_z=0.25,
    radiation_method='full' 
)

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
