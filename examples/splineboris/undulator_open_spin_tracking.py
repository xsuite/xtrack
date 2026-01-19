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

print(piecewise_undulator.particle_ref.anomalous_magnetic_moment)

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

# print(tw_undulator_corr_spin.spin_polarization_eq)