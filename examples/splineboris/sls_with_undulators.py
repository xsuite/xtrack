"""
SLS simulation with undulators.

This script loads the SLS MADX file, imports matched wiggler from construct_undulator,
inserts wigglers at 11 locations, computes twiss with radiation integrals, and prints results.
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


tw_undulator_corr = piecewise_undulator.twiss4d(betx=1, bety=1, include_collective=True)
tw_undulator_corr.plot('x y')
tw_undulator_corr.plot('betx bety', 'dx dy')
plt.show()

piecewise_undulator.discard_tracker()

wiggler_places = [
    'ars02_uind_0500_1',
    'ars03_uind_0380_1',
    'ars04_uind_0500_1',
    'ars05_uind_0650_1',
    'ars06_uind_0500_1',
    'ars07_uind_0200_1',
    'ars08_uind_0500_1',
    'ars09_uind_0790_1',
    'ars11_uind_0210_1',
    'ars11_uind_0610_1',
    'ars12_uind_0500_1',
]

tt = line_sls.get_table()
for wig_place in wiggler_places:
    print(f"Inserting piecewise_undulator {wig_place} at {tt['s', wig_place]}")
    line_sls.insert(piecewise_undulator, anchor='start', at=tt['s', wig_place])

line_sls.build_tracker()

tw_sls = line_sls.twiss4d(radiation_integrals=True)

# Plotting:
import matplotlib.pyplot as plt
plt.close('all')
tw_sls.plot('x y')
tw_sls.plot('betx bety', 'dx dy')
tw_sls.plot('betx2 bety2')
plt.show()

#['name', 's', 'x', 'px', 'y', 'py', 'zeta', 'delta', 'ptau', 'W_matrix', 'kin_px', 'kin_py', 'kin_ps', 'kin_xprime',
# 'kin_yprime', 'env_name', 'betx', 'bety', 'alfx', 'alfy', 'gamx', 'gamy', 'dx', 'dpx', 'dy', 'dpy', 'dx_zeta', 'dpx_zeta',
# 'dy_zeta', 'dpy_zeta', 'betx1', 'bety1', 'betx2', 'bety2', 'alfx1', 'alfy1', 'alfx2', 'alfy2', 'gamx1', 'gamy1',
# 'gamx2', 'gamy2', 'mux', 'muy', 'muzeta', 'nux', 'nuy', 'nuzeta', 'phix', 'phiy', 'phizeta', 'dmux', 'dmuy', 'dzeta',
# 'bx_chrom', 'by_chrom', 'ax_chrom', 'ay_chrom', 'wx_chrom', 'wy_chrom', 'ddx', 'ddpx', 'ddy', 'ddpy', 'c_minus_re',
# 'c_minus_im', 'c_r1', 'c_r2', 'c_phi1', 'c_phi2', 'k0l', 'k1l', 'k2l', 'k3l', 'k4l', 'k5l', 'k0sl', 'k1sl', 'k2sl',
# 'k3sl', 'k4sl', 'k5sl', 'angle_rad', 'rot_s_rad', 'hkick', 'vkick', 'ks', 'length', '_angle_force_body', 'element_type', 'isthick', 'parent_name']

# Extract and print results
print("=" * 80)
print("SLS WITH UNDULATORS")
print("=" * 80)
print(f"Tunes:")
print(f"  qx = {tw_sls.qx:.4e}")
print(f"  qy = {tw_sls.qy:.4e}")
print(f"  qs = {tw_sls.qs:.4e}")
print()
print(f"Chromaticity:")
print(f"  dqx = {tw_sls.dqx:.4e}")
print(f"  dqy = {tw_sls.dqy:.4e}")
print()
print(f"Partition numbers:")
print(f"  J_x = {tw_sls.rad_int_partition_number_x:.4e}")
print(f"  J_y = {tw_sls.rad_int_partition_number_y:.4e}")
print(f"  J_zeta = {tw_sls.rad_int_partition_number_zeta:.4e}")
print()
print(f"Damping constants per second:")
print(f"  alpha_x = {tw_sls.rad_int_damping_constant_x_s:.4e}")
print(f"  alpha_y = {tw_sls.rad_int_damping_constant_y_s:.4e}")
print(f"  alpha_zeta = {tw_sls.rad_int_damping_constant_zeta_s:.4e}")
print()
print(f"Equilibrium emittances:")
print(f"  eq_gemitt_x = {tw_sls.rad_int_eq_gemitt_x:.4e}")
print(f"  eq_gemitt_y = {tw_sls.rad_int_eq_gemitt_y:.4e}")
print(f"  eq_gemitt_zeta = {tw_sls.rad_int_eq_gemitt_zeta:.4e}")
print()
print(f"Energy loss per turn: {tw_sls.rad_int_eneloss_turn:.4e} eV")
print()
print(f"C^-: {tw_sls.c_minus:.4e}")
print()
print("=" * 80)

# Write results to file
output_dir = Path("/home/simonfan/cernbox/Documents/Presentations/Section_Meeting_Undulators")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "SLS_WITH_UNDULATORS.txt"

with open(output_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("SLS WITH UNDULATORS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Tunes:\n")
    f.write(f"  qx = {tw_sls.qx:.4e}\n")
    f.write(f"  qy = {tw_sls.qy:.4e}\n")
    f.write(f"  qs = {tw_sls.qs:.4e}\n")
    f.write("\n")
    f.write(f"Chromaticity:\n")
    f.write(f"  dqx = {tw_sls.dqx:.4e}\n")
    f.write(f"  dqy = {tw_sls.dqy:.4e}\n")
    f.write("\n")
    f.write(f"Partition numbers:\n")
    f.write(f"  J_x = {tw_sls.rad_int_partition_number_x:.4e}\n")
    f.write(f"  J_y = {tw_sls.rad_int_partition_number_y:.4e}\n")
    f.write(f"  J_zeta = {tw_sls.rad_int_partition_number_zeta:.4e}\n")
    f.write("\n")
    f.write(f"Damping constants per second:\n")
    f.write(f"  alpha_x = {tw_sls.rad_int_damping_constant_x_s:.4e}\n")
    f.write(f"  alpha_y = {tw_sls.rad_int_damping_constant_y_s:.4e}\n")
    f.write(f"  alpha_zeta = {tw_sls.rad_int_damping_constant_zeta_s:.4e}\n")
    f.write("\n")
    f.write(f"Equilibrium emittances:\n")
    f.write(f"  eq_gemitt_x = {tw_sls.rad_int_eq_gemitt_x:.4e}\n")
    f.write(f"  eq_gemitt_y = {tw_sls.rad_int_eq_gemitt_y:.4e}\n")
    f.write(f"  eq_gemitt_zeta = {tw_sls.rad_int_eq_gemitt_zeta:.4e}\n")
    f.write("\n")
    f.write(f"Energy loss per turn: {tw_sls.rad_int_eneloss_turn:.4e} eV\n")
    f.write("=" * 80 + "\n")