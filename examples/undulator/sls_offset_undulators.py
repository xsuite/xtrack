"""
SLS simulation with undulators.

This script loads the SLS MADX file, imports matched wiggler from construct_undulator,
inserts wigglers at 11 locations, computes twiss with radiation integrals, and prints results.
"""

import xtrack as xt
from pathlib import Path
from construct_undulator import _contruct_par_table
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

multipole_order = 3

n_steps = 1000

# Particle reference
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)

# Load SLS MADX file
madx_file = Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls' / 'b075_2024.09.25.madx'
env = xt.load(str(madx_file))
line_offset = env.ring

# Configure bend model
line_offset.configure_bend_model(core='mat-kick-mat')

# Set particle reference
line_offset.particle_ref = p0.copy()

BASE_DIR = Path(__file__).resolve().parent
# Use the field-fit parameters produced by the spline fitter example
filepath = (
    BASE_DIR
    / "spline_fitter"
    / "field_maps"
    / "field_fit_pars.csv"
)

df = pd.read_csv(filepath, index_col=['field_component', 'derivative_x', 'region_name', 's_start', 's_end', 'idx_start', 'idx_end', 'param_index'])

# work with a flat dataframe for easier masking by s
df_reset = df.reset_index()

# # Configure pandas to display full DataFrame without truncation
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)

# # Print the full DataFrame
# print(df_reset)

s_starts = np.sort(df_reset['s_start'].to_numpy(dtype=np.float64))
s_ends = np.sort(df_reset['s_end'].to_numpy(dtype=np.float64))

s_boundaries = np.sort(np.unique(np.concatenate((s_starts, s_ends))))

s_start = s_boundaries[0]
s_end = s_boundaries[-1]

par_dicts, par_table = _contruct_par_table(n_steps, s_start, s_end, df_reset, multipole_order=multipole_order)

# list of n_steps wigglers;
wiggler_list = []
name_list = ['wiggler_'+str(i) for i in range(1, n_steps+1)]

s_vals = np.linspace(s_start, s_end, n_steps)
l_wig = s_end - s_start
ds = (s_end - s_start) / n_steps

# Define offsets early so they can be passed to constructor
x_off = 0  # 0.0005 m offset in x
y_off = 5e-4  # 0.0005 m offset in y

for i in range(n_steps):
    # params should be a 2D array: [[param1, param2, ...]] for n_steps=1
    params_i = [par_table[i]]
    s_val_i = s_vals[i]
    
    # For each single-step element, s_start and s_end should define the range
    # in the field map that this step covers. Use a small interval around s_val_i.
    # The s coordinates here are in the field map coordkinate system (can be negative).
    elem_s_start = s_val_i - ds/2
    elem_s_end = s_val_i + ds/2
    
    wiggler_i = xt.SplineBoris(
        params=params_i,
        multipole_order=multipole_order, 
        s_start=elem_s_start, 
        s_end=elem_s_end, 
        length=elem_s_end - elem_s_start,
        n_steps=1,
        shift_x=x_off,  # Set shift values in constructor
        shift_y=y_off
    )
    wiggler_list.append(wiggler_i)
    env.elements[name_list[i]] = wiggler_i

piecewise_undulator = env.new_line(components=name_list)

piecewise_undulator.build_tracker()

piecewise_undulator.particle_ref = p0.copy()

tw_undulator = piecewise_undulator.twiss4d(betx=1, bety=1, include_collective=True)
tw_undulator.plot('x y')
tw_undulator.plot('betx bety', 'dx dy')
plt.show()

piecewise_undulator.discard_tracker()

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


# tw_undulator_corr = piecewise_undulator.twiss4d(betx=1, bety=1, include_collective=True)
# tw_undulator_corr.plot('x y')
# tw_undulator_corr.plot('betx bety', 'dx dy')
# plt.show()

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

tt = line_offset.get_table()
for wig_place in wiggler_places:
    print(f"Inserting piecewise_undulator {wig_place} at {tt['s', wig_place]}")
    line_offset.insert(piecewise_undulator, anchor='start', at=tt['s', wig_place])

line_offset.build_tracker()

tw_offset = line_offset.twiss4d(radiation_integrals=True)

# Plotting:
import matplotlib.pyplot as plt
plt.close('all')
tw_offset.plot('x y')
tw_offset.plot('betx bety', 'dx dy')
tw_offset.plot('betx2 bety2')
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
print("SLS WITH OFFSET UNDULATORS")
print("=" * 80)
print(f"Tunes:")
print(f"  qx = {tw_offset.qx:.4e}")
print(f"  qy = {tw_offset.qy:.4e}")
print(f"  qs = {tw_offset.qs:.4e}")
print()
print(f"Chromaticity:")
print(f"  dqx = {tw_offset.dqx:.4e}")
print(f"  dqy = {tw_offset.dqy:.4e}")
print()
print(f"Partition numbers:")
print(f"  J_x = {tw_offset.rad_int_partition_number_x:.4e}")
print(f"  J_y = {tw_offset.rad_int_partition_number_y:.4e}")
print(f"  J_zeta = {tw_offset.rad_int_partition_number_zeta:.4e}")
print()
print(f"Damping constants per second:")
print(f"  alpha_x = {tw_offset.rad_int_damping_constant_x_s:.4e}")
print(f"  alpha_y = {tw_offset.rad_int_damping_constant_y_s:.4e}")
print(f"  alpha_zeta = {tw_offset.rad_int_damping_constant_zeta_s:.4e}")
print()
print(f"Equilibrium emittances:")
print(f"  eq_gemitt_x = {tw_offset.rad_int_eq_gemitt_x:.4e}")
print(f"  eq_gemitt_y = {tw_offset.rad_int_eq_gemitt_y:.4e}")
print(f"  eq_gemitt_zeta = {tw_offset.rad_int_eq_gemitt_zeta:.4e}")
print()
print(f"Energy loss per turn: {tw_offset.rad_int_eneloss_turn:.4e} eV")
print("=" * 80)

# Write results to file
output_dir = Path("/home/simonfan/cernbox/Documents/Presentations/Section_Meeting_Undulators")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "SLS_WITH_OFFSET_UNDULATORS.txt"

with open(output_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("SLS WITH OFFSET UNDULATORS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Tunes:\n")
    f.write(f"  qx = {tw_offset.qx:.4e}\n")
    f.write(f"  qy = {tw_offset.qy:.4e}\n")
    f.write(f"  qs = {tw_offset.qs:.4e}\n")
    f.write("\n")
    f.write(f"Chromaticity:\n")
    f.write(f"  dqx = {tw_offset.dqx:.4e}\n")
    f.write(f"  dqy = {tw_offset.dqy:.4e}\n")
    f.write("\n")
    f.write(f"Partition numbers:\n")
    f.write(f"  J_x = {tw_offset.rad_int_partition_number_x:.4e}\n")
    f.write(f"  J_y = {tw_offset.rad_int_partition_number_y:.4e}\n")
    f.write(f"  J_zeta = {tw_offset.rad_int_partition_number_zeta:.4e}\n")
    f.write("\n")
    f.write(f"Damping constants per second:\n")
    f.write(f"  alpha_x = {tw_offset.rad_int_damping_constant_x_s:.4e}\n")
    f.write(f"  alpha_y = {tw_offset.rad_int_damping_constant_y_s:.4e}\n")
    f.write(f"  alpha_zeta = {tw_offset.rad_int_damping_constant_zeta_s:.4e}\n")
    f.write("\n")
    f.write(f"Equilibrium emittances:\n")
    f.write(f"  eq_gemitt_x = {tw_offset.rad_int_eq_gemitt_x:.4e}\n")
    f.write(f"  eq_gemitt_y = {tw_offset.rad_int_eq_gemitt_y:.4e}\n")
    f.write(f"  eq_gemitt_zeta = {tw_offset.rad_int_eq_gemitt_zeta:.4e}\n")
    f.write("\n")
    f.write(f"Energy loss per turn: {tw_offset.rad_int_eneloss_turn:.4e} eV\n")
    f.write("=" * 80 + "\n")