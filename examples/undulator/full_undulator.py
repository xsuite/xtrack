import xtrack as xt
import matplotlib.pyplot as plt
import xobjects as xo
import numpy as np
import pandas as pd
import time
from pathlib import Path

import sys
np.set_printoptions(threshold=sys.maxsize)

env = xt.Environment()
p0 = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9
)
env.particle_ref = p0.copy()

n_part = 1

BASE_DIR = Path(__file__).resolve().parent
filepath = BASE_DIR / "fit_parameters.csv"
#filepath = 'fit_parameters.csv'
df = pd.read_csv(filepath, index_col=['field_component', 'derivative_x', 'region_name', 's_start', 's_end', 'idx_start', 'idx_end', 'param_index'])

# work with a flat dataframe for easier masking by s
df_reset = df.reset_index()

s_starts = np.sort(df_reset['s_start'].to_numpy(dtype=np.float64))
s_ends = np.sort(df_reset['s_end'].to_numpy(dtype=np.float64))

s_boundaries = np.sort(np.unique(np.concatenate((s_starts, s_ends))))

s_start = s_boundaries[0]
s_end = s_boundaries[-1]

n_steps = 1000

def _contruct_par_table(n_steps, s_start, s_end, df_flat, multipole_order):
    """
    Construct parameter table for SplineBoris.
    
    Parameters are ordered as expected by the C code:
    - For multipole_order=3: bs_*, b_1_*, b_2_*, b_3_*, a_1_*, a_2_*, a_3_*
    - Within each group: ordered by polynomial order (0, 1, 2, 3, 4)
    - b_* maps to kn_* (normal multipole) and a_* maps to ks_* (skew multipole)
    """
    par_dicts = []
    par_table = []
    s_vals = np.linspace(float(s_start), float(s_end), n_steps)
    
    # Expected parameter order for the given multipole_order
    # C code expects: bs_*, kn_1_*, ..., kn_N_*, ks_1_*, ..., ks_N_*
    # Which maps to: bs_*, b_1_*, ..., b_N_*, a_1_*, ..., a_N_*
    expected_params = []
    # First: bs_ parameters (no multipole index)
    for poly_idx in range(5):
        expected_params.append(f"bs_{poly_idx}")
    # Second: b_ parameters (all multipoles) - these map to kn_*
    for multipole_idx in range(1, multipole_order + 1):
        for poly_idx in range(5):  # 0 to 4
            expected_params.append(f"b_{multipole_idx}_{poly_idx}")
    # Third: a_ parameters (all multipoles) - these map to ks_*
    for multipole_idx in range(1, multipole_order + 1):
        for poly_idx in range(5):  # 0 to 4
            expected_params.append(f"a_{multipole_idx}_{poly_idx}")
    
    for s_val_i in s_vals:
        # Filter rows that contain this s position
        # Only use derivative_x=0 (the field itself, not derivatives)
        mask = ((df_flat['s_start'] <= s_val_i) & 
                (df_flat['s_end'] >= s_val_i) & 
                (df_flat['derivative_x'] == 0))
        rows = df_flat[mask]
        
        # Create a dictionary from available parameters
        # Group by field_component to ensure we get all parameters from the same region
        # when possible, but allow different regions for different field components
        param_dict = {}
        if not rows.empty:
            # Sort by field_component and region to ensure consistent selection
            # when there are multiple matches, prefer the most specific region (smallest s range)
            rows_sorted = rows.copy()
            rows_sorted['region_size'] = rows_sorted['s_end'] - rows_sorted['s_start']
            rows_sorted = rows_sorted.sort_values(['field_component', 'region_size', 'param_name'])
            
            # Take the first occurrence of each param_name (should be from the smallest region)
            for _, row in rows_sorted.iterrows():
                param_name = row['param_name']
                if param_name not in param_dict:
                    param_dict[param_name] = row['param_value']
        
        # Build parameter list in expected order, filling missing values with 0
        param_values = []
        for param_name in expected_params:
            if param_name in param_dict:
                param_values.append(float(param_dict[param_name]))
            else:
                param_values.append(0.0)  # Fill missing parameters with 0
        
        par_dicts.append(param_dict)
        par_table.append(param_values)
    
    return par_dicts, par_table

multipole_order = 3

par_dicts, par_table = _contruct_par_table(n_steps, s_start, s_end, df_reset, multipole_order=multipole_order)

print(f"s_start = {s_start}, s_end = {s_end}, n_steps = {n_steps}")
# print(f"par_table[1] = {par_table[2]}")
#
#
# # print full DataFrame: all rows and columns
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
# # print only rows indexed by 'Poly_1' in level 'region_name'
# if 'region_name' in df.index.names:
#     poly1_df = df.xs('Poly_1', level='region_name')
# else:
#     poly1_df = df[df['region_name'] == 'Poly_1'] if 'region_name' in df.columns else df
# print(poly1_df.to_string())

splineboris_element = xt.SplineBoris(par_table=par_table, multipole_order=multipole_order, s_start=s_start, s_end=s_end, length=s_end-s_start, n_steps=n_steps)

env.elements['wiggler'] = splineboris_element

line = env.new_line(['wiggler'])

line.build_tracker()

start_time = time.time()
tw = line.twiss4d(betx=1, bety=1, include_collective=True)
end_time = time.time()
print(f"Time taken to compute twiss through SplineBoris: {end_time - start_time} seconds")

# tw.plot('x y')
# tw.plot('betx bety', 'dx dy')
# plt.show()


# list of n_steps wigglers;
wiggler_list = []
name_list = ['wiggler_'+str(i) for i in range(1,n_steps+1)]

s_vals = np.linspace(s_start, s_end, n_steps)
l_wig = s_end - s_start
ds = (s_end - s_start) / n_steps

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
        par_table=params_i, 
        multipole_order=multipole_order, 
        s_start=elem_s_start, 
        s_end=elem_s_end, 
        length=elem_s_end - elem_s_start,
        n_steps=1
    )
    wiggler_list.append(wiggler_i)
    env.elements[name_list[i]] = wiggler_i

piecewise_wiggler = env.new_line(components=name_list)

piecewise_wiggler.build_tracker()

start_time = time.time()
tw_new = piecewise_wiggler.twiss4d(betx=1, bety=1, include_collective=True)
end_time = time.time()
print(f"Time taken to compute twiss through list of SplineBoris: {end_time - start_time} seconds")

# tw_new.plot('x y')
# tw_new.plot('betx bety', 'dx dy')
# plt.show()

piecewise_wiggler.particle_ref = line.particle_ref

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

# Note: Use l_wig (line coordinate) for positions, not s_end (field map coordinate)
# s_end is in the field map coordinate system and can be negative or any value.
# l_wig = s_end - s_start is the physical length in the line coordinate system.

# Debug: Check the table before insertion
tt_before = piecewise_wiggler.get_table()
print(f"\nDebug: Before insertion")
print(f"  First element name: {tt_before.name[0]}")
print(f"  First element s_start (table): {tt_before['s_start', tt_before.name[0]]}")
print(f"  First element s_center (table): {tt_before['s_center', tt_before.name[0]]}")
first_elem = env.elements[tt_before.env_name[0]]
if hasattr(first_elem, 's_start'):
    print(f"  First element s_start (field map): {first_elem.s_start}")
    print(f"  First element length (element): {first_elem.length}")
print(f"  Last element s_end (table): {tt_before['s_end', tt_before.name[-2]]}")
print(f"  Line length: {piecewise_wiggler.get_length()}")
print(f"  l_wig: {l_wig}")

piecewise_wiggler.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_wig - 0.1),
    env.place('corr4', at=l_wig - 0.02),
    ], s_tol=5e-3
)

# Debug: Check the table after insertion
tt_after = piecewise_wiggler.get_table()
print(f"\nDebug: After insertion")
for corr_name in ['corr1', 'corr2', 'corr3', 'corr4']:
    if corr_name in tt_after.name:
        print(f"  {corr_name}: s_center = {tt_after['s_center', corr_name]}")

print(f"l_wig = {l_wig}")
print(f"s_start = {s_start}, s_end = {s_end}, n_steps = {n_steps}")

# Save reference to old environment before loading new one
env_wiggler = env

env = xt.load('/home/simonfan/projects/xsuite/xtrack/test_data/sls/b075_2024.09.25.madx')
line_sls = env.ring
line_no_wiggler = line_sls.copy(shallow=True)
line_off_wiggler = line_sls.copy(shallow=True)
env['ring_no_wiggler'] = line_no_wiggler
env['ring_off_wiggler'] = line_off_wiggler
line_no_wiggler.configure_bend_model(core='mat-kick-mat')
line_off_wiggler.configure_bend_model(core='mat-kick-mat')
line_sls.particle_ref = p0.copy()
line_no_wiggler.particle_ref = p0.copy()
line_off_wiggler.particle_ref = p0.copy()

# Copy all wiggler elements and correction elements to the new environment
print("Copying wiggler elements to new environment...")

# Copy variables FIRST (before elements that depend on them)
for var_name in ['k0l_corr1', 'k0l_corr2', 'k0l_corr3', 'k0l_corr4',
                 'k0sl_corr1', 'k0sl_corr2', 'k0sl_corr3', 'k0sl_corr4',
                 'on_wig_corr']:
    env[var_name] = env_wiggler[var_name]

# Copy wiggler elements
for name in name_list:
    # Directly copy the element since copy_element_from doesn't support SplineBoris
    env.elements[name] = env_wiggler.elements[name].copy()

# Create offset wiggler elements with x_off = 0.0005 m
x_off = 0.0005  # 0.0005 m offset in x
name_list_off = [f'wiggler_off_{i}' for i in range(1, n_steps+1)]
print(f"Creating offset wiggler elements with x_off = {x_off} m...")
for i, name in enumerate(name_list):
    orig_elem = env_wiggler.elements[name]
    # Create new element with offset
    wiggler_off_i = xt.SplineBoris(
        par_table=orig_elem.par_table,
        multipole_order=orig_elem.multipole_order,
        s_start=orig_elem.s_start,
        s_end=orig_elem.s_end,
        length=orig_elem.length,
        n_steps=orig_elem.n_steps,
        x_off=0.0,
        y_off=x_off
    )
    env.elements[name_list_off[i]] = wiggler_off_i

# Copy correction elements (they depend on the variables, so copy after variables)
for corr_name in ['corr1', 'corr2', 'corr3', 'corr4']:
    env.copy_element_from(corr_name, env_wiggler)

# Recreate piecewise_wiggler in the new environment
piecewise_wiggler = env.new_line(components=name_list)
piecewise_wiggler.build_tracker()
piecewise_wiggler.particle_ref = p0.copy()

# Re-insert correction elements
piecewise_wiggler.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_wig - 0.1),
    env.place('corr4', at=l_wig - 0.02),
    ], s_tol=5e-3
)

# Create offset wiggler line
piecewise_wiggler_off = env.new_line(components=name_list_off)
piecewise_wiggler_off.build_tracker()
piecewise_wiggler_off.particle_ref = p0.copy()

# Re-insert correction elements into offset wiggler
piecewise_wiggler_off.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_wig - 0.1),
    env.place('corr4', at=l_wig - 0.02),
    ], s_tol=5e-3
)

# tw_sls = line_sls.twiss4d(betx=1, bety=1, include_collective=True)

# tw_sls.plot('x y')
# tw_sls.plot('betx bety', 'dx dy')
# plt.show()

# To compute the kicks
opt = piecewise_wiggler.match(
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

time_start = time.time()
tw_corr_wig = piecewise_wiggler.twiss4d(betx=1, bety=1, include_collective=True)
time_end = time.time()
print(f"Time taken to compute twiss through piecewise_wiggler: {time_end - time_start} seconds")


tw_corr_wig.plot('x y')
tw_corr_wig.plot('betx bety', 'dx dy')
plt.show()

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

print(f"Time estimated for total number of wigglers: {len(wiggler_places) * (time_end - time_start)} seconds")

tt = line_sls.get_table()
for wig_place in wiggler_places:
    print(f"Inserting piecewise_wiggler {wig_place} at {tt['s', wig_place]}")
    line_sls.insert(piecewise_wiggler, anchor='start', at=tt['s', wig_place])

# Insert offset wiggler into line_off_wiggler
for wig_place in wiggler_places:
    print(f"Inserting piecewise_wiggler_off {wig_place} at {tt['s', wig_place]}")
    line_off_wiggler.insert(piecewise_wiggler_off, anchor='start', at=tt['s', wig_place])

time_start = time.time()
tw_sls = line_sls.twiss4d(radiation_integrals=True)
time_end = time.time()
print(f"Time taken to compute twiss through full line: {time_end - time_start} seconds")

time_start = time.time()
tw_no_wiggler = line_no_wiggler.twiss4d(radiation_integrals=True)
time_end = time.time()
print(f"Time taken to compute twiss through full line without wiggler: {time_end - time_start} seconds")

tw_sls.plot('x y')
tw_sls.plot('betx bety', 'dx dy')
plt.show()

time_start = time.time()
tw_off_wiggler = line_off_wiggler.twiss4d(radiation_integrals=True)
time_end = time.time()
print(f"Time taken to compute twiss through full line with offset wiggler: {time_end - time_start} seconds")

# print(tw_sls.cols.names)
# print(tw_no_wiggler.cols.name)

#['name', 's', 'x', 'px', 'y', 'py', 'zeta', 'delta', 'ptau', 'W_matrix', 'kin_px', 'kin_py', 'kin_ps', 'kin_xprime',
# 'kin_yprime', 'env_name', 'betx', 'bety', 'alfx', 'alfy', 'gamx', 'gamy', 'dx', 'dpx', 'dy', 'dpy', 'dx_zeta', 'dpx_zeta',
# 'dy_zeta', 'dpy_zeta', 'betx1', 'bety1', 'betx2', 'bety2', 'alfx1', 'alfy1', 'alfx2', 'alfy2', 'gamx1', 'gamy1',
# 'gamx2', 'gamy2', 'mux', 'muy', 'muzeta', 'nux', 'nuy', 'nuzeta', 'phix', 'phiy', 'phizeta', 'dmux', 'dmuy', 'dzeta',
# 'bx_chrom', 'by_chrom', 'ax_chrom', 'ay_chrom', 'wx_chrom', 'wy_chrom', 'ddx', 'ddpx', 'ddy', 'ddpy', 'c_minus_re',
# 'c_minus_im', 'c_r1', 'c_r2', 'c_phi1', 'c_phi2', 'k0l', 'k1l', 'k2l', 'k3l', 'k4l', 'k5l', 'k0sl', 'k1sl', 'k2sl',
# 'k3sl', 'k4sl', 'k5sl', 'angle_rad', 'rot_s_rad', 'hkick', 'vkick', 'ks', 'length', '_angle_force_body', 'element_type', 'isthick', 'parent_name']

# Write .txt file that contains a table comparing:
# - Tunes
# - Chromaticity
# - Energy loss per turn
# - Radiation damping constants per second
# - Radiation partition numbers
# - Equilibrium emittances
# - Damping constants per second

# With undulator:
# Partition numbers
J_x_with_wiggler = tw_sls.rad_int_partition_number_x
J_y_with_wiggler = tw_sls.rad_int_partition_number_y
J_zeta_with_wiggler = tw_sls.rad_int_partition_number_zeta

# Damping constants per second
alpha_x_with_wiggler = tw_sls.rad_int_damping_constant_x_s
alpha_y_with_wiggler = tw_sls.rad_int_damping_constant_y_s
alpha_zeta_with_wiggler = tw_sls.rad_int_damping_constant_zeta_s

# Equilibrium emittances
eq_gemitt_x_with_wiggler = tw_sls.rad_int_eq_gemitt_x
eq_gemitt_y_with_wiggler = tw_sls.rad_int_eq_gemitt_y
eq_gemitt_zeta_with_wiggler = tw_sls.rad_int_eq_gemitt_zeta

# Chromaticity
chrom_x_with_wiggler = tw_sls.dqx
chrom_y_with_wiggler = tw_sls.dqy


# Tunes
tune_x_with_wiggler = tw_sls.qx
tune_y_with_wiggler = tw_sls.qy
tune_s_with_wiggler = tw_sls.qs

# Energy loss per turn
eneloss_turn_with_wiggler = tw_sls.rad_int_eneloss_turn

# Without undulator:
# Partition numbers
J_x_no_wiggler = tw_no_wiggler.rad_int_partition_number_x
J_y_no_wiggler = tw_no_wiggler.rad_int_partition_number_y
J_zeta_no_wiggler = tw_no_wiggler.rad_int_partition_number_zeta

# Damping constants per second
alpha_x_no_wiggler = tw_no_wiggler.rad_int_damping_constant_x_s
alpha_y_no_wiggler = tw_no_wiggler.rad_int_damping_constant_y_s
alpha_zeta_no_wiggler = tw_no_wiggler.rad_int_damping_constant_zeta_s

# Equilibrium emittances
eq_gemitt_x_no_wiggler = tw_no_wiggler.rad_int_eq_gemitt_x
eq_gemitt_y_no_wiggler = tw_no_wiggler.rad_int_eq_gemitt_y
eq_gemitt_zeta_no_wiggler = tw_no_wiggler.rad_int_eq_gemitt_zeta

# Chromaticity
chrom_x_no_wiggler = tw_no_wiggler.dqx
chrom_y_no_wiggler = tw_no_wiggler.dqy

# Tunes
tune_x_no_wiggler = tw_no_wiggler.qx
tune_y_no_wiggler = tw_no_wiggler.qy
tune_s_no_wiggler = tw_no_wiggler.qs

# Energy loss per turn
eneloss_turn_no_wiggler = tw_no_wiggler.rad_int_eneloss_turn

# Offset wiggler:
J_x_off_wiggler = tw_off_wiggler.rad_int_partition_number_x
J_y_off_wiggler = tw_off_wiggler.rad_int_partition_number_y
J_zeta_off_wiggler = tw_off_wiggler.rad_int_partition_number_zeta

alpha_x_off_wiggler = tw_off_wiggler.rad_int_damping_constant_x_s
alpha_y_off_wiggler = tw_off_wiggler.rad_int_damping_constant_y_s
alpha_zeta_off_wiggler = tw_off_wiggler.rad_int_damping_constant_zeta_s

eq_gemitt_x_off_wiggler = tw_off_wiggler.rad_int_eq_gemitt_x
eq_gemitt_y_off_wiggler = tw_off_wiggler.rad_int_eq_gemitt_y
eq_gemitt_zeta_off_wiggler = tw_off_wiggler.rad_int_eq_gemitt_zeta

chrom_x_off_wiggler = tw_off_wiggler.dqx
chrom_y_off_wiggler = tw_off_wiggler.dqy

tune_x_off_wiggler = tw_off_wiggler.qx
tune_y_off_wiggler = tw_off_wiggler.qy
tune_s_off_wiggler = tw_off_wiggler.qs

eneloss_turn_off_wiggler = tw_off_wiggler.rad_int_eneloss_turn

# Write to .txt file
with open('full_undulator.txt', 'w') as f:
    f.write(f"With undulator:\n")
    f.write(f"  Partition numbers: J_x = {J_x_with_wiggler:.4f}, J_y = {J_y_with_wiggler:.4f}, J_zeta = {J_zeta_with_wiggler:.4f}\n")
    f.write(f"  Damping constants per second: alpha_x = {alpha_x_with_wiggler:.4f}, alpha_y = {alpha_y_with_wiggler:.4f}, alpha_zeta = {alpha_zeta_with_wiggler:.4f}\n")
    f.write(f"  Equilibrium emittances: eq_gemitt_x = {eq_gemitt_x_with_wiggler:.4f}, eq_gemitt_y = {eq_gemitt_y_with_wiggler:.4f}, eq_gemitt_zeta = {eq_gemitt_zeta_with_wiggler:.4f}\n")
    f.write(f"  Chromaticity: chrom_x = {chrom_x_with_wiggler:.4f}, chrom_y = {chrom_y_with_wiggler:.4f}\n")
    f.write(f"  Tunes: tune_x = {tune_x_with_wiggler:.4f}, tune_y = {tune_y_with_wiggler:.4f}, tune_s = {tune_s_with_wiggler:.4f}\n")
    f.write(f"  Energy loss per turn: eneloss_turn = {eneloss_turn_with_wiggler:.4f}\n")
    f.write(f"Without undulator:\n")
    f.write(f"  Partition numbers: J_x = {J_x_no_wiggler:.4f}, J_y = {J_y_no_wiggler:.4f}, J_zeta = {J_zeta_no_wiggler:.4f}\n")
    f.write(f"  Damping constants per second: alpha_x = {alpha_x_no_wiggler:.4f}, alpha_y = {alpha_y_no_wiggler:.4f}, alpha_zeta = {alpha_zeta_no_wiggler:.4f}\n")
    f.write(f"  Equilibrium emittances: eq_gemitt_x = {eq_gemitt_x_no_wiggler:.4f}, eq_gemitt_y = {eq_gemitt_y_no_wiggler:.4f}, eq_gemitt_zeta = {eq_gemitt_zeta_no_wiggler:.4f}\n")
    f.write(f"  Chromaticity: chrom_x = {chrom_x_no_wiggler:.4f}, chrom_y = {chrom_y_no_wiggler:.4f}\n")
    f.write(f"  Tunes: tune_x = {tune_x_no_wiggler:.4f}, tune_y = {tune_y_no_wiggler:.4f}, tune_s = {tune_s_no_wiggler:.4f}\n")
    f.write(f"  Energy loss per turn: eneloss_turn = {eneloss_turn_no_wiggler:.4f}\n")

# Print everything:
print(f"With undulator:")
print(f"  Partition numbers: J_x = {J_x_with_wiggler:.4f}, J_y = {J_y_with_wiggler:.4f}, J_zeta = {J_zeta_with_wiggler:.4f}")
print(f"  Damping constants per second: alpha_x = {alpha_x_with_wiggler:.4f}, alpha_y = {alpha_y_with_wiggler:.4f}, alpha_zeta = {alpha_zeta_with_wiggler:.4f}")
print(f"  Equilibrium emittances: eq_gemitt_x = {eq_gemitt_x_with_wiggler:.4f}, eq_gemitt_y = {eq_gemitt_y_with_wiggler:.4f}, eq_gemitt_zeta = {eq_gemitt_zeta_with_wiggler:.4f}")
print(f"  Chromaticity: chrom_x = {chrom_x_with_wiggler:.4f}, chrom_y = {chrom_y_with_wiggler:.4f}")
print(f"  Tunes: tune_x = {tune_x_with_wiggler:.4f}, tune_y = {tune_y_with_wiggler:.4f}, tune_s = {tune_s_with_wiggler:.4f}")
print(f"  Energy loss per turn: eneloss_turn = {eneloss_turn_with_wiggler:.4f}\n")

print(f"Without undulator:")
print(f"  Partition numbers: J_x = {J_x_no_wiggler:.4f}, J_y = {J_y_no_wiggler:.4f}, J_zeta = {J_zeta_no_wiggler:.4f}")
print(f"  Damping constants per second: alpha_x = {alpha_x_no_wiggler:.4f}, alpha_y = {alpha_y_no_wiggler:.4f}, alpha_zeta = {alpha_zeta_no_wiggler:.4f}")
print(f"  Equilibrium emittances: eq_gemitt_x = {eq_gemitt_x_no_wiggler:.4f}, eq_gemitt_y = {eq_gemitt_y_no_wiggler:.4f}, eq_gemitt_zeta = {eq_gemitt_zeta_no_wiggler:.4f}")
print(f"  Chromaticity: chrom_x = {chrom_x_no_wiggler:.4f}, chrom_y = {chrom_y_no_wiggler:.4f}")
print(f"  Tunes: tune_x = {tune_x_no_wiggler:.4f}, tune_y = {tune_y_no_wiggler:.4f}, tune_s = {tune_s_no_wiggler:.4f}")
print(f"  Energy loss per turn: eneloss_turn = {eneloss_turn_no_wiggler:.4f}\n")

print(f"Offset wiggler (x_off = 0.0005 m):")
print(f"  Partition numbers: J_x = {J_x_off_wiggler:.4f}, J_y = {J_y_off_wiggler:.4f}, J_zeta = {J_zeta_off_wiggler:.4f}")
print(f"  Damping constants per second: alpha_x = {alpha_x_off_wiggler:.4f}, alpha_y = {alpha_y_off_wiggler:.4f}, alpha_zeta = {alpha_zeta_off_wiggler:.4f}")
print(f"  Equilibrium emittances: eq_gemitt_x = {eq_gemitt_x_off_wiggler:.4f}, eq_gemitt_y = {eq_gemitt_y_off_wiggler:.4f}, eq_gemitt_zeta = {eq_gemitt_zeta_off_wiggler:.4f}")
print(f"  Chromaticity: chrom_x = {chrom_x_off_wiggler:.4f}, chrom_y = {chrom_y_off_wiggler:.4f}")
print(f"  Tunes: tune_x = {tune_x_off_wiggler:.4f}, tune_y = {tune_y_off_wiggler:.4f}, tune_s = {tune_s_off_wiggler:.4f}")
print(f"  Energy loss per turn: eneloss_turn = {eneloss_turn_off_wiggler:.4f}")