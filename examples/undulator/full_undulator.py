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
    Construct parameter table for BPMethElement.
    
    Parameters are ordered as expected by the C code:
    - For multipole_order=3: a_1_*, a_2_*, a_3_*, b_1_*, b_2_*, b_3_*, bs_*
    - Within each group: ordered by polynomial order (0, 1, 2, 3, 4)
    """
    par_dicts = []
    par_table = []
    s_vals = np.linspace(float(s_start), float(s_end), n_steps)
    
    # Expected parameter order for the given multipole_order
    # Format: prefix_multipole_poly (e.g., a_1_0, a_2_3, bs_0)
    expected_params = []
    for prefix in ['a_', 'b_']:
        for multipole_idx in range(1, multipole_order + 1):
            for poly_idx in range(5):  # 0 to 4
                expected_params.append(f"{prefix}{multipole_idx}_{poly_idx}")
    # Add bs_ parameters (no multipole index)
    for poly_idx in range(5):
        expected_params.append(f"bs_{poly_idx}")
    
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

multipole_order = 3 # The CSV only contains multipole_order=1 parameters (a_1_*, b_1_*, bs_*)

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

bpmeth_element = xt.BPMethElement(params=par_table, multipole_order=multipole_order, s_start=s_start, s_end=s_end, length=s_end-s_start, n_steps=n_steps)

env.elements['wiggler'] = bpmeth_element

line = env.new_line(['wiggler'])

line.build_tracker()

start_time = time.time()
tw = line.twiss4d(betx=1, bety=1, include_collective=True)
end_time = time.time()
print(f"Time taken to compute twiss through BPMethElement: {end_time - start_time} seconds")

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
    
    wiggler_i = xt.BPMethElement(
        params=params_i, 
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
print(f"Time taken to compute twiss through list of BPMethElements: {end_time - start_time} seconds")

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
env['ring_no_wiggler'] = line_sls.copy(shallow=True)
line_sls.configure_bend_model(core='mat-kick-mat')
line_sls.particle_ref = p0.copy()
line_no_wiggler.particle_ref = p0.copy()

# Copy all wiggler elements and correction elements to the new environment
print("Copying wiggler elements to new environment...")

# Copy variables FIRST (before elements that depend on them)
for var_name in ['k0l_corr1', 'k0l_corr2', 'k0l_corr3', 'k0l_corr4',
                 'k0sl_corr1', 'k0sl_corr2', 'k0sl_corr3', 'k0sl_corr4',
                 'on_wig_corr']:
    env[var_name] = env_wiggler[var_name]

# Copy wiggler elements
for name in name_list:
    # Directly copy the element since copy_element_from doesn't support BPMethElement
    env.elements[name] = env_wiggler.elements[name].copy()

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

time_start = time.time()
tw_sls = line_sls.twiss4d(include_collective=True)
time_end = time.time()
print(f"Time taken to compute twiss through full line: {time_end - time_start} seconds")

time_start = time.time()
tw_no_wiggler = line_no_wiggler.twiss4d(include_collective=True)
time_end = time.time()
print(f"Time taken to compute twiss through full line without wiggler: {time_end - time_start} seconds")

tw_sls.plot('x y')
tw_sls.plot('betx bety', 'dx dy')
plt.show()