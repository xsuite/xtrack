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
env.particle_ref = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9
)

# So given this dataframe, I'd like to do the following.
# - The data frame is indexed by field component, derivative and region start. The region start may differ per field component or derivative.
# - I'd like to create one flattened array that contains

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

tw.plot('x y')
tw.plot('betx bety', 'dx dy')
plt.show()


# list of n_steps wigglers;
list = []
name_list = ['wiggler_'+str(i) for i in range(1,n_steps+1)]

s_vals = np.linspace(s_start, s_end, n_steps)
ds = (s_end - s_start) / n_steps

for i in range(n_steps):
    # params should be a 2D array: [[param1, param2, ...]] for n_steps=1
    params_i = [par_table[i]]
    s_val_i = s_vals[i]
    
    # For each single-step element, s_start and s_end should define the range
    # in the field map that this step covers. Use a small interval around s_val_i.
    # The s coordinates here are in the field map coordinate system (can be negative).
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
    list.append(wiggler_i)
    env.elements[name_list[i]] = wiggler_i

new_line = env.new_line(components=name_list)

new_line.build_tracker()

start_time = time.time()
tw_new = new_line.twiss4d(betx=1, bety=1, include_collective=True)
end_time = time.time()
print(f"Time taken to compute twiss through list of BPMethElements: {end_time - start_time} seconds")

tw_new.plot('x y')
tw_new.plot('betx bety', 'dx dy')
plt.show()

