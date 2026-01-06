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

def _contruct_par_table(n_steps, s_start, s_end, df_flat):
    par_dicts = []
    par_table = []
    s_vals = np.linspace(float(s_start), float(s_end), n_steps)
    prefix_order = {'a_': 0, 'b_': 1, 'bs_': 2}

    def sort_key(name):
        for p, pr in prefix_order.items():
            if isinstance(name, str) and name.startswith(p):
                return (pr, name)
        return (99, name)

    for s_val_i in s_vals:
        mask = (df_flat['s_start'] <= s_val_i) & (df_flat['s_end'] >= s_val_i)
        rows = df_flat[mask]
        if rows.empty:
            par_dicts.append({})
            par_table.append([])
            continue
        rows = rows.copy()
        rows['__sort_key'] = rows['param_name'].apply(sort_key)
        rows = rows.sort_values('__sort_key')
        names_sorted = rows['param_name'].to_numpy()
        vals_sorted = rows['param_value'].to_numpy(dtype=np.float64)
        par_dicts.append(dict(zip(names_sorted.tolist(), vals_sorted.tolist())))
        par_table.append(vals_sorted.tolist())
    return par_dicts, par_table

par_dicts, par_table = _contruct_par_table(n_steps, s_start, s_end, df_reset)

multipole_order = 3 # Corresponds to sextupolar component.

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

bpmeth_element = xt.BPMethElement(params=par_table, multipole_order=3, s_start=s_start, s_end=s_end, length=s_end-s_start, n_steps=n_steps)

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

for i in range(n_steps):
    params_i = [par_table[i]]
    s_val_i = s_vals[i]
    # find interval in s_boundaries that contains s_val_i
    idx = np.searchsorted(s_boundaries, s_val_i, side='right')-1
    if idx < 0:
        s_start = s_boundaries[0]
    else:
        s_start = s_boundaries[idx]
    s_end = s_boundaries[idx + 1] if (idx + 1) < len(s_boundaries) else s_boundaries[-1]
    wiggler_i = xt.BPMethElement(params=params_i, multipole_order=multipole_order, s_start=s_start, s_end=s_end, n_steps=1)
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