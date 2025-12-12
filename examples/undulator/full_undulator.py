import xtrack as xt
import matplotlib.pyplot as plt
import xobjects as xo
import numpy as np
import pandas as pd
import time

env = xt.Environment()
env.particle_ref = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9
)

# So given this dataframe, I'd like to do the following.
# - The data frame is indexed by field component, derivative and region start. The region start may differ per field component or derivative.
# - I'd like to create one flattened array that contains

n_part = 1
filepath = 'fit_parameters.csv'
df = pd.read_csv(filepath, index_col=['field_component', 'derivative_x', 'region_name', 's_start', 's_end', 'idx_start', 'idx_end', 'param_index'])

s_start = np.sort(df.reset_index()['s_start'].to_numpy(dtype=np.float64))
s_end = np.sort(df.reset_index()['s_end'].to_numpy(dtype=np.float64))
s_boundaries = np.sort(np.unique(np.concatenate((s_start, s_end))))

n_steps = int(1000)

param_names  = df["param_name"].to_numpy()
param_values = df["param_value"].to_numpy()

def _contruct_par_table(n_steps, s_start, s_end, param_names, param_values):
    par_dicts = []
    par_table = []
    s_vals = np.linspace(s_start[0], s_end[-1], n_steps)
    for i in range(n_steps):
        s_val_i = s_vals[i]
        mask = (s_start <= s_val_i) & (s_end >= s_val_i)
        names = param_names[mask]
        vals = param_values[mask]
        par_dicts.append(dict(zip(names, vals)))
        par_table.append(np.array(vals))
    return par_dicts, par_table

par_dicts, par_table = _contruct_par_table(n_steps, s_start, s_end, param_names, param_values)

multipole_order = 3 # Corresponds to sextupolar component.

bpmeth_element = xt.BPMethElement(params=par_table, multipole_order=3, s_start=s_boundaries[0], s_end=s_boundaries[-1], n_steps=n_steps)

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
# list = []
# name_list = ['wiggler_'+str(i) for i in range(1,n_steps+1)]
#
# for i in range(n_steps):
#     start_s = s_start[i]
#     end_s = s_end[i+1]
#     params_i = [par_table[i]]
#     wiggler_i = xt.BPMethElement(params=params_i, multipole_order=multipole_order, s_start=start_s, s_end=end_s, n_steps=1)
#     list.append(wiggler_i)
#     env.elements[name_list[i]] = wiggler_i
#
# new_line = env.new_line(name_list)
#
# new_line.build_tracker()
#
# start_time = time.time()
# tw_new = new_line.twiss4d(betx=1, bety=1, include_collective=True)
# end_time = time.time()
# print(f"Time taken to compute twiss through list of BPMethElements: {end_time - start_time} seconds")
#
# tw_new.plot('x y')
# tw_new.plot('betx bety', 'dx dy')
# plt.show()