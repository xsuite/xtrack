import xtrack as xt
import xobjects as xo
import numpy as np
import pandas as pd

# env = xt.Environment()
# env.particle_ref = xt.Particles(
#     mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9
#)

# So given this dataframe, I'd like to do the following.
# - The data frame is indexed by field component, derivative and region start. The region start may differ per field component or derivative.
# - I'd like to create one flattened array that contains

n_part = 20
filepath = 'fit_parameters.csv'
df = pd.read_csv(filepath, index_col=['field_component', 'derivative_x', 'region_name', 's_start', 's_end', 'idx_start', 'idx_end', 'param_index'])

s_start = df.reset_index()['s_start'].values
s_end = df.reset_index()['s_end'].values
s_boundaries = np.unique(np.concatenate((s_start, s_end)))

n_steps = 1000

s_vals = np.linspace(s_boundaries[0], s_boundaries[-1], n_steps)

param_names  = df["param_name"].to_numpy()
param_values = df["param_value"].to_numpy()

def _contruct_par_table(s_vals, n_steps, s_start, s_end, param_names, param_values):
    par_dicts = []
    par_table = []
    for i in range(n_steps):
        s_val_i = s_vals[i]
        mask = (s_start <= s_val_i) & (s_end > s_val_i)
        names = param_names[mask]
        vals = param_values[mask]
        par_dicts.append(dict(zip(names, vals)))
        par_table.append(list(vals))
    return par_dicts, par_table

par_dicts, par_table = _contruct_par_table(s_vals, n_steps, s_start, s_end, param_names, param_values)

multipole_order = 3 # Corresponds to sextupolar component.

bpmeth_element = xt.BPMethElement(params=par_table, multipole_order=3, s_start=s_start, s_end=s_end, n_steps=n_steps)

particles = xt.Particles(
    x=np.linspace(-1e-3, 1e-3, n_part),
    px=np.linspace(-1e-3, 1e-3, n_part),
    y=np.linspace(-1e-3, 1e-3, n_part),
    py=np.linspace(-1e-3, 1e-3, n_part),
    zeta=np.zeros(n_part),
    delta=np.zeros(n_part),
)

bpmeth_element.track(particles)