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

def create_flattened_fit_pars(df):
    max_params = df.groupby(['field_component', 'derivative_x', 'region_name']).size().max()
    n_regions = df.reset_index()['region_name'].nunique()
    n_field_components = df.reset_index()['field_component'].nunique()
    flattened_fit_pars = np.zeros((n_regions * n_field_components, max_params))
    for i, ((field_component, derivative_x, region_name), group) in enumerate(df.groupby(['field_component', 'derivative_x', 'region_name'])):
        params = group.sort_values('param_index')['param_value'].values
        flattened_fit_pars[i, :len(params)] = params
    return flattened_fit_pars

flattened_fit_pars = create_flattened_fit_pars(df)
print("Flattened fit parameters:", flattened_fit_pars)
prrrrr
# TODO: Insert the proper element here.
test_element = xt.Sietse(Bs=0.5, length=1)

particles = xt.Particles(
    x=np.linspace(-1e-3, 1e-3, n_part),
    px=np.linspace(-1e-3, 1e-3, n_part),
    y=np.linspace(-1e-3, 1e-3, n_part),
    py=np.linspace(-1e-3, 1e-3, n_part),
    zeta=np.zeros(n_part),
    delta=np.zeros(n_part),
)

initial_x = particles.x.copy()
initial_y = particles.y.copy()
initial_px = particles.px.copy()
initial_py = particles.py.copy()
initial_zeta = particles.zeta.copy()
initial_delta = particles.delta.copy()

print("Initial State:")
print(f"initial_x = {initial_x}")
print(f"initial_y = {initial_y}")
print(f"initial_px = {initial_px}")
print(f"initial_py = {initial_py}")
print(f"initial_zeta = {initial_zeta}")
print(f"initial_delta = {initial_delta}")

test_element.track(particles)