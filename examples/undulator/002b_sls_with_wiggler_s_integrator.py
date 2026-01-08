import xtrack as xt
import matplotlib.pyplot as plt
import xobjects as xo
import numpy as np
import pandas as pd
import time
from pathlib import Path

import sys

np.set_printoptions(threshold=sys.maxsize)

env = xt.load('test_data/sls_2.0/b075_2024.09.25.madx')
line = env.ring
env['ring_no_wiggler'] = line.copy(shallow=True)
line.configure_bend_model(core='mat-kick-mat')

# Set particle reference for electron storage ring
if env.particle_ref is None:
    env.particle_ref = xt.Particles(
        mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9
    )
line.particle_ref = env.particle_ref

# So given this dataframe, I'd like to do the following.
# - The data frame is indexed by field component, derivative and region start. The region start may differ per field component or derivative.
# - I'd like to create one flattened array that contains

n_part = 1

BASE_DIR = Path(__file__).resolve().parent
filepath = BASE_DIR / "fit_parameters.csv"
# filepath = 'fit_parameters.csv'
df = pd.read_csv(filepath, index_col=['field_component', 'derivative_x', 'region_name', 's_start', 's_end', 'idx_start',
                                      'idx_end', 'param_index'])

# work with a flat dataframe for easier masking by s
df_reset = df.reset_index()

s_starts = np.sort(df_reset['s_start'].to_numpy(dtype=np.float64))
s_ends = np.sort(df_reset['s_end'].to_numpy(dtype=np.float64))

s_boundaries = np.sort(np.unique(np.concatenate((s_starts, s_ends))))

s_start = s_boundaries[0]
s_end = s_boundaries[-1]

l_wig = s_end - s_start

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

# list of n_steps wigglers;
list = []
name_list = ['wiggler_' + str(i) for i in range(1, n_steps + 1)]

s_vals = np.linspace(s_start, s_end, n_steps)
ds = (s_end - s_start) / n_steps

for i in range(n_steps):
    # params should be a 2D array: [[param1, param2, ...]] for n_steps=1
    params_i = [par_table[i]]
    s_val_i = s_vals[i]

    # For each single-step element, s_start and s_end should define the range
    # in the field map that this step covers. Use a small interval around s_val_i.
    # The s coordinates here are in the field map coordinate system (can be negative).
    elem_s_start = s_val_i - ds / 2
    elem_s_end = s_val_i + ds / 2

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

wiggler = env.new_line(components=name_list)
wiggler.particle_ref = line.particle_ref

wiggler.build_tracker()

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

wiggler.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_wig - 0.1),
    env.place('corr4', at=l_wig - 0.02),
    ], s_tol=5e-3
)

# Rebuild tracker after insert and ensure particle_ref is set
wiggler.particle_ref = line.particle_ref
wiggler.build_tracker()

# Computed for 1000 slices, 1000 steps
# env.vars.update(
# {'k0l_corr1': np.float64(-0.0004540792291112204),
#  'k0sl_corr1': np.float64(-1.213769189237666e-06),
#  'k0l_corr2': np.float64(0.0008135172335552242),
#  'k0sl_corr2': np.float64(0.00023470961164860475),
#  'k0l_corr3': np.float64(-0.0001955197609031625),
#  'k0sl_corr3': np.float64(-0.00021394733008765638),
#  'k0l_corr4': np.float64(-0.00015806879956816854),
#  'k0sl_corr4': np.float64(3.370506139561265e-05)})

# For x0 = 0.5e-3
# env.vars.update(
# {'k0l_corr1': np.float64(-0.0004640274435485036),
#  'k0sl_corr1': np.float64(-1.2297340793905685e-06),
#  'k0l_corr2': np.float64(0.0008265782650066877),
#  'k0sl_corr2': np.float64(0.0002344711077490433),
#  'k0l_corr3': np.float64(-0.00018319740840498774),
#  'k0sl_corr3': np.float64(-0.00021346101458338208),
#  'k0l_corr4': np.float64(-0.00016749244113701785),
#  'k0sl_corr4': np.float64(3.3646895667713495e-05)})

# To compute the kicks
# Get the actual element names after insertion and slicing
tt_wig = wiggler.get_table()
# Find elements at approximately 1/6 and 5/6 of the wiggler length
s_target_1 = l_wig / 6
s_target_2 = 5 * l_wig / 6
# Find closest elements to these positions
idx_1 = np.argmin(np.abs(tt_wig['s_center'] - s_target_1))
idx_2 = np.argmin(np.abs(tt_wig['s_center'] - s_target_2))
elem_name_1 = tt_wig['name', idx_1]
elem_name_2 = tt_wig['name', idx_2]

opt = wiggler.match(
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
        xt.TargetSet(x=0., y=0, at=elem_name_1),
        xt.TargetSet(x=0., y=0, at=elem_name_2)
        ],
)
opt.step(2)

print('Twiss wiggler only')
start_time = time.time()
tw_wig_only = wiggler.twiss(include_collective=True, betx=1, bety=1, particle_ref=wiggler.particle_ref)
end_time = time.time()
print(f"Time to compute twiss wiggler only: {end_time - start_time} seconds")
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

tt = line.get_table()
for wig_place in wiggler_places:
    line.insert(wiggler, anchor='start', at=tt['s', wig_place])

env['on_wig_corr'] = 0
mywig.scale = 0
tw_no_wig = line.twiss4d(strengths=True)
tw_vs_momentum_no_wig = {}
for dd in deltas:
    tw_vs_momentum_no_wig[dd] = line.twiss4d(delta0=dd,
                                         compute_chromatic_properties=False)

env['on_wig_corr'] = 1.0
mywig.scale = 1.0

print('Twiss full line with wiggler')
p_co = tw_no_wig.particle_on_co.copy()
p_co.at_element=0

tw = line.twiss4d(include_collective=True, particle_on_co=p_co,
                  compute_chromatic_properties=False)



tw_vs_momentum = {}
for delta in deltas:
    print(f'Twiss off momentum, delta = {delta}')
    p_off = p_co.copy()
    p_off.delta += delta
    p_off.x += tw.dx[0] * delta
    p_off.px += tw.dpx[0] * delta
    p_off.y += tw.dy[0] * delta
    p_off.py += tw.dpy[0] * delta
    p_off.at_element=0
    tw_vs_momentum[delta] = line.twiss4d(include_collective=True,
                                         particle_on_co=p_off,
                                         compute_chromatic_properties=False)


cols_chrom, scalars_chrom = xt.twiss._compute_chromatic_functions(line, init=None,
                                      delta_chrom=delta_chrom,
                                      steps_r_matrix=None,
                                      matrix_responsiveness_tol=None,
                                      matrix_stability_tol=None,
                                      symplectify=None,
                                      tw_chrom_res=[tw_vs_momentum[-delta_chrom],
                                                    tw_vs_momentum[delta_chrom]],
                                      on_momentum_twiss_res=tw)

tw._data.update(cols_chrom)
tw._data.update(scalars_chrom)
tw._col_names += list(cols_chrom.keys())

dl = np.diff(s_cuts)
wig_mult_places = []
for ii, (bbx, bby) in enumerate(zip(Bx_mid, By_mid)):
    nn = f'wig_mult_{ii}'
    pp = env.new(nn, xt.Bend,
                 length=dl[ii],
                 knl=[dl[ii] * b1_mid[ii] / p0.rigidity0[0], 0, dl[ii] * b3_mid[ii] / p0.rigidity0[0]],
                 ksl=[dl[ii] * a1_mid[ii] / p0.rigidity0[0], 0, dl[ii] * a3_mid[ii] / p0.rigidity0[0]],
                 shift_x=x0,
                 at=s_mid[ii])
    wig_mult_places.append(pp)

wiggler_mult = wiggler.copy(shallow=True)
tt_slices = wiggler.get_table().rows['wigslic.*']

wiggler_mult.remove(tt_slices.name)
wiggler_mult.insert(wig_mult_places)

tw_wig_mult = wiggler_mult.twiss(betx=1, bety=1)

line_wig_mult = env['ring_no_wiggler'].copy(shallow=True)
line_wig_mult.particle_ref = p0.copy()

for wig_place in wiggler_places:
    line_wig_mult.insert(wiggler_mult, anchor='start', at=tt['s', wig_place])

tw_wig_mult = line_wig_mult.twiss4d()

tt_mult = wiggler_mult.get_table().rows['wig_mult_.*']
for nn in tt_mult.name:
    env[nn].knl = 0
    env[nn].ksl = 0
env['on_wig_corr'] = 0.0
tw_wig_mult_off = line_wig_mult.twiss4d()


plt.close('all')
plt.figure(1, figsize=(6.4, 4.8))
ax = plt.subplot(111)
ax.plot(tw_wig_mult.s, tw_wig_mult.betx/tw_wig_mult_off.betx - 1, label='multipoles')
ax.plot(tw.s, tw.betx/tw_no_wig.betx - 1, '-', label='BPMETH')
ax.set_ylabel(r'$\Delta \beta_x / \beta_x$')
ax.set_xlabel('s [m]')
ax.legend()

plt.figure(2, figsize=(6.4, 4.8))
ax = plt.subplot(111)
ax.plot(tw_wig_mult.s, tw_wig_mult.betx2, label=r'Multipoles, $|C^-|$='+f'{tw_wig_mult.c_minus:.2e}')
ax.plot(tw.s, tw.betx2, '-', label='BPMETH, '+f'$|C^-|$={tw.c_minus:.2e}')
ax.set_ylabel(r'$\beta_{x,2}$ [m]')
ax.set_xlabel('s [m]')
ax.legend()
ax.set_ylim(0, 0.2)

plt.show()