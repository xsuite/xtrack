"""
Spin tracking with undulators and radiation.

This script builds and corrects an SLS undulator from fitted Spline4 field
data, saves it to JSON, computes twiss with spin tracking and radiation, and
displays results.
"""

import xtrack as xt
import pandas as pd

#################################################
# Polynomial fit on the data from the field map #
#################################################

# Load the raw field map data from shared test_data
field_map_path = "../../test_data/sls/undulator_field_map.txt"
df_raw_data = pd.read_csv(
    field_map_path,
    sep=r"\s+",
    header=None,
    names=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
).set_index(["X", "Y", "Z"])

# Use fitting procedure to extract field and derivatives on the reference trajectory.
# This class is taylored for this example data, use your own fitting procedure for other datasets.
from xtrack._temp.splineboris.field_fitter import FieldFitter
field_fitter = FieldFitter(
    raw_data=df_raw_data,
    xy_point=(0, 0),
    distance_unit=0.001, # dataset uses mm
    min_region_size=10,
    deg=2,
)
spline_data = field_fitter.get_spline_data()

# `spline_data` contains for each longitudinal interval the 4th-order
# polynomial coefficients (in the form of value at start/end of interval,
# longitudinal derivative at start/end of interval, and mean value) for the field
# components and their transverse derivatives. For example:
# spline_data[0] is:
# {'s_start': -1.1,
#  's_end': -1.095,
#  'idx_start': 0,
#  'idx_end': 5,
#  'bs':
#      Spline4(val_start=0.0, der_start=0.0, val_end=0.0, der_end=0.0, mean=0.0),
#  'bx': (
#         # bx on axis (x=0,y=0)
#         Spline4(val_start=0.0002597788559479, der_start=0.003908159160349505,
#                 val_end=0.00027929455770183206, der_end=0.00389142774801777,
#                 mean=0.00026954160840155075),
#         # d bx/d x on axis (x=0,y=0)
#         Spline4(val_start=0.0, der_start=0.0, val_end=0.0, der_end=0.0, mean=0.0),
#         # d^2 bx/d x^2 on axis (x=0,y=0)
#         Spline4(val_start=-0.03587500889997759, der_start=-42.00397475010047,
#                 val_end=-0.04413421209996192, der_end=-6.919120799986231,
#                 mean=-0.04245506171996625)
#        ),
#  'by': (
#       # by on axis (x=0,y=0)
#       Spline4(val_start=0.0020494067017488, der_start=0.0627448571810936,
#               val_end=0.0023965826590473483, der_end=0.07647819250243666,
#               mean=0.0022172898542134225),
#       # d by/d x on axis (x=0,y=0)
#       Spline4(val_start=0.0, der_start=0.0, val_end=0.0, der_end=0.0, mean=0.0),
#       # d^2 by/d x^2 on axis (x=0,y=0)
#       Spline4(val_start=-0.5104028836997457, der_start=-29.057401352068492,
#               val_end=-0.6429682228011369, der_end=-48.71395930245775,
#               mean=-0.5725836650301926)
#      )
# }


#######################################
# Build Xsuite model of the undulator #
#######################################

# Create an xtrack environment
env = xt.Environment()
env.set_particle_ref('electron', p0c=2.7e9)

# Build SplineBoris elements for each interval using the fitted field data:
undulator_element_names = []
for ii, piece in enumerate(spline_data):
    element_name = f'undulator_splineboris_{ii}'

    # Match the field-map resolution: one Boris step per interval
    # between adjacent data points in this piece.
    nn_steps = max(1, piece['idx_end'] - piece['idx_start'])

    env.elements[element_name] = xt.SplineBoris(
        length=piece['s_end'] - piece['s_start'],
        n_steps=nn_steps,
        bs=piece['bs'],
        bx=piece['bx'],
        by=piece['by'],
    )
    undulator_element_names.append(element_name)

# Assemble the undulator line
undulator = env.new_line(components=undulator_element_names)

###########################################################################
# Install thin dipole correctors at the edges of the undulator to control #
# trajectory along the undulator.                                         #
###########################################################################

# Create env variables for corrector strengths (needed for matching)
env['k0l_corr1'] = 0.
env['k0l_corr2'] = 0.
env['k0l_corr3'] = 0.
env['k0l_corr4'] = 0.
env['k0sl_corr1'] = 0.
env['k0sl_corr2'] = 0.
env['k0sl_corr3'] = 0.
env['k0sl_corr4'] = 0.

# Create corrector elements with expressions referencing env variables
env.new('corr1', xt.Multipole, knl=['k0l_corr1'], ksl=['k0sl_corr1'])
env.new('corr2', xt.Multipole, knl=['k0l_corr2'], ksl=['k0sl_corr2'])
env.new('corr3', xt.Multipole, knl=['k0l_corr3'], ksl=['k0sl_corr3'])
env.new('corr4', xt.Multipole, knl=['k0l_corr4'], ksl=['k0sl_corr4'])

# Insert correctors
l_undulator = undulator.get_length()
undulator.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_undulator - 0.1),
    env.place('corr4', at=l_undulator - 0.02),
], s_tol=5e-3) # large s_tol avoids slicing the SplineBoris elements

# Use optimizer to control the orbit
opt = undulator.match(
    solve=False,
    betx=1, bety=1,
    include_collective=True,
    vary=xt.VaryList(['k0l_corr1', 'k0sl_corr1',
                      'k0l_corr2', 'k0sl_corr2',
                      'k0l_corr3', 'k0sl_corr3',
                      'k0l_corr4', 'k0sl_corr4',
                      ], step=1e-6),
    targets=[
        xt.TargetSet(x=0, px=0, y=0, py=0., at=xt.END),
        xt.TargetSet(x=0., y=0, at='corr2'),
        xt.TargetSet(x=0., y=0, at='corr3')
        ],
)
opt.solve()

###############################
# Save undulator to json file #
###############################

undulator.to_json('sls_undulator.json')

####################
# Checks and plots #
####################

undulator.particle_ref.anomalous_magnetic_moment = 0.00115965218128

# Save the corrected undulator before enabling radiation below. Example 005
# reloads this file and inserts the undulator into the SLS ring.

tw_undulator_corr_spin = undulator.twiss4d(
    betx=1, bety=1,
    spin=True,
    spin_x=0.5, spin_y=0.25, spin_z=0.25
    )
tw_undulator_corr_spin.plot('x y')
tw_undulator_corr_spin.plot('betx bety', 'dx dy')
tw_undulator_corr_spin.plot('spin_x')
tw_undulator_corr_spin.plot('spin_y')
tw_undulator_corr_spin.plot('spin_z')


# Enable radiation on all elements
undulator.configure_radiation(model='mean')

# Run twiss4d including radiation effects (average energy loss)
tw_undulator_corr_spin_rad = undulator.twiss(
    betx=1, bety=1,
    include_collective=True,
    spin=True,
    spin_x=0.5, spin_y=0.25, spin_z=0.25,
    radiation_method='full'
)

import matplotlib.pyplot as plt

# Plot results to compare with/without radiation
fig, axs = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

# Plot closed orbit x
axs[0].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.x, label='no rad')
axs[0].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.x, label='with rad')
axs[0].set_ylabel('x [m]')
axs[0].legend()
axs[0].grid(True)

# Plot energy loss (delta)
axs[1].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.delta, label='delta (no rad)', linestyle='-')
axs[1].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.delta, label='delta (with rad)', linestyle='--')
axs[1].set_ylabel('delta (relative energy deviation)')
axs[1].legend()
axs[1].grid(True)

# Plot spin components
axs[2].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.spin_x, label='spin_x (no rad)', linestyle='-')
axs[2].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.spin_y, label='spin_y (no rad)', linestyle='-')
axs[2].plot(tw_undulator_corr_spin.s, tw_undulator_corr_spin.spin_z, label='spin_z (no rad)', linestyle='-')
axs[2].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.spin_x, label='spin_x (with rad)', linestyle='--')
axs[2].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.spin_y, label='spin_y (with rad)', linestyle='--')
axs[2].plot(tw_undulator_corr_spin_rad.s, tw_undulator_corr_spin_rad.spin_z, label='spin_z (with rad)', linestyle='--')
axs[2].set_ylabel('spin')
axs[2].set_xlabel('s [m]')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.show()
