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

##################################
# Plot orbit along the undulator #
##################################

import matplotlib.pyplot as plt

tw_undulator = undulator.twiss4d(betx=1, bety=1)
tw_undulator.plot('x y')

plt.show()
