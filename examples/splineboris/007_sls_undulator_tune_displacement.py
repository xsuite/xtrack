import xtrack as xt
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from xtrack._temp.splineboris.field_fitter import FieldFitter


multipole_order = 3

E0 = 2.7e9

# Particle reference
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0)

# Load SLS MADX file
madx_file = Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls' / 'sls.madx'
env = xt.load(str(madx_file))
line_sls = env.ring

# Configure bend model
line_sls.configure_bend_model(core='mat-kick-mat')

# Set particle reference
line_sls.particle_ref = p0.copy()

BASE_DIR = Path(__file__).resolve().parent

# Load the raw field map data from shared test_data
field_map_path = BASE_DIR.parent.parent / "test_data" / "sls" / "undulator_field_map.txt"
df_raw_data = pd.read_csv(
    field_map_path,
    sep=r"\s+",
    header=None,
    names=["X", "Y", "Z", "Bskew", "Bnorm", "Bs"],
).set_index(["X", "Y", "Z"])

# Distance unit in meters (the dataset uses mm, so 1 mm = 0.001 m)
distance_unit = 0.001

field_fitter = FieldFitter(
    raw_data=df_raw_data,
    xy_point=(0, 0),
    distance_unit=distance_unit,
    min_region_size=10,
    deg=multipole_order-1,
)

# Get the fitted field data for each longitudinal interval. Tracking settings
# are intentionally applied below when constructing the SplineBoris elements.
spline_data = field_fitter.get_spline_data(
    multipole_order=multipole_order,
)

# Build and register the SplineBoris elements explicitly.
undulator_element_names = []
for ii, piece in enumerate(spline_data):
    element_name = f'undulator_splineboris_{ii}'
    env.elements[element_name] = xt.SplineBoris(
        length=piece['s_end'] - piece['s_start'],
        n_steps=max(1, piece['idx_end'] - piece['idx_start']),
        bs=piece['bs'],
        bx=piece['bx'],
        by=piece['by'],
    )
    undulator_element_names.append(element_name)

piecewise_undulator = env.new_line(components=undulator_element_names)
l_wig = sum(piece['s_end'] - piece['s_start'] for piece in spline_data)

piecewise_undulator.build_tracker()

piecewise_undulator.particle_ref = p0.copy()

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

# Insert correctors at nearest element boundary (s_tol avoids slicing)
piecewise_undulator.insert([
    env.place('corr1', at=0.02),
    env.place('corr2', at=0.1),
    env.place('corr3', at=l_wig - 0.1),
    env.place('corr4', at=l_wig - 0.02),
], s_tol=5e-3)

opt = piecewise_undulator.match(
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
        xt.TargetSet(x=0., y=0, at='corr2'),
        xt.TargetSet(x=0., y=0, at='corr3')
        ],
)
opt.step(2)

piecewise_undulator.discard_tracker()

# Uncomment which undulator you want to insert
wiggler_places = [
    #'ars11_uind_0210_1',
    'ars11_uind_0610_1',
]

# Tunes in the case of no undulator.
qx_0 = line_sls.twiss4d(include_collective=True).qx
qy_0 = line_sls.twiss4d(include_collective=True).qy

tt = line_sls.get_table()
for wig_place in wiggler_places:
    print(f"Inserting piecewise_undulator {wig_place} at {tt['s', wig_place]}")
    line_sls.insert(piecewise_undulator, anchor='start', at=tt['s', wig_place])

deltaqx_list = []
deltaqy_list = []

n_tunes = 30

hor_off_list = np.linspace(-0.05e-3, 0.05e-3, n_tunes)

spline_names = [
    nn for nn in line_sls.element_names
    if isinstance(line_sls[nn], xt.SplineBoris)
    ]

for dx in hor_off_list:   # dx in meters
    # apply horizontal offset to all undulator slices
    for nn in spline_names:
        line_sls[nn].shift_x = dx
    # then compute tune for this offset
    tw = line_sls.twiss4d(include_collective=True)
    deltaqx_list.append(tw.qx - qx_0)
    deltaqy_list.append(tw.qy - qy_0)

