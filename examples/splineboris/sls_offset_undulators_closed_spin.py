"""
SLS simulation with offset undulators and closed spin tracking.

This script loads the SLS MADX file, builds undulator using SplineBorisSequence
with transverse offset, inserts wigglers at 11 locations, and computes twiss
with spin tracking.
"""

import xtrack as xt
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from spline_fitter.field_fitter import FieldFitter


multipole_order = 3

# Particle reference
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)

# Load SLS MADX file
madx_file = Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls' / 'b075_2024.09.25.madx'
env = xt.load(str(madx_file))
line_offset = env.ring

# Configure bend model
line_offset.configure_bend_model(core='mat-kick-mat')

# Set particle reference
line_offset.particle_ref = p0.copy()

BASE_DIR = Path(__file__).resolve().parent

# Load the raw field map data from knot_map_test.txt
field_map_path = BASE_DIR / "field_maps" / "knot_map_test.txt"
df_raw_data = pd.read_csv(
    field_map_path,
    sep='\t',
    header=None,
    names=['X', 'Y', 'Z', 'Bx', 'By', 'Bs'],
)
df_raw_data = df_raw_data.set_index(['X', 'Y', 'Z'])

# Grid spacing in meters (the dataset uses mm, so 1 mm = 0.001 m)
dx = 0.001
dy = 0.001
ds = 0.001

field_fitter = FieldFitter(
    df_raw_data=df_raw_data,
    xy_point=(0, 0),
    dx=dx,
    dy=dy,
    ds=ds,
    min_region_size=10,
    deg=multipole_order-1,
)

field_fitter.fit()
field_fitter.save_fit_pars(
    BASE_DIR
    / "field_maps"
    / "field_fit_pars.csv"
)

# Define offsets for the undulator
x_off = 5e-4  # 0.0005 m offset in x
y_off = 0  # 0 offset in y

# Build undulator using SplineBorisSequence with offset
seq = xt.SplineBorisSequence(
    df_fit_pars=field_fitter.df_fit_pars,
    multipole_order=multipole_order,
    steps_per_point=1,
    shift_x=x_off,
    shift_y=y_off,
)

# Get the Line of SplineBoris elements (pass env for insert support)
piecewise_undulator = seq.to_line(env=env)
l_wig = seq.length

piecewise_undulator.build_tracker()

piecewise_undulator.particle_ref = p0.copy()

piecewise_undulator.discard_tracker()

# The issue: When you use betx=1, bety=1, twiss4d treats the line as OPEN (non-periodic).
# For an open line, the orbit is computed from initial conditions in particle_on_co.
# If particle_on_co has zero initial conditions (x=0, px=0, y=0, py=0), the orbit will
# be zero unless there are kicks from the wiggler.
#
# Solution: Use only_orbit=True to explicitly compute the orbit, which should properly
# propagate through the wiggler and show any kicks/deviations.

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


# tw_undulator_corr = piecewise_undulator.twiss4d(betx=1, bety=1, include_collective=True)
# tw_undulator_corr.plot('x y')
# tw_undulator_corr.plot('betx bety', 'dx dy')
# plt.show()

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

tt = line_offset.get_table()
for wig_place in wiggler_places:
    print(f"Inserting piecewise_undulator {wig_place} at {tt['s', wig_place]}")
    line_offset.insert(piecewise_undulator, anchor='start', at=tt['s', wig_place])

line_offset.build_tracker()

tw_offset = line_offset.twiss4d(radiation_integrals=True, spin=True, polarization=True)

# Plotting:
import matplotlib.pyplot as plt
plt.close('all')
tw_offset.plot('x y')
tw_offset.plot('betx bety', 'dx dy')
tw_offset.plot('betx2 bety2')
tw_offset.plot('spin_x spin_z')
tw_offset.plot('spin_y')
plt.show()

#['name', 's', 'x', 'px', 'y', 'py', 'zeta', 'delta', 'ptau', 'W_matrix', 'kin_px', 'kin_py', 'kin_ps', 'kin_xprime',
# 'kin_yprime', 'env_name', 'betx', 'bety', 'alfx', 'alfy', 'gamx', 'gamy', 'dx', 'dpx', 'dy', 'dpy', 'dx_zeta', 'dpx_zeta',
# 'dy_zeta', 'dpy_zeta', 'betx1', 'bety1', 'betx2', 'bety2', 'alfx1', 'alfy1', 'alfx2', 'alfy2', 'gamx1', 'gamy1',
# 'gamx2', 'gamy2', 'mux', 'muy', 'muzeta', 'nux', 'nuy', 'nuzeta', 'phix', 'phiy', 'phizeta', 'dmux', 'dmuy', 'dzeta',
# 'bx_chrom', 'by_chrom', 'ax_chrom', 'ay_chrom', 'wx_chrom', 'wy_chrom', 'ddx', 'ddpx', 'ddy', 'ddpy', 'c_minus_re',
# 'c_minus_im', 'c_r1', 'c_r2', 'c_phi1', 'c_phi2', 'k0l', 'k1l', 'k2l', 'k3l', 'k4l', 'k5l', 'k0sl', 'k1sl', 'k2sl',
# 'k3sl', 'k4sl', 'k5sl', 'angle_rad', 'rot_s_rad', 'hkick', 'vkick', 'ks', 'length', '_angle_force_body', 'element_type', 'isthick', 'parent_name']

# Extract and print results
print("=" * 80)
print("SLS WITH OFFSET UNDULATORS")
print("=" * 80)
print(f"Tunes:")
print(f"  qx = {tw_offset.qx:.4e}")
print(f"  qy = {tw_offset.qy:.4e}")
print(f"  qs = {tw_offset.qs:.4e}")
print()
print(f"Chromaticity:")
print(f"  dqx = {tw_offset.dqx:.4e}")
print(f"  dqy = {tw_offset.dqy:.4e}")
print()
print(f"Partition numbers:")
print(f"  J_x = {tw_offset.rad_int_partition_number_x:.4e}")
print(f"  J_y = {tw_offset.rad_int_partition_number_y:.4e}")
print(f"  J_zeta = {tw_offset.rad_int_partition_number_zeta:.4e}")
print()
print(f"Damping constants per second:")
print(f"  alpha_x = {tw_offset.rad_int_damping_constant_x_s:.4e}")
print(f"  alpha_y = {tw_offset.rad_int_damping_constant_y_s:.4e}")
print(f"  alpha_zeta = {tw_offset.rad_int_damping_constant_zeta_s:.4e}")
print()
print(f"Equilibrium emittances:")
print(f"  eq_gemitt_x = {tw_offset.rad_int_eq_gemitt_x:.4e}")
print(f"  eq_gemitt_y = {tw_offset.rad_int_eq_gemitt_y:.4e}")
print(f"  eq_gemitt_zeta = {tw_offset.rad_int_eq_gemitt_zeta:.4e}")
print()
print(f"Energy loss per turn: {tw_offset.rad_int_eneloss_turn:.4e} eV")
print()
print(f"C^-: {tw_offset.c_minus:.4e}")
print()
print(f"Spin polarization: {tw_offset.spin_polarization_eq:.4e}")
print()
print("=" * 80)

# Write results to file
output_dir = Path("/home/simonfan/cernbox/Documents/Presentations/Section_Meeting_Undulators")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "SLS_WITH_OFFSET_UNDULATORS.txt"

with open(output_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("SLS WITH OFFSET UNDULATORS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Tunes:\n")
    f.write(f"  qx = {tw_offset.qx:.4e}\n")
    f.write(f"  qy = {tw_offset.qy:.4e}\n")
    f.write(f"  qs = {tw_offset.qs:.4e}\n")
    f.write("\n")
    f.write(f"Chromaticity:\n")
    f.write(f"  dqx = {tw_offset.dqx:.4e}\n")
    f.write(f"  dqy = {tw_offset.dqy:.4e}\n")
    f.write("\n")
    f.write(f"Partition numbers:\n")
    f.write(f"  J_x = {tw_offset.rad_int_partition_number_x:.4e}\n")
    f.write(f"  J_y = {tw_offset.rad_int_partition_number_y:.4e}\n")
    f.write(f"  J_zeta = {tw_offset.rad_int_partition_number_zeta:.4e}\n")
    f.write("\n")
    f.write(f"Damping constants per second:\n")
    f.write(f"  alpha_x = {tw_offset.rad_int_damping_constant_x_s:.4e}\n")
    f.write(f"  alpha_y = {tw_offset.rad_int_damping_constant_y_s:.4e}\n")
    f.write(f"  alpha_zeta = {tw_offset.rad_int_damping_constant_zeta_s:.4e}\n")
    f.write("\n")
    f.write(f"Equilibrium emittances:\n")
    f.write(f"  eq_gemitt_x = {tw_offset.rad_int_eq_gemitt_x:.4e}\n")
    f.write(f"  eq_gemitt_y = {tw_offset.rad_int_eq_gemitt_y:.4e}\n")
    f.write(f"  eq_gemitt_zeta = {tw_offset.rad_int_eq_gemitt_zeta:.4e}\n")
    f.write("\n")
    f.write(f"Energy loss per turn: {tw_offset.rad_int_eneloss_turn:.4e} eV\n")
    f.write("\n")
    f.write(f"C^-: {tw_offset.c_minus:.4e}\n")
    f.write("\n")
    f.write(f"Spin polarization: {tw_offset.spin_polarization_eq:.4e}\n")
    f.write("=" * 80 + "\n")