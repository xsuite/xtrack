"""
SLS simulation with undulators.

This script loads the SLS MADX file, builds undulator using SplineBorisSequence,
inserts wigglers at 11 locations, computes twiss with radiation integrals, and prints results.
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
line_sls = env.ring

# Configure bend model
line_sls.configure_bend_model(core='mat-kick-mat')

# Set particle reference
line_sls.particle_ref = p0.copy()

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
    raw_data=df_raw_data,
    xy_point=(0, 0),
    dx=dx,
    dy=dy,
    ds=ds,
    min_region_size=5,
    deg=multipole_order-1,
)

field_fitter.set()
field_fitter.save_fit_pars(
    BASE_DIR
    / "field_maps"
    / "field_fit_pars.csv"
)

# Build undulator using SplineBorisSequence - automatically creates one SplineBoris
# element per polynomial piece with n_steps based on the data point count
seq = xt.SplineBorisSequence(
    df_fit_pars=field_fitter.df_fit_pars,
    multipole_order=multipole_order,
    steps_per_point=3,
)

splineborisline = seq.to_line(env=env)
splineborisline.particle_ref = p0.copy()

tw = splineborisline.twiss4d(betx=1, bety=1, include_collective=True)
tw.plot('x y')
tw.plot('betx bety', 'dx dy')
plt.show()

l_wig = seq.length

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

# Get element boundaries from the sequence
# Each element has s_start, so boundaries are at s=0 and each element's s_end
element_boundaries = [0.0]
for elem in seq.elements:
    element_boundaries.append(float(elem.s_end) - float(seq.elements[0].s_start))

# Desired corrector positions (relative to undulator start)
desired_positions = {
    'corr1': 0.02,
    'corr2': 0.1,
    'corr3': l_wig - 0.1,
    'corr4': l_wig - 0.02,
}

# Find nearest boundary for each corrector
def find_nearest_boundary_index(s_target, boundaries):
    """Find index of boundary closest to s_target."""
    return min(range(len(boundaries)), key=lambda i: abs(boundaries[i] - s_target))

corrector_insertions = {}  # boundary_index -> list of corrector names
for corr_name, s_target in desired_positions.items():
    idx = find_nearest_boundary_index(s_target, element_boundaries)
    if idx not in corrector_insertions:
        corrector_insertions[idx] = []
    corrector_insertions[idx].append(corr_name)
    print(f"{corr_name}: requested s={s_target:.4f}, inserting at boundary s={element_boundaries[idx]:.4f}")

# Build the line with correctors inserted at element boundaries
# (SplineBoris elements are already registered in env by seq.to_line(env=env))

# Build element name list with correctors inserted at boundaries
element_names_with_correctors = []
for i, sb_name in enumerate(seq.element_names):
    # Insert correctors before this element (at boundary i)
    if i in corrector_insertions:
        element_names_with_correctors.extend(corrector_insertions[i])
    element_names_with_correctors.append(sb_name)

# Insert correctors after the last element (at final boundary)
final_idx = len(seq.element_names)
if final_idx in corrector_insertions:
    element_names_with_correctors.extend(corrector_insertions[final_idx])

# Create the line with correctors at element boundaries
piecewise_undulator = xt.Line(env=env, element_names=element_names_with_correctors)

piecewise_undulator.build_tracker()

piecewise_undulator.particle_ref = p0.copy()

# Matching targets use corrector element names for intermediate positions
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


tw_undulator_corr = piecewise_undulator.twiss4d(betx=1, bety=1, include_collective=True)
tw_undulator_corr.plot('x y')
tw_undulator_corr.plot('betx bety', 'dx dy')
plt.show()
piecewise_undulator.discard_tracker()

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

tt = line_sls.get_table()
for wig_place in wiggler_places:
    print(f"Inserting piecewise_undulator {wig_place} at {tt['s', wig_place]}")
    line_sls.insert(piecewise_undulator, anchor='start', at=tt['s', wig_place])

line_sls.build_tracker()

tw_sls = line_sls.twiss4d(radiation_integrals=True)

# Plotting:
import matplotlib.pyplot as plt
plt.close('all')
tw_sls.plot('x y')
tw_sls.plot('betx bety', 'dx dy')
tw_sls.plot('betx2 bety2')
plt.show(block=False)

#['name', 's', 'x', 'px', 'y', 'py', 'zeta', 'delta', 'ptau', 'W_matrix', 'kin_px', 'kin_py', 'kin_ps', 'kin_xprime',
# 'kin_yprime', 'env_name', 'betx', 'bety', 'alfx', 'alfy', 'gamx', 'gamy', 'dx', 'dpx', 'dy', 'dpy', 'dx_zeta', 'dpx_zeta',
# 'dy_zeta', 'dpy_zeta', 'betx1', 'bety1', 'betx2', 'bety2', 'alfx1', 'alfy1', 'alfx2', 'alfy2', 'gamx1', 'gamy1',
# 'gamx2', 'gamy2', 'mux', 'muy', 'muzeta', 'nux', 'nuy', 'nuzeta', 'phix', 'phiy', 'phizeta', 'dmux', 'dmuy', 'dzeta',
# 'bx_chrom', 'by_chrom', 'ax_chrom', 'ay_chrom', 'wx_chrom', 'wy_chrom', 'ddx', 'ddpx', 'ddy', 'ddpy', 'c_minus_re',
# 'c_minus_im', 'c_r1', 'c_r2', 'c_phi1', 'c_phi2', 'k0l', 'k1l', 'k2l', 'k3l', 'k4l', 'k5l', 'k0sl', 'k1sl', 'k2sl',
# 'k3sl', 'k4sl', 'k5sl', 'angle_rad', 'rot_s_rad', 'hkick', 'vkick', 'ks', 'length', '_angle_force_body', 'element_type', 'isthick', 'parent_name']

# Extract and print results
print("=" * 80)
print("SLS WITH UNDULATORS")
print("=" * 80)
print(f"Tunes:")
print(f"  qx = {tw_sls.qx:.4e}")
print(f"  qy = {tw_sls.qy:.4e}")
print(f"  qs = {tw_sls.qs:.4e}")
print()
print(f"Chromaticity:")
print(f"  dqx = {tw_sls.dqx:.4e}")
print(f"  dqy = {tw_sls.dqy:.4e}")
print()
print(f"Partition numbers:")
print(f"  J_x = {tw_sls.rad_int_partition_number_x:.4e}")
print(f"  J_y = {tw_sls.rad_int_partition_number_y:.4e}")
print(f"  J_zeta = {tw_sls.rad_int_partition_number_zeta:.4e}")
print()
print(f"Damping constants per second:")
print(f"  alpha_x = {tw_sls.rad_int_damping_constant_x_s:.4e}")
print(f"  alpha_y = {tw_sls.rad_int_damping_constant_y_s:.4e}")
print(f"  alpha_zeta = {tw_sls.rad_int_damping_constant_zeta_s:.4e}")
print()
print(f"Equilibrium emittances:")
print(f"  eq_gemitt_x = {tw_sls.rad_int_eq_gemitt_x:.4e}")
print(f"  eq_gemitt_y = {tw_sls.rad_int_eq_gemitt_y:.4e}")
print(f"  eq_gemitt_zeta = {tw_sls.rad_int_eq_gemitt_zeta:.4e}")
print()
print(f"Energy loss per turn: {tw_sls.rad_int_eneloss_turn:.4e} eV")
print()
print(f"C^-: {tw_sls.c_minus:.4e}")
print()
print("=" * 80)

# Write results to file
output_dir = BASE_DIR / "spline_fitter" / "field_maps"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "SLS_WITH_UNDULATORS.txt"

with open(output_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("SLS WITH UNDULATORS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Tunes:\n")
    f.write(f"  qx = {tw_sls.qx:.4e}\n")
    f.write(f"  qy = {tw_sls.qy:.4e}\n")
    f.write(f"  qs = {tw_sls.qs:.4e}\n")
    f.write("\n")
    f.write(f"Chromaticity:\n")
    f.write(f"  dqx = {tw_sls.dqx:.4e}\n")
    f.write(f"  dqy = {tw_sls.dqy:.4e}\n")
    f.write("\n")
    f.write(f"Partition numbers:\n")
    f.write(f"  J_x = {tw_sls.rad_int_partition_number_x:.4e}\n")
    f.write(f"  J_y = {tw_sls.rad_int_partition_number_y:.4e}\n")
    f.write(f"  J_zeta = {tw_sls.rad_int_partition_number_zeta:.4e}\n")
    f.write("\n")
    f.write(f"Damping constants per second:\n")
    f.write(f"  alpha_x = {tw_sls.rad_int_damping_constant_x_s:.4e}\n")
    f.write(f"  alpha_y = {tw_sls.rad_int_damping_constant_y_s:.4e}\n")
    f.write(f"  alpha_zeta = {tw_sls.rad_int_damping_constant_zeta_s:.4e}\n")
    f.write("\n")
    f.write(f"Equilibrium emittances:\n")
    f.write(f"  eq_gemitt_x = {tw_sls.rad_int_eq_gemitt_x:.4e}\n")
    f.write(f"  eq_gemitt_y = {tw_sls.rad_int_eq_gemitt_y:.4e}\n")
    f.write(f"  eq_gemitt_zeta = {tw_sls.rad_int_eq_gemitt_zeta:.4e}\n")
    f.write("\n")
    f.write(f"Energy loss per turn: {tw_sls.rad_int_eneloss_turn:.4e} eV\n")
    f.write("=" * 80 + "\n")