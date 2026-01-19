"""
SLS simulation without undulators.

This script loads the SLS MADX file, creates a copy of the ring without wigglers,
computes twiss with radiation integrals, and prints the results.
"""

import xtrack as xt
from pathlib import Path

# Particle reference
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9)

# Load SLS MADX file
madx_file = Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls' / 'b075_2024.09.25.madx'
env = xt.load(str(madx_file))
line_sls = env.ring

# Create copy of ring without wigglers
line_no_wiggler = line_sls.copy(shallow=True)
env['ring_no_wiggler'] = line_no_wiggler

# Configure bend model
line_no_wiggler.configure_bend_model(core='mat-kick-mat')

# Set particle reference
line_no_wiggler.particle_ref = p0.copy()

# Compute twiss with radiation integrals
tw_no_wiggler = line_no_wiggler.twiss4d(radiation_integrals=True, spin=True, polarization=True)

# Plotting:
import matplotlib.pyplot as plt
plt.close('all')
tw_no_wiggler.plot('x y')
tw_no_wiggler.plot('betx bety', 'dx dy')
tw_no_wiggler.plot('betx2 bety2')
tw_no_wiggler.plot('spin_x spin_y spin_z')
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
print("SLS WITHOUT UNDULATORS")
print("=" * 80)
print(f"Tunes:")
print(f"  qx = {tw_no_wiggler.qx:.4f}")
print(f"  qy = {tw_no_wiggler.qy:.4f}")
print(f"  qs = {tw_no_wiggler.qs:.4f}")
print()
print(f"Chromaticity:")
print(f"  dqx = {tw_no_wiggler.dqx:.4f}")
print(f"  dqy = {tw_no_wiggler.dqy:.4f}")
print()
print(f"Partition numbers:")
print(f"  J_x = {tw_no_wiggler.rad_int_partition_number_x:.4f}")
print(f"  J_y = {tw_no_wiggler.rad_int_partition_number_y:.4f}")
print(f"  J_zeta = {tw_no_wiggler.rad_int_partition_number_zeta:.4f}")
print()
print(f"Damping constants per second:")
print(f"  alpha_x = {tw_no_wiggler.rad_int_damping_constant_x_s:.4f}")
print(f"  alpha_y = {tw_no_wiggler.rad_int_damping_constant_y_s:.4f}")
print(f"  alpha_zeta = {tw_no_wiggler.rad_int_damping_constant_zeta_s:.4f}")
print()
print(f"Equilibrium emittances:")
print(f"  eq_gemitt_x = {tw_no_wiggler.rad_int_eq_gemitt_x:.4f}")
print(f"  eq_gemitt_y = {tw_no_wiggler.rad_int_eq_gemitt_y:.4f}")
print(f"  eq_gemitt_zeta = {tw_no_wiggler.rad_int_eq_gemitt_zeta:.4f}")
print()
print(f"Energy loss per turn: {tw_no_wiggler.rad_int_eneloss_turn:.4f} eV")
print()
print(f"C^-: {tw_no_wiggler.c_minus:.4e}")
print()
print(f"Spin polarization: {tw_no_wiggler.spin_polarization_eq:.4e}")
print()
print("=" * 80)

# Write results to file
output_dir = Path("/home/simonfan/cernbox/Documents/Presentations/Section_Meeting_Undulators")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "SLS_WITHOUT_UNDULATORS.txt"

with open(output_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("SLS WITHOUT UNDULATORS\n")
    f.write("=" * 80 + "\n")
    f.write(f"Tunes:\n")
    f.write(f"  qx = {tw_no_wiggler.qx:.4e}\n")
    f.write(f"  qy = {tw_no_wiggler.qy:.4e}\n")
    f.write(f"  qs = {tw_no_wiggler.qs:.4e}\n")
    f.write("\n")
    f.write(f"Chromaticity:\n")
    f.write(f"  dqx = {tw_no_wiggler.dqx:.4e}\n")
    f.write(f"  dqy = {tw_no_wiggler.dqy:.4e}\n")
    f.write("\n")
    f.write(f"Partition numbers:\n")
    f.write(f"  J_x = {tw_no_wiggler.rad_int_partition_number_x:.4e}\n")
    f.write(f"  J_y = {tw_no_wiggler.rad_int_partition_number_y:.4e}\n")
    f.write(f"  J_zeta = {tw_no_wiggler.rad_int_partition_number_zeta:.4e}\n")
    f.write("\n")
    f.write(f"Damping constants per second:\n")
    f.write(f"  alpha_x = {tw_no_wiggler.rad_int_damping_constant_x_s:.4e}\n")
    f.write(f"  alpha_y = {tw_no_wiggler.rad_int_damping_constant_y_s:.4e}\n")
    f.write(f"  alpha_zeta = {tw_no_wiggler.rad_int_damping_constant_zeta_s:.4e}\n")
    f.write("\n")
    f.write(f"Equilibrium emittances:\n")
    f.write(f"  eq_gemitt_x = {tw_no_wiggler.rad_int_eq_gemitt_x:.4e}\n")
    f.write(f"  eq_gemitt_y = {tw_no_wiggler.rad_int_eq_gemitt_y:.4e}\n")
    f.write(f"  eq_gemitt_zeta = {tw_no_wiggler.rad_int_eq_gemitt_zeta:.4e}\n")
    f.write("\n")
    f.write(f"Energy loss per turn: {tw_no_wiggler.rad_int_eneloss_turn:.4e} eV\n")
    f.write("\n")
    f.write(f"C^-: {tw_no_wiggler.c_minus:.4e}\n")
    f.write("\n")
    f.write(f"Spin polarization: {tw_no_wiggler.spin_polarization_eq:.4e}\n")
    f.write("\n")
    f.write("=" * 80 + "\n")