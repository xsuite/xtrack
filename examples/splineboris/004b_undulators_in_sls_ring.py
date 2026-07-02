import xtrack as xt
import matplotlib.pyplot as plt

# Load the SLS ring
madx_file = '../../test_data/sls/sls.madx'
env = xt.load(str(madx_file))
line_sls = env.lines['ring']
line_sls.set_particle_ref('positron', p0c=2.7e9)
tt = line_sls.get_table()

# Import the undulator in the environment containing the ring
undulator = xt.load('./sls_undulator.json')
env.import_line(undulator, line_name='undulator')

# Install the undulator at several locations in the ring
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
insertions = []
for wig_place in wiggler_places:
    insertions.append(
        env.place(env['undulator'], anchor='start', at=tt['s_start', wig_place]))
line_sls.insert(insertions)

# Twiss with undulators
tw = line_sls.twiss4d()

# Plot and save the closed orbit
fig_closed_orbit = plt.figure(1, figsize=(10, 6))
tw.plot('x y', figure=fig_closed_orbit)
fig_closed_orbit.savefig('splineboris_sls_closed_orbit.png', dpi=200,
                         bbox_inches='tight')

#!end-doc-part

line_sls.particle_ref.anomalous_magnetic_moment=1.15965218076e-3

# Twiss with undulators
tw_sls = line_sls.twiss4d(radiation_integrals=True, spin=True, polarization_analysis=True)

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
print(f"Damping constants per second:")
print(f"  alpha_x = {tw_sls.rad_int_damping_constant_x_s:.4e}")
print(f"  alpha_y = {tw_sls.rad_int_damping_constant_y_s:.4e}")
print(f"  alpha_zeta = {tw_sls.rad_int_damping_constant_zeta_s:.4e}")
print()
print(f"Equilibrium emittances:")
print(f"  eq_gemitt_x = {tw_sls.rad_int_eq_gemitt_x:.4e}")
print(f"  eq_gemitt_y = {tw_sls.rad_int_eq_gemitt_y:.4e}")
print()
print(f"C^-: {tw_sls.c_minus:.4e}")
print()
print(f"Spin polarization: {tw_sls.spin_polarization_eq}")
print()
print("=" * 80)

import numpy as np
for kk in tw_sls.keys():
    if kk.startswith('spin'):
        if not np.isscalar(getattr(tw_sls, kk)):
            continue
        print(f"{kk}: {getattr(tw_sls, kk)}")

# Plotting:
import matplotlib.pyplot as plt
plt.close('all')
fig_co = plt.figure(1, figsize=(10, 6))
tw_sls.plot('x y', figure=fig_co)
tw_sls.plot('betx bety', 'dx dy')
tw_sls.plot('betx2 bety1')
tw_sls.plot('spin_x spin_z')
tw_sls.plot('spin_y')
plt.show()
