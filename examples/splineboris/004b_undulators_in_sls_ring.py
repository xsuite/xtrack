"""
SLS simulation with undulators, closed spin tracking and radiation.

This script loads the SLS MADX file and the corrected undulator saved by
example 004, inserts it at 11 locations, and computes twiss with spin tracking
and radiation.
"""

import xtrack as xt
from pathlib import Path
import matplotlib.pyplot as plt

E0 = 2.7e9
BASE_DIR = Path(__file__).resolve().parent
UNDULATOR_JSON = BASE_DIR / 'sls_undulator.json'

# Particle reference
p0 = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=E0)

# Load the corrected undulator produced by example 004. Its environment is
# used as the destination because import_line cannot yet copy SplineBoris
# elements from another environment.
piecewise_undulator = xt.load(UNDULATOR_JSON)
# env = piecewise_undulator.env
# env['undulator'] = piecewise_undulator

# Load the SLS ring and import it into the undulator environment.
madx_file = Path(__file__).resolve().parent.parent.parent / 'test_data' / 'sls' / 'sls.madx'
env = xt.load(str(madx_file))

line_sls = env.lines['ring']
env.import_line(piecewise_undulator, line_name='undulator')


# env.import_line(sls_env.ring, line_name='ring')
# line_sls = env.lines['ring']



# Configure bend model
line_sls.configure_bend_model(core='mat-kick-mat')

# Set particle reference
line_sls.particle_ref = p0.copy()

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
insertions = []
for wig_place in wiggler_places:
    insertions.append(env.place(env['undulator'].copy(), anchor='start', at=tt['s', wig_place]))
line_sls.insert(insertions)

line_sls.build_tracker()

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
print(f"Spin polarization: {tw_sls.spin_polarization_eq:.4e}")
print()
print("=" * 80)

# Plotting:
import matplotlib.pyplot as plt
plt.close('all')
tw_sls.plot('x y')
tw_sls.plot('betx bety', 'dx dy')
tw_sls.plot('betx2 bety1')
tw_sls.plot('spin_x spin_z')
tw_sls.plot('spin_y')
plt.show()
