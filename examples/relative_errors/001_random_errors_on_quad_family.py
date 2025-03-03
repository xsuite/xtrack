import xtrack as xt
import numpy as np

env = xt.load_madx_lattice('../../test_data/fcc_ee/fccee_z.seq')
line = env.fccee_p_ring
line.particle_ref = xt.Particles(p0c=45.6e9, mass0=xt.ELECTRON_MASS_EV)

tt = line.get_table()

# I want to add a random octupolar error to all quadrupoles whose name starts
# with 'qf'

# Table with all quads
tt_quad = tt.rows[tt.element_type=='Quadrupole']
tt_selected = tt_quad.rows['qf.*']
# is:
# Table: 752 rows, 11 cols
# name                s element_type isthick isreplica ...
# qfg2.1        468.213 Quadrupole      True     False
# qfg2.2        520.504 Quadrupole      True     False
# qfg2.3        572.795 Quadrupole      True     False
# qfg2.4        625.085 Quadrupole      True     False
# qfg2.5        677.376 Quadrupole      True     False
# qfg2.6        729.667 Quadrupole      True     False
# qfg2.7        781.957 Quadrupole      True     False
# qfg2.8        834.248 Quadrupole      True     False
# qfg2.9        886.539 Quadrupole      True     False
# qfg2.10        938.83 Quadrupole      True     False
# ...
# qf2.351         88811 Quadrupole      True     False
# qf2.352       88915.2 Quadrupole      True     False
# qf2.353       89019.4 Quadrupole      True     False

# Generate random octupolar errors
k3l_rel_rms = 1e-3
k3l_rel = k3l_rel_rms * (np.random.rand(len(tt_selected)))

# Generate error dictionary
errors = {}
for ii, nn in enumerate(tt_selected.name):
    errors[nn] = {'rel_knl': [0, 0, k3l_rel[ii]]}

# Set errors in the line
env.set_multipolar_errors(errors)

# The number of multipolar kicks along the elements is set automatically
# (one kick upstream by default for quadrupoles). In can be changed as follows
env.set(tt_selected.name, num_multipole_kicks=4)

# Inspect one element:
env['qfg2.5'].knl
# is : [0.00000000e+00, 0.00000000e+00, 4.02838126e-05, 0.00000000e+00,
#       0.00000000e+00, 0.00000000e+00]
env['qfg2.5'].num_multipole_kicks
# is: 4
