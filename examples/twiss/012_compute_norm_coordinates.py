# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import xtrack as xt

# Load a line and build tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_noerrors_nobb/line_and_particle.json')
line.particle_ref = xt.Particles(
                    mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.build_tracker()

# Generate some particles with known normalized coordinates
particles = line.build_particles(
    nemitt_x=2.5e-6, nemitt_y=1e-6,
    x_norm=[-1, 0, 0.5], y_norm=[0.3, -0.2, 0.2],
    px_norm=[0.1, 0.2, 0.3], py_norm=[0.5, 0.6, 0.8],
    zeta=[0, 0.1, -0.1], delta=[1e-4, 0., -1e-4])

# Inspect physical coordinates
tab = particles.get_table()
tab.show()
# prints
#
# Table: 3 rows, 17 cols
# particle_id s            x           px            y          py zeta   delta chi ...
#           0 0 -0.000253245  3.33271e-06  5.10063e-05 1.00661e-06    0  0.0001   1
#           1 0 -2.06127e-09  3.32087e-07 -3.42343e-05 5.59114e-08  0.1       0   1
#           2 0  0.000152331 -7.62878e-07  3.45785e-05  1.0462e-06 -0.1 -0.0001   1


# Compute twiss
tw = line.twiss()

# Use twiss to compute normalized coordinates
norm_coord = tw.get_normalized_coordinates(particles, nemitt_x=2.5e-6,
                                           nemitt_y=1e-6)

# Inspect normalized coordinates
norm_coord.show()
#
# Table: 3 rows, 8 cols
# particle_id at_element       x_norm px_norm y_norm py_norm   zeta_norm  pzeta_norm
#           0          0           -1     0.1    0.3     0.5 1.06651e-07  0.00313799
#           1          0 -1.59607e-20     0.2   -0.2     0.6  0.00318676 1.12046e-05
#           2          0          0.5     0.3    0.2     0.8  -0.0031868  -0.0031492

#!end-doc-part

# Check that the computed normalized coordinates are correct
assert np.allclose(norm_coord['x_norm'], [-1, 0, 0.5], atol=5e-14, rtol=0)
assert np.allclose(norm_coord['y_norm'], [0.3, -0.2, 0.2], atol=5e-14, rtol=0)
assert np.allclose(norm_coord['px_norm'], [0.1, 0.2, 0.3], atol=5e-14, rtol=0)
assert np.allclose(norm_coord['py_norm'], [0.5, 0.6, 0.8], atol=5e-14, rtol=0)
