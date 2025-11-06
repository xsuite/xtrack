import xtrack as xt
import xobjects as xo

import numpy as np

bend = xt.Bend(
    length=2,
    h=0.1,
    k0=0,
    # shift_x=0.01,
    # shift_y=-0.03,
    # shift_s=0.02,
    # rot_x_rad=-0.03,
    # rot_y_rad=0.02,
    # rot_s_rad_no_frame=0.01,
    # rot_s_rad=0.04,
    # rot_shift_anchor=0.2,
)
bend.integrator = 'uniform'
bend.num_multipole_kicks = 1



line_test = xt.Line(elements=[bend], element_names=['bend'])
line_test.configure_spin('auto')

p0 = xt.Particles(
    x=0.2,
    y=-0.6,
    # px=0.1,
    # py=0.02,
    # zeta=0.5,
    # delta=0.9,
    spin_z=1.,
    # p0c=700e9,
    mass0=xt.ELECTRON_MASS_EV,
    anomalous_magnetic_moment=0.00115965218128,
)

p = p0.copy()
line_test.track(p)

expected_norm = 0.1
result_norm = np.linalg.norm([p.spin_x, p.spin_y, p.spin_z])
xo.assert_allclose(result_norm, expected_norm, atol=1e-15, rtol=1e-15)

