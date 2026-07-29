import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()

angle = 0.1
env.new('rb', 'RBend', length_straight=2., angle=0.1,
        edge_entry_model='full', edge_exit_model='full',
        rbend_model='curved-body')

anomalous_magnetic_moment = 1.15965218091e-3

line = env.new_line(components=['rb'])
line.set_particle_ref('electron', energy0=5e9,
                      anomalous_magnetic_moment=anomalous_magnetic_moment)


two = line.twiss4d(betx=1, bety=1, spin=True, spin_x=1)

spin_angle = np.atan2(two.spin_z, two.spin_x)
expected_spin_angle = anomalous_magnetic_moment * line.particle_ref.gamma0[0] * angle

xo.assert_allclose(spin_angle[-1], expected_spin_angle, rtol=1e-8)