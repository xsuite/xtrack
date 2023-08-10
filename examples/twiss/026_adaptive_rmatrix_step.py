import numpy as np
import xtrack as xt

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()
collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

collider.lhcb1.twiss_default['nemitt_x'] = 1e-6
collider.lhcb1.twiss_default['nemitt_y'] = 1e-6
collider.lhcb2.twiss_default['nemitt_x'] = 1e-6
collider.lhcb2.twiss_default['nemitt_y'] = 1e-6

tw = collider.twiss()
assert tw.lhcb1.steps_r_matrix['adapted'] == False
assert tw.lhcb2.steps_r_matrix['adapted'] == False

collider.lhcb1.twiss_default['nemitt_x'] = 1e-8
tw = collider.twiss()
assert tw.lhcb1.steps_r_matrix['adapted'] == True
assert tw.lhcb2.steps_r_matrix['adapted'] == False

collider.lhcb2.twiss_default['nemitt_y'] = 2e-8
tw = collider.twiss()
assert tw.lhcb1.steps_r_matrix['adapted'] == True
assert tw.lhcb2.steps_r_matrix['adapted'] == True

expected_dx_b1 = 0.01 * np.sqrt(1e-8 * 0.15 / collider.lhcb1.particle_ref._xobject.gamma0[0])
expected_dy_b1 = 0.01 * np.sqrt(1e-6 * 0.15 / collider.lhcb1.particle_ref._xobject.gamma0[0])
expected_dx_b2 = 0.01 * np.sqrt(1e-6 * 0.15 / collider.lhcb1.particle_ref._xobject.gamma0[0])
expected_dy_b2 = 0.01 * np.sqrt(2e-8 * 0.15 / collider.lhcb2.particle_ref._xobject.gamma0[0])

assert np.isclose(tw.lhcb1.steps_r_matrix['dx'], expected_dx_b1, atol=0, rtol=1e-4)
assert np.isclose(tw.lhcb1.steps_r_matrix['dy'], expected_dy_b1, atol=0, rtol=1e-4)
assert np.isclose(tw.lhcb2.steps_r_matrix['dx'], expected_dx_b2, atol=0, rtol=1e-4)
assert np.isclose(tw.lhcb2.steps_r_matrix['dy'], expected_dy_b2, atol=0, rtol=1e-4)