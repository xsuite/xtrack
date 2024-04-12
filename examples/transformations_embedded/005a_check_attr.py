import xtrack as xt
import numpy as np

bend = xt.Bend(k0=0.4, h=0.3, length=1, shift_x=1e-3, shift_y=2e-3, rot_s_rad=0.2,
               k1=0.1,
               knl=[0.7, 0.8, 0.9, 1.0], ksl=[0.1, 0.2, 0.3, 0.4])
quad = xt.Quadrupole(k1=0.1, k1s=0.2,
                     length=0.5, shift_x=2e-3, shift_y=1e-3, rot_s_rad=0.1)
sext = xt.Sextupole(k2=0.1, k2s=0.2,
                    length=0.3, shift_x=3e-3, shift_y=3e-3, rot_s_rad=0.3)
octu = xt.Octupole(k3=0.1, k3s=0.2,
                     length=0.4, shift_x=4e-3, shift_y=4e-3, rot_s_rad=0.4)
mult = xt.Multipole(knl=[0.7, 0.8, 0.9, 1.0], ksl=[0.1, 0.2, 0.3, 0.4],
                    length=0.4, shift_x=5e-3, shift_y=6e-3, rot_s_rad=0.7,
                    hxl=0.1)

line = xt.Line(elements=[bend, quad, sext, octu, mult])
line.build_tracker()
tt = line.get_table(attr=True)

assert_allclose = np.testing.assert_allclose

assert tt['element_type','e0'] == 'Bend'
assert tt['isreplica', 'e0'] == False
assert tt['parent_name', 'e0'] is None
assert tt['isthick', 'e0'] == True
assert tt['iscollective', 'e0'] == False
assert_allclose(tt['length', 'e0'], 1, rtol=0, atol=1e-14)
assert_allclose(tt['angle_rad', 'e0'], 0.3, rtol=0, atol=1e-14)
assert_allclose(tt['rot_s_rad', 'e0'], 0.2, rtol=0, atol=1e-14)
assert_allclose(tt['shift_x', 'e0'], 1e-3, rtol=0, atol=1e-14)
assert_allclose(tt['shift_y', 'e0'], 2e-3, rtol=0, atol=1e-14)
assert_allclose(tt['k0l', 'e0'], 0.4 * 1 + 0.7, rtol=0, atol=1e-14)
assert_allclose(tt['k1l', 'e0'], 0.1 * 1 + 0.8, rtol=0, atol=1e-14)
assert_allclose(tt['k2l', 'e0'], 0.9, rtol=0, atol=1e-14)
assert_allclose(tt['k3l', 'e0'], 1.0, rtol=0, atol=1e-14)
assert_allclose(tt['k0sl', 'e0'], 0.1, rtol=0, atol=1e-14)
assert_allclose(tt['k1sl', 'e0'], 0.2, rtol=0, atol=1e-14)
assert_allclose(tt['k2sl', 'e0'], 0.3, rtol=0, atol=1e-14)
assert_allclose(tt['k3sl', 'e0'], 0.4, rtol=0, atol=1e-14)

assert tt['element_type','e1'] == 'Quadrupole'
assert tt['isreplica', 'e1'] == False
assert tt['parent_name', 'e1'] is None
assert tt['isthick', 'e1'] == True
assert tt['iscollective', 'e1'] == False
assert_allclose(tt['length', 'e1'], 0.5, rtol=0, atol=1e-14)
assert_allclose(tt['angle_rad', 'e1'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['rot_s_rad', 'e1'], 0.1, rtol=0, atol=1e-14)
assert_allclose(tt['shift_x', 'e1'], 2e-3, rtol=0, atol=1e-14)
assert_allclose(tt['shift_y', 'e1'], 1e-3, rtol=0, atol=1e-14)
assert_allclose(tt['k0l', 'e1'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['k1l', 'e1'], 0.1 * 0.5, rtol=0, atol=1e-14)
assert_allclose(tt['k2l', 'e1'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['k3l', 'e1'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['k0sl', 'e1'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['k1sl', 'e1'], 0.2 * 0.5, rtol=0, atol=1e-14)
assert_allclose(tt['k2sl', 'e1'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['k3sl', 'e1'], 0.0, rtol=0, atol=1e-14)

assert tt['element_type','e2'] == 'Sextupole'
assert tt['isreplica', 'e2'] == False
assert tt['parent_name', 'e2'] is None
assert tt['isthick', 'e2'] == True
assert tt['iscollective', 'e2'] == False
assert_allclose(tt['length', 'e2'], 0.3, rtol=0, atol=1e-14)
assert_allclose(tt['angle_rad', 'e2'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['rot_s_rad', 'e2'], 0.3, rtol=0, atol=1e-14)
assert_allclose(tt['shift_x', 'e2'], 3e-3, rtol=0, atol=1e-14)
assert_allclose(tt['shift_y', 'e2'], 3e-3, rtol=0, atol=1e-14)
assert_allclose(tt['k0l', 'e2'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['k1l', 'e2'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['k2l', 'e2'], 0.1 * 0.3, rtol=0, atol=1e-14)
assert_allclose(tt['k3l', 'e2'], 0, rtol=0, atol=1e-14)
assert_allclose(tt['k0sl', 'e2'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['k1sl', 'e2'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['k2sl', 'e2'], 0.2 * 0.3, rtol=0, atol=1e-14)
assert_allclose(tt['k3sl', 'e2'], 0, rtol=0, atol=1e-14)

assert tt['element_type','e3'] == 'Octupole'
assert tt['isreplica', 'e3'] == False
assert tt['parent_name', 'e3'] is None
assert tt['isthick', 'e3'] == True
assert tt['iscollective', 'e3'] == False
assert_allclose(tt['length', 'e3'], 0.4, rtol=0, atol=1e-14)
assert_allclose(tt['angle_rad', 'e3'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['rot_s_rad', 'e3'], 0.4, rtol=0, atol=1e-14)
assert_allclose(tt['shift_x', 'e3'], 4e-3, rtol=0, atol=1e-14)
assert_allclose(tt['shift_y', 'e3'], 4e-3, rtol=0, atol=1e-14)
assert_allclose(tt['k0l', 'e3'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['k1l', 'e3'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['k2l', 'e3'], 0, rtol=0, atol=1e-14)
assert_allclose(tt['k3l', 'e3'], 0.1 * 0.4, rtol=0, atol=1e-14)
assert_allclose(tt['k0sl', 'e3'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['k1sl', 'e3'], 0.0, rtol=0, atol=1e-14)
assert_allclose(tt['k2sl', 'e3'], 0, rtol=0, atol=1e-14)
assert_allclose(tt['k3sl', 'e3'], 0.2 * 0.4, rtol=0, atol=1e-14)

assert tt['element_type','e4'] == 'Multipole'
assert tt['isreplica', 'e4'] == False
assert tt['parent_name', 'e4'] is None
assert tt['isthick', 'e4'] == False
assert tt['iscollective', 'e4'] == False
assert_allclose(tt['length', 'e4'], 0.4, rtol=0, atol=1e-14)
assert_allclose(tt['angle_rad', 'e4'], 0.1, rtol=0, atol=1e-14)
assert_allclose(tt['rot_s_rad', 'e4'], 0.7, rtol=0, atol=1e-14)
assert_allclose(tt['shift_x', 'e4'], 5e-3, rtol=0, atol=1e-14)
assert_allclose(tt['shift_y', 'e4'], 6e-3, rtol=0, atol=1e-14)
assert_allclose(tt['k0l', 'e4'], 0.7, rtol=0, atol=1e-14)
assert_allclose(tt['k1l', 'e4'], 0.8, rtol=0, atol=1e-14)
assert_allclose(tt['k2l', 'e4'], 0.9, rtol=0, atol=1e-14)
assert_allclose(tt['k3l', 'e4'], 1.0, rtol=0, atol=1e-14)
assert_allclose(tt['k0sl', 'e4'], 0.1, rtol=0, atol=1e-14)
assert_allclose(tt['k1sl', 'e4'], 0.2, rtol=0, atol=1e-14)
assert_allclose(tt['k2sl', 'e4'], 0.3, rtol=0, atol=1e-14)
assert_allclose(tt['k3sl', 'e4'], 0.4, rtol=0, atol=1e-14)

