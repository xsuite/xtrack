import xtrack as xt

bend = xt.Bend(k0=0.4, h=0.3, length=1, shift_x=1e-3, shift_y=2e-3, rot_s_rad=0.2,
               k1=0.1,
               knl=[0.7, 0.8, 0.9, 1.0], ksl=[0.1, 0.2, 0.3, 0.4])
quad = xt.Quadrupole(k1=0.1, k1s=0.2,
                     length=0.1, shift_x=2e-3, shift_y=1e-3, rot_s_rad=0.1)
sext = xt.Sextupole(k2=0.1, k2s=0.2,
                    length=0.1, shift_x=3e-3, shift_y=3e-3, rot_s_rad=0.3)
octu = xt.Octupole(k3=0.1, k3s=0.2,
                     length=0.1, shift_x=4e-3, shift_y=4e-3, rot_s_rad=0.4)

line = xt.Line(elements=[bend, quad, sext, octu])
line.build_tracker()
tt = line.get_table(attr=True)

assert tt['element_type','e0'] == 'Bend'
assert tt['isreplica', 'e0'] == False
assert tt['parent_name', 'e0'] is None
assert tt['isthick', 'e0'] == True
assert tt['iscollective', 'e0'] == False
assert tt['length', 'e0'] == 1
assert tt['angle_rad', 'e0'] == 0.3
assert tt['rot_s_rad', 'e0'] == 0.2
assert tt['shift_x', 'e0'] == 1e-3
assert tt['shift_y', 'e0'] == 2e-3
assert tt['k0l', 'e0'] == 0.4 * 1 + 0.7
assert tt['k1l', 'e0'] == 0.1 * 1 + 0.8
assert tt['k2l', 'e0'] == 0.9
assert tt['k3l', 'e0'] == 1.0
assert tt['k0sl', 'e0'] == 0.1
assert tt['k1sl', 'e0'] == 0.2
assert tt['k2sl', 'e0'] == 0.3
assert tt['k3sl', 'e0'] == 0.4

