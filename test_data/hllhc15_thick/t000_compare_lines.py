import xtrack as xt

line_old = xt.load('temp_line_before.json')
line_new = xt.load('lhc_thick_with_knobs.json')

env = line_new.env

sinc = env.functions['sinc']

tt_rbend = env.elements.get_table().rows.match('RBend', 'element_type')
for nn in tt_rbend.name:
    ee_ref = env.ref[nn]
    ee = env.get(nn)
    if ee.k0_from_h:
        continue
    k0 = ee_ref.k0._expr or ee_ref.k0._value
    angle = ee_ref.angle._expr or ee_ref.angle._value

    line_new[nn].k0 = k0 * sinc(angle / 2)
tw_old = line_old.twiss4d()
tw_new = line_new.twiss4d()

two_new = line_new.twiss(betx=1, bety=1)