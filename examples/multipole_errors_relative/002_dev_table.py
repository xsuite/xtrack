from pyparsing import line

import xtrack as xt
import xobjects as xo

# TODO:
# - Slices and replicas

env = xt.Environment()

line1 = env.new_line(components=[
    env.new('bend', xt.Bend, length=0.1, angle=0.01, at=0.3,
            knl=[0.002, 0.03, 0.4, 5, 6],
            ksl=[0.003, 0.04, 0.5, 6, 7],
            knl_rel=[0.1, 0.2, 0.3, 0.4, 0.5],
            ksl_rel=[0.1, 0.2, 0.3, 0.4, 0.5]),
    env.new('rbend', xt.RBend, length_straight=0.1, angle=0.01, at=0.6,
            knl=[0.003, 0.04, 0.5, 6, 7],
            ksl=[0.004, 0.05, 0.6, 7, 8],
            knl_rel=[0.2, 0.3, 0.4, 0.5, 0.6],
            ksl_rel=[0.7, 0.6, 0.5, 0.4, 0.3]),
    env.new('quad', xt.Quadrupole, length=0.1, k1=2, at=1,
            knl=[0.001, 0.02, 0.3, 4, 5],
            ksl=[0.002, 0.03, 0.4, 5, 6],
            knl_rel=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            ksl_rel=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
    env.new('skew_quad', xt.Quadrupole, length=0.1, k1s=2, main_is_skew=True, at=2,
            knl=[0.001, 0.02, 0.3, 4, 5],
            ksl=[0.002, 0.03, 0.4, 5, 6],
            knl_rel=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            ksl_rel=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    env.new('sext', xt.Sextupole, length=0.1, k2=3, at=3,
            knl=[0.0005, 0.01, 0.2, 3, 4],
            ksl=[0.001, 0.02, 0.3, 4, 5],
            knl_rel=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ksl_rel=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    env.new('skew_sext', xt.Sextupole, length=0.1, k2s=3, main_is_skew=True, at=4,
            knl=[0.0005, 0.01, 0.2, 3, 4],
            ksl=[0.001, 0.02, 0.3, 4, 5],
            knl_rel=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            ksl_rel=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    env.new('oct', xt.Octupole, length=0.1, k3=4, at=5),
    env.new('skew_oct', xt.Octupole, length=0.1, k3s=4, main_is_skew=True, at=6,
            knl=[0.0001, 0.001, 0.01, 2, 3],
            ksl=[0.0002, 0.002, 0.02, 3, 4],
            knl_rel=[0.5, 0.6, 0.7, 0.8, 0.9, 1],
            ksl_rel=[0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    env.new('multipole', xt.Multipole, length=0.1, knl=[0,0,0,0,2], isthick=True,
            main_order=4, at=7,
            ksl=[1,2,3,4,5],
            knl_rel=[0.1, 0.2, 0.3, 0.4, 0.5],
            ksl_rel=[0.5, 0.4, 0.3, 0.2, 0.1],
            ),
    env.new('skew_multipole', xt.Multipole, length=0.1, ksl=[0,0,0,0,3], isthick=True,
            main_is_skew=True, main_order=4, at=8,
            knl=[5,4,3,2,1],
             knl_rel=[0.5, 0.4, 0.3, 0.2, 0.1],
             ksl_rel=[0.1, 0.2, 0.3, 0.4, 0.5],
             ),
])
line2 = line1.copy(shallow=True)
line2.slice_thick_elements(slicing_strategies=[
    xt.Strategy(slicing=xt.Teapot(2, mode='thick'))])

line3 = line1.copy(shallow=True)
line3.slice_thick_elements(slicing_strategies=[
    xt.Strategy(slicing=xt.Teapot(2, mode='thin'))])

line = line1 + line2 + line3

tt = line.get_table(attr=True)

# Check _main_strength, k0l, k0sl, k2l, k2sl, k3l, k3sl, k4l, k4sl, k5l, k5sl
for nn in tt.name:

    if nn == '_end_point':
        continue

    ee = line[nn]

    if isinstance(ee, (xt.Bend, xt.RBend, xt.Quadrupole, xt.Sextupole, xt.Octupole, xt.Multipole)):
        xo.assert_allclose(ee.main_strength, tt['_main_strength', nn], rtol=0, atol=1e-14)
        knl, ksl = ee.get_total_knl_ksl()
        for ii in range(6):
            if ii >= len(knl):
                assert tt[f'k{ii}l', nn] == 0
                assert tt[f'k{ii}sl', nn] == 0
            else:
                xo.assert_allclose(knl[ii], tt[f'k{ii}l', nn], rtol=0, atol=1e-14)
                xo.assert_allclose(ksl[ii], tt[f'k{ii}sl', nn], rtol=0, atol=1e-14)
    elif (ee.__class__.__name__.startswith('ThickSlice')
          or ee.__class__.__name__.startswith('ThinSlice')
          or ee.__class__.__name__.startswith('DriftSlice')):
        xo.assert_allclose(ee._parent.main_strength*ee.weight*ee._inherit_strengths, tt['_main_strength', nn], rtol=0, atol=1e-14)
        knl_parent, ksl_parent = ee._parent.get_total_knl_ksl()
        for ii in range(6):
            if ii >= len(knl_parent):
                assert tt[f'k{ii}l', nn] == 0
                assert tt[f'k{ii}sl', nn] == 0
            else:
                xo.assert_allclose(knl_parent[ii]*ee.weight*ee._inherit_strengths, tt[f'k{ii}l', nn], rtol=0, atol=1e-14)
                xo.assert_allclose(ksl_parent[ii]*ee.weight*ee._inherit_strengths, tt[f'k{ii}sl', nn], rtol=0, atol=1e-14)
    else:
        assert isinstance(ee, (xt.Drift, xt.Marker))
        assert tt['_main_strength', nn] == 0