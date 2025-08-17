import xtrack as xt
import xobjects as xo
import numpy as np

line = xt.Line(elements={
    's1': xt.Sextupole(k2=0.1, length=9.0)
})
line.particle_ref = xt.Particles(p0c=7000e9)

line_1slice = line.copy(shallow=True)
line_1slice.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(1), element_type=xt.Sextupole),
    ])

line_3slices = line.copy(shallow=True)
line_3slices.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(3), element_type=xt.Sextupole),
    ])

tw_1slice = line_1slice.twiss(betx=1, bety=1, x=1e-2)
tw_3slices = line_3slices.twiss(betx=1, bety=1, x=1e-2)

tw_1kick = line.twiss(betx=1, bety=1, x=1e-2)

line.configure_sextupole_model(num_multipole_kicks=3)

assert line['s1'].num_multipole_kicks == 3
tw_3kicks = line.twiss(betx=1, bety=1, x=1e-2)

assert np.abs(tw_1slice.x[-1] - tw_3slices.x[-1]) > 1e-6

xo.assert_allclose(tw_1slice.s[-1], tw_1kick.s[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_1slice.x[-1], tw_1kick.x[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_1slice.px[-1], tw_1kick.px[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_1slice.y[-1], tw_1kick.y[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_1slice.py[-1], tw_1kick.py[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_1slice.zeta[-1], tw_1kick.zeta[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_1slice.ptau[-1], tw_1kick.ptau[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_1slice.betx[-1], tw_1kick.betx[-1], atol=1e-10, rtol=0)
xo.assert_allclose(tw_1slice.bety[-1], tw_1kick.bety[-1], atol=1e-10, rtol=0)

xo.assert_allclose(tw_3slices.s[-1], tw_3kicks.s[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_3slices.x[-1], tw_3kicks.x[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_3slices.px[-1], tw_3kicks.px[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_3slices.y[-1], tw_3kicks.y[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_3slices.py[-1], tw_3kicks.py[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_3slices.zeta[-1], tw_3kicks.zeta[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_3slices.ptau[-1], tw_3kicks.ptau[-1], atol=1e-15, rtol=0)
xo.assert_allclose(tw_3slices.betx[-1], tw_3kicks.betx[-1], atol=1e-10, rtol=0)
xo.assert_allclose(tw_3slices.bety[-1], tw_3kicks.bety[-1], atol=1e-10, rtol=0)
