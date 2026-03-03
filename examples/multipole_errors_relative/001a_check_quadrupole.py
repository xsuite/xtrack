import xtrack as xt
import xobjects as xo
import numpy as np

main_is_skew = True

knl = np.array([0.001, 1e-3, 2e-2, 3e-2, 4, 50])
ksl = np.array([0.002, 2e-3, 3e-2, 4e-2, 5, 60])
knl_rel = np.array([1, 20, 30, 400, 50000, 6000000])
ksl_rel = np.array([2, 40, 60, 800, 100000, 1200000])
k1 = 0.001
k1s = 0.002

el_test = xt.Quadrupole(k1=k1, k1s=k1s, length=0.2,
                       knl=knl,
                       ksl=ksl,
                       knl_rel=knl_rel,
                       ksl_rel=ksl_rel,
                       main_is_skew=main_is_skew)

if main_is_skew:
    xo.assert_allclose(el_test.main_strength, 0.002 * 0.2, rtol=0, atol=1e-12)
else:
    xo.assert_allclose(el_test.main_strength, 0.001 * 0.2, rtol=0, atol=1e-12)

expected_knl = knl + knl_rel * el_test.main_strength
expected_ksl = ksl + ksl_rel * el_test.main_strength

expected_knl[1] += k1 * el_test.length
expected_ksl[1] += k1s * el_test.length

knl_tot, ksl_tot = el_test.get_total_knl_ksl()

xo.assert_allclose(knl_tot, expected_knl, rtol=0, atol=1e-12)
xo.assert_allclose(ksl_tot, expected_ksl, rtol=0, atol=1e-12)

el_ref = xt.Quadrupole(k1=k1, k1s=k1s, length=0.2,
                      knl=expected_knl, ksl=expected_ksl)
el_ref.knl[1] -= k1 * el_ref.length  # Not to double count the main quadrupole contribution
el_ref.ksl[1] -= k1s * el_ref.length

p0 = xt.Particles(p0c=1e9, x=1e-2, y=2e-2)

p_test = p0.copy()
el_test.track(p_test)

p_ref = p0.copy()
el_ref.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

line_test = xt.Line(elements=[el_test])
line_ref = xt.Line(elements=[el_ref])

# Check thick slicing
line_test_slice_thick = line_test.copy(shallow=True)
line_ref_slice_thick = line_ref.copy(shallow=True)

line_test_slice_thick.slice_thick_elements(
        slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])
line_ref_slice_thick.slice_thick_elements(
        slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thick'))])

p_test = p0.copy()
line_test_slice_thick.track(p_test)
p_ref = p0.copy()
line_ref_slice_thick.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)

# Check thin slicing
line_test_slice_thin = line_test.copy(shallow=True)
line_ref_slice_thin = line_ref.copy(shallow=True)
line_test_slice_thin.slice_thick_elements(
        slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])
line_ref_slice_thin.slice_thick_elements(
        slicing_strategies=[xt.Strategy(slicing=xt.Uniform(2, mode='thin'))])

p_test = p0.copy()
line_test_slice_thin.track(p_test)
p_ref = p0.copy()
line_ref_slice_thin.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)