import xtrack as xt
import xobjects as xo
import numpy as np

rel_ref_is_skew = True

knl = np.array([0.001, 1e-3, 2e-2, 3e-2, 4, 50])
ksl = np.array([0.002, 2e-3, 3e-2, 4e-2, 5, 60])
knl_rel = np.array([1, 20, 30, 400, 50000, 6000000])
ksl_rel = np.array([2, 40, 60, 800, 100000, 1200000])
k1 = 0.001
k1s = 0.002

q_test = xt.Quadrupole(k1=k1, k1s=k1s, length=0.2,
                       knl=knl,
                       ksl=ksl,
                       knl_rel=knl_rel,
                       ksl_rel=ksl_rel,
                       rel_ref_is_skew=rel_ref_is_skew)

if rel_ref_is_skew:
    xo.assert_allclose(q_test.rel_ref_strength, 0.002 * 0.2, rtol=0, atol=1e-12)
else:
    xo.assert_allclose(q_test.rel_ref_strength, 0.001 * 0.2, rtol=0, atol=1e-12)

expected_knl = knl + knl_rel * q_test.rel_ref_strength
expected_ksl = ksl + ksl_rel * q_test.rel_ref_strength

expected_knl[1] += k1 * q_test.length
expected_ksl[1] += k1s * q_test.length

knl_tot, ksl_tot = q_test.get_total_knl_ksl()

xo.assert_allclose(knl_tot, expected_knl, rtol=0, atol=1e-12)
xo.assert_allclose(ksl_tot, expected_ksl, rtol=0, atol=1e-12)

q_ref = xt.Quadrupole(k1=k1, k1s=k1s, length=0.2,
                      knl=expected_knl, ksl=expected_ksl)
q_ref.knl[1] -= k1 * q_ref.length  # Not to double count the main quadrupole contribution
q_ref.ksl[1] -= k1s * q_ref.length

p0 = xt.Particles(p0c=1e9, x=1e-2, y=2e-2)

p_test = p0.copy()
q_test.track(p_test)

p_ref = p0.copy()
q_ref.track(p_ref)

xo.assert_allclose(p_test.x, p_ref.x, rtol=0, atol=1e-13)
xo.assert_allclose(p_test.y, p_ref.y, rtol=0, atol=1e-13)
xo.assert_allclose(p_test.px, p_ref.px, rtol=0, atol=1e-13)
xo.assert_allclose(p_test.py, p_ref.py, rtol=0, atol=1e-13)
