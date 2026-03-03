import xtrack as xt
import xobjects as xo
import numpy as np

knl = np.array([0.001, 1e-4, 2e-3, 3e-2, 4e-1, 5])
ksl = np.array([0.002, 2e-4, 3e-3, 4e-2, 5e-1, 6])
knl_rel = np.array([1, 20, 30, 40, 50, 60])
ksl_rel = np.array([2, 40, 60, 80, 100, 120])
k1 = 0.001
k1s = 0.002

q_test = xt.Quadrupole(k1=k1, k1s=k1s, length=0.2,
                       knl=knl,
                       ksl=ksl,
                       knl_rel=knl_rel,
                       ksl_rel=ksl_rel)

xo.assert_allclose(q_test.rel_ref_strength, 0.001 * 0.2, rtol=0, atol=1e-12)

expected_knl = knl + knl_rel * q_test.rel_ref_strength
expected_ksl = ksl + ksl_rel * q_test.rel_ref_strength

expected_knl[1] += k1 * q_test.length
expected_ksl[1] += k1s * q_test.length

knl_tot, ksl_tot = q_test.get_total_knl_ksl()

xo.assert_allclose(knl_tot, expected_knl, rtol=0, atol=1e-12)
xo.assert_allclose(ksl_tot, expected_ksl, rtol=0, atol=1e-12)