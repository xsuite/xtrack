import xtrack as xt
import numpy as np
import xobjects as xo

PI = np.pi

env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')

line = env.lhcb1
tw0 = line.twiss(method='4d')

# Coupling RDTs
line['kqs.a45b1'] = 1e-4 # Introduce a skew quadrupole
rdts = xt.rdt_first_order_perturbation(
    rdt=['f1001', 'f1010', 'f0110'],
    twiss=tw0,
    strengths=line.get_table(attr=True)
)
tw_edw_teng = line.twiss4d(coupling_edw_teng=True)
line['kqs.a45b1'] = 0.0 # Remove skew quadrupole

# outliers came from the fact that at the sources the RDT jumps at entry
# instead of exit of the element or viceversa
xo.assert_allclose(rdts.f1001, tw_edw_teng.f1001, rtol=0.05)
xo.assert_allclose(rdts.f1010, tw_edw_teng.f1010, rtol=0.05, max_outliers=5)
xo.assert_allclose(rdts.f0110, tw_edw_teng.f0110, rtol=0.05)
xo.assert_allclose(rdts.f1001, np.conj(rdts.f0110), rtol=1e-12)


# #   # Example error
# # line['kof.a56b1'] = 1000  # Example error



# tw = line.twiss4d(coupling_edw_teng=True)
# strengths = line.get_table(attr=True)
# f1020 = compute_rdt_first_order_perturbation('f1020', tw, strengths)
# f1001 = compute_rdt_first_order_perturbation('f1001', tw, strengths)
# f2020 = compute_rdt_first_order_perturbation('f2020', tw, strengths)

# tw_ng = line.madng_twiss(rdts=['f1020', 'f2020'])