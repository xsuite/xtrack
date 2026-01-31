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
    rdt=['f1001', 'f1010'],
    twiss=tw0,
    strengths=line.get_table(attr=True)
)
tw_edw_teng = line.twiss4d(coupling_edw_teng=True)

xo.assert_allclose(rdts.f1001, tw_edw_teng.f1001, rtol=0.05)
xo.assert_allclose(rdts.f1010, tw_edw_teng.f1010, rtol=0.05)


# #   # Example error
# # line['kof.a56b1'] = 1000  # Example error



# tw = line.twiss4d(coupling_edw_teng=True)
# strengths = line.get_table(attr=True)
# f1020 = compute_rdt_first_order_perturbation('f1020', tw, strengths)
# f1001 = compute_rdt_first_order_perturbation('f1001', tw, strengths)
# f2020 = compute_rdt_first_order_perturbation('f2020', tw, strengths)

# tw_ng = line.madng_twiss(rdts=['f1020', 'f2020'])