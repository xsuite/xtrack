import xtrack as xt
import numpy as np
import xobjects as xo

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

# RDTs from chromaticity sextupoles
rdt_names = ['f3000', 'f1200', 'f1020', 'f0120', 'f0111']
rdts = xt.rdt_first_order_perturbation(
    rdt=rdt_names,
    twiss=tw0,
    strengths=line.get_table(attr=True)
)
tw_ng = line.madng_twiss(rdts=rdt_names)
for nn in rdt_names:
    # outliers came from the fact that at the sources the RDT jumps at entry
    # instead of exit of the element or viceversa
    xo.assert_allclose(rdts[nn], tw_ng[nn],
                       atol=0.1*np.max(np.abs(tw_ng[nn])),
                       max_outliers=0.01*len(tw_ng))

line['on_x1'] = 0
line['on_x5'] = 250
line['on_disp'] = 2
tw1 = line.twiss(method='4d', coupling_edw_teng=True)
rdt_names = ['f1001', 'f1010', 'f0110']
rdts = xt.rdt_first_order_perturbation(
    rdt=rdt_names,
    twiss=tw0,
    orbit=tw1,
    strengths=line.get_table(attr=True)
)
for nn in rdt_names:
    # outliers came from the fact that at the sources the RDT jumps at entry
    # instead of exit of the element or viceversa
    xo.assert_allclose(rdts[nn], tw1[nn],
                       atol=0.06*np.max(np.abs(tw1[nn])),
                       max_outliers=0.005*len(tw1))
