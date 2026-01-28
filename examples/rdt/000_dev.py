import xtrack as xt
import numpy as np
from rdt_first_order import compute_rdt_first_order_perturbation

from math import factorial

PI = np.pi

env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')

line = env.lhcb1

line['kqs.a45b1'] = 1e-4  # Example error
line['kof.a56b1'] = 1000  # Example error
tw = line.twiss4d(coupling_edw_teng=True)
strengths = line.get_table(attr=True)
f1020 = compute_rdt_first_order_perturbation('f1020', tw, strengths)
f1001 = compute_rdt_first_order_perturbation('f1001', tw, strengths)
f2020 = compute_rdt_first_order_perturbation('f2020', tw, strengths)

tw_ng = line.madng_twiss(rdts=['f1020', 'f2020'])