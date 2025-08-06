import xtrack as xt
import numpy as np

env = xt.load_madx_lattice('../../test_data/lhc_2024/lhc.seq', reverse_lines=['lhcb2'])
env.vars.load('../../test_data/lhc_2024/injection_optics.madx')

sv1 = env.lhcb1.survey()
sv2_rev = env.lhcb2.survey().reverse()

from cpymad.madx import Madx
madx = Madx()
madx.call('../../test_data/lhc_2024/lhc.seq')
madx.call('../../test_data/lhc_2024/injection_optics.madx')
madx.beam(sequence='lhcb2', bv=-1)
madx.beam(sequence='lhcb1', bv=1)

madx.use('lhcb2')

madx.survey(sequence='lhcb2')
sv2_madx = xt.Table(madx.table.survey, _copy_cols=True)




