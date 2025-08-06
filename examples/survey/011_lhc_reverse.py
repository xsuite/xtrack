import xtrack as xt
import numpy as np

env = xt.load_madx_lattice('../../test_data/lhc_2024/lhc.seq', reverse_lines=['lhcb2'])
env.vars.load('../../test_data/lhc_2024/injection_optics.madx')

sv1 = env.lhcb1.survey(element0='ip5')
sv2_rev = env.lhcb2.survey(element0='ip5').reverse()


