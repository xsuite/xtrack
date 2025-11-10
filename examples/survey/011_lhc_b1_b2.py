import xtrack as xt
import numpy as np

env = xt.load(['../../test_data/lhc_2024/lhc.seq',
               '../../test_data/lhc_2024/injection_optics.madx'],
                reverse_lines=['lhcb2'])

sv1 = env.lhcb1.survey(element0='ip5')
sv2_rev = env.lhcb2.survey(element0='ip5').reverse()

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(sv1['Z'], sv1['X'], label='lhcb1')
plt.plot(sv2_rev['Z'], sv2_rev['X'], label='lhcb2 (reversed)')
plt.xlabel('s [m]')
plt.ylabel('x [m]')
plt.legend()
plt.grid()
plt.title('Survey of LHC Elements')
plt.show()
