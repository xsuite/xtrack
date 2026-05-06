import xtrack as xt
import numpy as np

env = xt.load(['../../test_data/lhc_2024/lhc.seq',
               '../../test_data/lhc_2024/injection_optics.madx'],
                reverse_lines=['lhcb2'])


sv1 = env.lhcb1.survey()
sv2 = env.lhcb2.survey(theta0=np.pi)

tw1 = env.lhcb1.twiss4d()
tw2 = env.lhcb2.twiss4d()

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(sv1.Z, sv1.X, color='red', label='B1', linestyle='--')
plt.plot(sv2.Z, sv2.X, color='blue', label='B2', linestyle='--')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')

plt.show()