import xtrack as xt
import numpy as np

env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')

del env.lhcb2.twiss_default['reverse']

env.ref['on_x1hs'] = 10
env.ref['on_x5hs'] = 10

sv1 = env.lhcb1.survey(element0='ip5')
sv2 = env.lhcb2.survey(theta0=np.pi, element0='ip5')

tw1 = env.lhcb1.twiss4d()
tw2 = env.lhcb2.twiss4d()

pb1_global = tw1.x[:, None] * sv1.ex + tw1.y[:, None] * sv1.ey + sv1.p0
pb2_global = tw2.x[:, None] * sv2.ex + tw2.y[:, None] * sv2.ey + sv2.p0

xb1_global = pb1_global[:, 0]
xb2_global = pb2_global[:, 0]
zb1_global = pb1_global[:, 2]
zb2_global = pb2_global[:, 2]

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(sv1.Z, sv1.X, color='blue', label='B1', linestyle='--')
plt.plot(sv2.Z, sv2.X, color='red', label='B2', linestyle='--')
plt.plot(zb1_global, xb1_global, color='blue', label='B1 global')
plt.plot(zb2_global, xb2_global, color='red', label='B2 global')
plt.xlabel('Z [m]')
plt.ylabel('X [m]')

plt.show()