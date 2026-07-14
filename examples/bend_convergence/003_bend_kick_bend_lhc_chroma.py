import xtrack as xt

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tw_rot_kick_rot = line.twiss4d()

line.configure_bend_model(core='bend-kick-bend')
tw_bend_kick_bend = line.twiss4d()

import matplotlib.pyplot as plt
tw_bend_kick_bend.plot('x')

print("Q'x:")
print(f"  rot-kick-rot:   {tw_rot_kick_rot.dqx:.10e}")
print(f"  bend-kick-bend: {tw_bend_kick_bend.dqx:.10e}")
print("Q'y:")
print(f"  rot-kick-rot:   {tw_rot_kick_rot.dqy:.10e}")
print(f"  bend-kick-bend: {tw_bend_kick_bend.dqy:.10e}")

plt.show()