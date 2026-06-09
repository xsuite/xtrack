import xtrack as xt

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line['vrf400'] = 16.

tw_rot_kick_rot_4d = line.twiss4d()
tw_rot_kick_rot_6d = line.twiss6d()

line.configure_bend_model(core='bend-kick-bend')
tw_bend_kick_bend_4d = line.twiss4d()
tw_bend_kick_bend_6d = line.twiss6d()


import matplotlib.pyplot as plt
tw_bend_kick_bend_4d.plot('x')

print("Q'x:")
print(f"  rot-kick-rot:   {tw_rot_kick_rot_4d.dqx:.10e}")
print(f"  bend-kick-bend: {tw_bend_kick_bend_4d.dqx:.10e}")
print("Q'y:")
print(f"  rot-kick-rot:   {tw_rot_kick_rot_4d.dqy:.10e}")
print(f"  bend-kick-bend: {tw_bend_kick_bend_4d.dqy:.10e}")

plt.show()