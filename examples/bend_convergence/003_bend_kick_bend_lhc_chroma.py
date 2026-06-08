import xtrack as xt

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tw = line.twiss4d()

line.configure_bend_model(core='bend-kick-bend')
tw_bkb = line.twiss4d()

import matplotlib.pyplot as plt
tw_bkb.plot('x')

plt.show()