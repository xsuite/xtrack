import xtrack as xt
import xpart as xp

from cpymad.madx import Madx

mad=Madx()
mad.call('../../test_data/hllhc15_thick/lhc.seq')
mad.call('../../test_data/hllhc15_thick/hllhc_sequence.madx')
mad.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
mad.use('lhcb1')
mad.input('beam, sequence=lhcb2, particle=proton, energy=7000, bv=-1;')
mad.use('lhcb2')
mad.call("../../test_data/hllhc15_thick/opt_round_150_1500.madx")
mad.twiss()

mad.call('../../../hllhc15/toolkit/macro.madx')
mad.call('make_one_crossing_knob.madx')

mad.use('lhcb1')
mad.use('lhcb2')

mad.globals['on_x8'] = 100
mad.input('twiss, sequence=lhcb1, table=twb1;')
mad.input('twiss, sequence=lhcb2, table=twb2;')

import xdeps as xd
twb1 = xd.Table(mad.table.twb1)
twb2 = xd.Table(mad.table.twb2)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(twb1.s, twb1.x, label='x')
plt.plot(twb1.s, twb1.y, label='y')
plt.plot(twb2.s, twb2.x, label='x')
plt.plot(twb2.s, twb2.y, label='y')

plt.show()