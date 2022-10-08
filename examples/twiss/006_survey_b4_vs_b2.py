import numpy as np
from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

mad_b2 = Madx()
mad_b2.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad_b2.use(sequence='lhcb2')
twb2mad = mad_b2.twiss()
summb2mad = mad_b2.table.summ
survb2mad = mad_b2.survey().dframe()

mad_b4 = Madx()
mad_b4.call('../../test_data/hllhc15_noerrors_nobb/sequence_b4.madx')
mad_b4.use(sequence='lhcb2')
twb4mad = mad_b4.twiss()
summb4mad = mad_b4.table.summ
survb4mad = mad_b4.survey().dframe()

line_b4 = xt.Line.from_madx_sequence(mad_b4.sequence['lhcb2'],
                                     #deferred_expressions=True
                                     )
line_b4.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, p0c=7000e9)

tracker_b4 = xt.Tracker(line=line_b4)
twb4xt = tracker_b4.twiss()
survb4xt = tracker_b4.survey().to_pandas(index='name')

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(survb2mad['s'], survb2mad['x'], 'r')
plt.plot(survb2mad.loc['ip2', 's'], survb2mad.loc['ip2', 'x'], 'o', color='r')
plt.plot(survb4mad['s'], survb4mad['x'], 'lightblue')
plt.plot(survb4mad.loc['ip2', 's'], survb4mad.loc['ip2', 'x'], 'o', color='lightblue')
plt.plot(survb4xt.s, survb4xt.X, '--', color='b')
plt.plot(survb4xt.loc['ip2', 's'], survb4xt.loc['ip2', 'X'], 'o', color='b')
plt.subplot(2,1,2)
plt.plot(survb2mad['s'], survb2mad['z'], 'r', lw=5)
plt.plot(survb4mad['s'], survb4mad['z'], 'lightblue')
plt.plot(survb4xt.s, survb4xt.Z, '--', color='b')
plt.show()

from xtrack.survey_from_tracker import SurveyTable
def temp_mirror(self):
    new = SurveyTable()
    for kk, vv in self.items():
        new[kk] = vv

    for kk in new.keys():
        new[kk] = new[kk][::-1]

    new.s = new.s[0] - new.s

    return new

survb2xt = temp_mirror(tracker_b4.survey())