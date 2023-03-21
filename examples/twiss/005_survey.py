import numpy as np
from cpymad.madx import Madx
import xtrack as xt
import xpart as xp

mad = Madx()
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use(sequence='lhcb1')
tw_mad = mad.twiss()
summ_mad = mad.table.summ
surv_mad = mad.survey()

line = xt.Line.from_madx_sequence(mad.sequence['lhcb1'],
                                     deferred_expressions=True)
line.build_tracker()
surv_xt = line.survey()

import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(surv_mad.s, surv_mad.X, 'b')
plt.plot(surv_xt.s, surv_xt.X, '--', color='lightblue')
plt.plot(surv_mad.s, surv_mad.Z, 'r')
plt.plot(surv_xt.s, surv_xt.Z, '--', color='darkred')

plt.figure(2)
plt.plot(surv_mad.s, surv_mad.phi, 'b')
plt.plot(surv_xt.s, surv_xt.phi, '--', color='lightblue')
plt.plot(surv_mad.s, surv_mad.theta, 'r')
plt.plot(surv_xt.s, surv_xt.theta, '--', color='darkred')
plt.plot(surv_mad.s, surv_mad.psi, 'g')
plt.plot(surv_xt.s, surv_xt.psi, '--', color='darkgreen')

plt.show()
