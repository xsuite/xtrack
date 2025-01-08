# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
import matplotlib.pyplot as plt

# https://gitlab.cern.ch/e-beam/pybetacool
from pybetacool import PyBetaCool

# unofficial Betacool: https://github.com/dgamba/betacool
obj = PyBetaCool(BLDfilename='LEIR.bld', betacoolExec='/home/pkruyt/cernbox/BETACOOL_ecooling/betacool/Betacool_Linux')

# Execute Betacool force
obj.runBetacool('/f')

#################################
# parse file and plot data
#################################

tmp = obj.parseCurveFile('flong.cur')
tmp = tmp[:(100000)]

v_diff=tmp[:, 0]
force=tmp[:,1]

plt.figure()
plt.plot(v_diff,-force, label='betacool')
plt.xlabel('delta v')
plt.ylabel('-Force [eV/m]')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.legend()
plt.xlim([0,6e5])
plt.show()


#################################
# save data as reference for test
#################################
np.savez('../force_betacool.npz', v_diff=v_diff, force=force)