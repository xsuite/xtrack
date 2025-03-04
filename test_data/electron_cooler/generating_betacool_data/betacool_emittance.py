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

# Execute Betacool emittance
obj.runBetacool('/m')

#################################
# parse file and plot data
#################################

tmp  = obj.parseCurveFile('emittance.cur')
time = tmp[:,0]
emittance = tmp[:,1]*1e-6

plt.figure()
plt.plot(time,emittance, label='emittance')
plt.title('betacool')
plt.ylabel('Emittance')
plt.xlabel('Time')
plt.legend()
plt.show()

#################################
# save data as reference for test
#################################
filpath='../emittance_betacool.npz'
np.savez(filpath, time=time, emittance=emittance)