import time
import numpy as np
from scipy import constants
protonMass = constants.value('proton mass energy equivalent in MeV')*1E6
from matplotlib import pyplot as plt

import xobjects as xo
import xtrack as xt
from xtrack.pyheadtail_interface.pyhtxtparticles import PyHtXtParticles

from PyHEADTAIL.particles.slicing import UniformBinSlicer
from PyHEADTAIL.impedances.wakes import WakeTable, WakeField

context = xo.ContextCpu(omp_num_threads=0)
bunch_intensity = 2E11
n_macroparticles = int(1E4)
energy = 7E3 # [GeV]
gamma = energy*1E9/protonMass
betar = np.sqrt(1-1/gamma**2)
normemit = 2E-6
beta_x = 100.0
beta_y = 100.0
sigma_z = 7.5E-2
sigma_delta = 1E-4
beta_s = sigma_z/sigma_delta

particles = PyHtXtParticles(circumference=27E3,particlenumber_per_mp=bunch_intensity/n_macroparticles,
                         _context=context,
                         q0 = 1,
                         mass0 = protonMass,
                         gamma0 = gamma,
                         x=np.sqrt(normemit*beta_x/gamma/betar)*np.random.randn(n_macroparticles),
                         px=np.sqrt(normemit/beta_x/gamma/betar)*np.random.randn(n_macroparticles),
                         y=np.sqrt(normemit*beta_y/gamma/betar)*np.random.randn(n_macroparticles),
                         py=np.sqrt(normemit/beta_y/gamma/betar)*np.random.randn(n_macroparticles),
                         zeta=sigma_z*np.random.randn(n_macroparticles),
                         delta=sigma_delta*np.random.randn(n_macroparticles)
                         )

arc = xt.LinearTransferMatrixWithDetuning(alpha_x_0 = 0.0, beta_x_0 = beta_x, disp_x_0 = 0.0,
                           alpha_x_1 = 0.0, beta_x_1 = beta_x, disp_x_1 = 0.0,
                           alpha_y_0 = 0.0, beta_y_0 = beta_y, disp_y_0 = 0.0,
                           alpha_y_1 = 0.0, beta_y_1 = beta_y, disp_y_1 = 0.0,
                           Q_x = 0.31, Q_y=0.32,
                           beta_s = beta_s, Q_s = 0.002,
                           chroma_x = 10.0,chroma_y = 0.0,
                           energy_ref_increment=0.0,energy_increment=0)

n_slices_wakes = 100
limit_z = 3 * sigma_z
wakefile = 'wakes/wakeforhdtl_PyZbase_Allthemachine_7000GeV_B1_2021_TeleIndex1_wake.dat'
slicer_for_wakefields = UniformBinSlicer(n_slices_wakes, z_cuts=(-limit_z, limit_z))
waketable = WakeTable(wakefile, ['time', 'dipole_x', 'dipole_y', 'quadrupole_x', 'quadrupole_y'])
wake_field = WakeField(slicer_for_wakefields, waketable)

nTurn = int(1E4)
turns = np.arange(nTurn)
x = np.zeros(nTurn,dtype=float)
for turn in range(nTurn):
    time0 = time.time()
    arc.track(particles)
    time1 = time.time()
    wake_field.track(particles)
    time2 = time.time()
    x[turn] = np.average(particles.x)
    if turn%1000 == 0:
        print(f'{turn}: time for arc {time1-time0}s, for wake {time2-time1}s')

plt.figure(0)
plt.plot(turns,x)
plt.show()
    
