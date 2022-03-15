import sys
sys.path.append('/afs/cern.ch/user/x/xbuffat/harpy')
from harmonic_analysis import HarmonicAnalysis

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

import xobjects as xo
import xtrack as xt
import xpart as xp

context = xo.ContextCpu()

q_x_set = .28
q_y_set = .31
Q_s_set = 0.001

beta_x = 1.0
beta_y = 10.0
alpha_x = -1.0
alpha_y = 8.0
energy = 45.6
beta_s = 800.0

gamma_x = (1.0+alpha_x**2)/beta_x
gamma_y = (1.0+alpha_y**2)/beta_y

el = xt.LinearTransferMatrix(_context=context,
        Q_x=q_x_set, Q_y=q_y_set, Q_s=Q_s_set,
        beta_s=beta_s,
        beta_x_0 = beta_x,beta_y_0 = beta_y,beta_x_1 = beta_x,beta_y_1= beta_y,
        alpha_x_0 = alpha_x,alpha_y_0 = alpha_y,alpha_x_1 = alpha_x,alpha_y_1= alpha_y,
        radiation_model = 'uncorrelated',
        damping_rate_x = 1E-3,damping_rate_y = 2E-3,damping_rate_z = 4E-3
        )

part = xp.Particles(_context=context, x=[1.0], y=[1.0], zeta=[1.0],
                    p0c=energy*1E9)

n_turns = 2048
x = np.zeros(n_turns,dtype=float)
px = np.zeros_like(x)
y = np.zeros_like(x)
py = np.zeros_like(x)
z = np.zeros_like(x)
delta = np.zeros_like(x)
emit_x = np.zeros_like(x)
emit_y = np.zeros_like(x)
emit_z = np.zeros_like(x)
for turn in range(n_turns):
    x[turn] = part.x[0]
    px[turn] = part.px[0]
    y[turn] = part.y[0]
    py[turn] = part.py[0]
    z[turn] = part.zeta[0]
    delta[turn] = part.delta[0]
    emit_x[turn] = 0.5*(gamma_x*part.x[0]**2+2*alpha_x*part.x[0]*part.px[0]+beta_x*part.px[0]**2)
    emit_y[turn] = 0.5*(gamma_y*part.y[0]**2+2*alpha_y*part.y[0]*part.py[0]+beta_y*part.py[0]**2)
    emit_z[turn] = 0.5*(part.zeta[0]**2/beta_s+beta_s*part.delta[0]**2)
    el.track(part)

plt.figure(0)
plt.plot(x,px,'x')
plt.figure(1)
plt.plot(y,py,'x')
plt.figure(2)
plt.plot(z,delta,'x')

turns = np.arange(n_turns)
fit_x = linregress(turns,np.log(emit_x))
fit_y = linregress(turns,np.log(emit_y))
fit_z = linregress(turns,np.log(emit_z))
print(fit_x.slope,fit_y.slope,fit_z.slope)
plt.figure(10)
plt.plot(turns,emit_x)
plt.plot(turns,np.exp(fit_x.intercept+fit_x.slope*turns),'--k')
plt.plot(turns,emit_y)
plt.plot(turns,np.exp(fit_y.intercept+fit_y.slope*turns),'--k')
plt.figure(11)
plt.plot(turns,emit_z)
plt.plot(turns,np.exp(fit_z.intercept+fit_z.slope*turns),'--k')

plt.show()
