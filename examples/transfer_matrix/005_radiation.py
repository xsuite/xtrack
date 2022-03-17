import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

import xobjects as xo
import xtrack as xt
import xpart as xp

context = xo.ContextCpu()

q_x_set = .18
q_y_set = .22
Q_s_set = 0.025

beta_x = 1.0
beta_y = 10.0
alpha_x = -10.0
alpha_y = 1000.0
energy = 45.6

gamma_x = (1.0+alpha_x**2)/beta_x
gamma_y = (1.0+alpha_y**2)/beta_y

damping_rate_x = 5E-4
damping_rate_y = 1E-3
damping_rate_s = 2E-3

equ_emit_x = 0.3E-9
equ_emit_y = 1E-12
equ_length = 3.5E-3
equ_delta = 3.8E-4
beta_s = equ_length/equ_delta
equ_emit_s = equ_length*equ_delta

el = xt.LinearTransferMatrix(_context=context,
        Q_x=q_x_set, Q_y=q_y_set, Q_s=Q_s_set,
        beta_s=beta_s,
        beta_x_0 = beta_x,beta_y_0 = beta_y,beta_x_1 = beta_x,beta_y_1= beta_y,
        alpha_x_0 = alpha_x,alpha_y_0 = alpha_y,alpha_x_1 = alpha_x,alpha_y_1= alpha_y,
        damping_rate_x = damping_rate_x,damping_rate_y = damping_rate_y,damping_rate_s = damping_rate_s,
        equ_emit_x = equ_emit_x, equ_emit_y = equ_emit_y, equ_emit_s = equ_emit_s
        )

part = xp.Particles(_context=context, x=[10*np.sqrt(equ_emit_x*beta_x)], y=[10*np.sqrt(equ_emit_y*beta_y)], zeta=[10*np.sqrt(equ_emit_s*beta_s)],
                    p0c=energy*1E9)

part._init_random_number_generator();

n_turns = int(1E5)
x = np.zeros(n_turns,dtype=float)
px = np.zeros_like(x)
y = np.zeros_like(x)
py = np.zeros_like(x)
z = np.zeros_like(x)
delta = np.zeros_like(x)
emit_x = np.zeros_like(x)
emit_y = np.zeros_like(x)
emit_s = np.zeros_like(x)
for turn in range(n_turns):
    x[turn] = part.x[0]
    px[turn] = part.px[0]
    y[turn] = part.y[0]
    py[turn] = part.py[0]
    z[turn] = part.zeta[0]
    delta[turn] = part.delta[0]
    emit_x[turn] = 0.5*(gamma_x*part.x[0]**2+2*alpha_x*part.x[0]*part.px[0]+beta_x*part.px[0]**2)
    emit_y[turn] = 0.5*(gamma_y*part.y[0]**2+2*alpha_y*part.y[0]*part.py[0]+beta_y*part.py[0]**2)
    emit_s[turn] = 0.5*(part.zeta[0]**2/beta_s+beta_s*part.delta[0]**2)
    el.track(part)

plt.figure(0)
plt.plot(x,px,'x')
plt.figure(1)
plt.plot(y,py,'x')
plt.figure(2)
plt.plot(z,delta,'x')

turns = np.arange(n_turns)
fit_range = 1000
fit_x = linregress(turns[:fit_range],np.log(emit_x[:fit_range]))
fit_y = linregress(turns[:fit_range],np.log(emit_y[:fit_range]))
fit_s = linregress(turns[:fit_range],np.log(emit_s[:fit_range]))
print(-fit_x.slope/damping_rate_x,-fit_y.slope/damping_rate_y,-fit_s.slope/damping_rate_s)
averga_start = int(3E4)
emit_x_0 = np.average(emit_x[averga_start:])
emit_y_0 = np.average(emit_y[averga_start:])
emit_s_0 = np.average(emit_s[averga_start:])
length_0 = np.std(z[averga_start:])
print(emit_x_0/equ_emit_x,emit_y_0/equ_emit_y,emit_s_0/equ_emit_s,length_0/equ_length)
plt.figure(10)
plt.plot(turns,emit_x)
plt.plot(turns,np.exp(fit_x.intercept+fit_x.slope*turns),'--k')
plt.plot([turns[0],turns[-1]],[emit_x_0,emit_x_0],'--k')
plt.plot(turns,emit_y)
plt.plot(turns,np.exp(fit_y.intercept+fit_y.slope*turns),'--k')
plt.plot([turns[0],turns[-1]],[emit_y_0,emit_y_0],'--k')
plt.figure(11)
plt.plot(turns,emit_s)
plt.plot(turns,np.exp(fit_s.intercept+fit_s.slope*turns),'--k')
plt.plot([turns[0],turns[-1]],[emit_s_0,emit_s_0],'--k')

plt.show()
