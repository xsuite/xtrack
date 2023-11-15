import numpy as np
import xtrack as xt

pi = np.pi

elements = {
    'mqf.1': xt.Quadrupole(length=0.3, k1=0.1),
    'd1.1':  xt.Drift(length=1),
    'mb1.1': xt.Bend(length=2, k0=pi / 2, h=pi / 2),
    'd2.1':  xt.Drift(length=1),

    'mqd.1': xt.Quadrupole(length=0.3, k1=-0.1),
    'd3.1':  xt.Drift(length=1),
    'mb2.1': xt.Bend(length=2, k0=pi / 2, h=pi / 2),
    'd4.1':  xt.Drift(length=1),

    'mqf.2': xt.Quadrupole(length=0.3, k1=0.1),
    'd1.2':  xt.Drift(length=1),
    'mb1.2': xt.Bend(length=2, k0=pi / 2, h=pi / 2),
    'd2.2':  xt.Drift(length=1),

    'mqd.2': xt.Quadrupole(length=0.3, k1=-0.1),
    'd3.2':  xt.Drift(length=1),
    'mb2.2': xt.Bend(length=2, k0=pi / 2, h=pi / 2),
    'd4.2':  xt.Drift(length=1),
}

# Build the ring
line = xt.Line(elements=elements, element_names=list(elements.keys()))
line.particle_ref = xt.Particles(p0c=1.2e9, mass0=xt.PROTON_MASS_EV)

# tw = line.twiss(
#     ele_start='mqf.1', ele_stop=len(line)-1, twiss_init=xt.TwissInit(betx=1, bety=1))
tw = line.twiss(method='4d')

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
plt.plot(tw.s, tw.betx, label='betx')
plt.plot(tw.s, tw.bety, label='bety')
plt.xlabel('s [m]')
plt.ylabel(r'$\beta_{x,y}$ [m]')
plt.legend()

tt = line.get_table(attr=True)
ttbends = tt.rows['mb.*']
for ii in range(len(ttbends)):
    ss = ttbends.s[ii]
    ll = ttbends.length[ii]
    plt.axvspan(ss, ss+ll, facecolor='b', alpha=0.2)

ttquads = tt.rows['mq.*']
for ii in range(len(ttquads)):
    ss = ttquads.s[ii]
    ll = ttquads.length[ii]
    plt.axvspan(ss, ss+ll, facecolor='r', alpha=0.2)


plt.show()