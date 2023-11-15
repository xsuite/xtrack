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
tw0 = line.twiss(method='4d')

xtsl = xt.slicing
# slicing to see beta inside bends
line_sliced = line.copy()
line_sliced.slice_thick_elements(
    slicing_strategies=[
        xtsl.Strategy(slicing=None), # default
        xtsl.Strategy(slicing=xtsl.Teapot(100, mode='thick'), element_type=xt.Bend),
    ]
)

# tw = line.twiss(
#     ele_start='mqf.1', ele_stop=len(line)-1, twiss_init=xt.TwissInit(betx=1, bety=1))
tw = line_sliced.twiss(method='4d')

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)
ax1 = plt.subplot(2, 1, 1)
plt.plot(tw.s, tw.betx, label='betx')
plt.plot(tw.s, tw.bety, label='bety')
# plt.plot(tw0.s, tw0.betx, '.', color='C0')
# plt.plot(tw0.s, tw0.bety, '.', color='C1')
plt.xlabel('s [m]')
plt.ylabel(r'$\beta_{x,y}$ [m]')
plt.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.plot(tw.s, tw.dx, color='C2', label='dx')
ax2.set_ylabel(r'$D_x$ [m]')

tt = line.get_table(attr=True)
for ax in [ax1, ax2]:
    ttbends = tt.rows['mb.*']
    for ii in range(len(ttbends)):
        ss = ttbends.s[ii]
        ll = ttbends.length[ii]
        ax.axvspan(ss, ss+ll, facecolor='b', alpha=0.2)

    ttquads = tt.rows['mq.*']
    for ii in range(len(ttquads)):
        ss = ttquads.s[ii]
        ll = ttquads.length[ii]
        ax.axvspan(ss, ss+ll, facecolor='r', alpha=0.2)

sv = line_sliced.survey()
plt.figure(2)
plt.plot(sv.Z, sv.X, '.')

plt.show()