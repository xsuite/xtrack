"""
Compare an exact thin fringe created as inverse drift - exact fringe - inverse bend to the Forest fringe
"""

import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt

# Exact fringe from 0 to 0.1 over length 0.05, with derivatives zero at the endpoints
# Fringe shape is specified as polynomial in s, lowest coefficient first
bfringe=np.array([[0,0,120,-1600]])
length = 0.05
bmax = 0.1

exactFringe = xt.FieldExpansion(length=length, a=np.array([[0,0,0,0]]), b=bfringe, bs=np.array([0,0,0,0]), ny=10)
invDrift = xt.FieldExpansion(length=-length/2, a=np.array([[0]]), b=np.array([[0]]), bs=np.array([0]), ny=5)
invBend = xt.FieldExpansion(length=-length/2, a=np.array([[0]]), b=np.array([[bmax]]), bs=np.array([0]), ny=5)
thinFringe = xt.Line(elements=[invDrift, exactFringe, invBend])

# Xsuite fringe with same parameters
gap = 0.04
ss = np.linspace(0,length,100)
bvals = bfringe[0,0]+bfringe[0,1]*ss+bfringe[0,2]*ss**2+bfringe[0,3]*ss**3
fint = np.trapezoid((bmax - bvals)*bvals / bmax**2 / gap, ss)
PTCfringe = xt.Bend(length=0, k0=0.1, edge_entry_model="full", edge_entry_fint=fint, edge_entry_hgap=gap/2, edge_exit_active=0)

# Check effect of vertical offset including "SAD"-term: one can observe a deviation at large y values
yy = np.linspace(-0.1, 0.1, 100)
p0 = xt.Particles(y=yy, x=0, px=0, py=0, zeta=0, delta=0, beta0=1)
p1 = p0.copy()

thinFringe.track(p0, _force_no_end_turn_actions=True)
PTCfringe.track(p1)

plt.plot(yy, p0.py, label='Thin fringe')
plt.plot(yy, p1.py, label='PTC fringe')
plt.legend()
plt.xlabel('y')
plt.ylabel('py')

# Effect of delta at given y
dd = np.linspace(-0.1, 0.1, 100)
p0 = xt.Particles(y=0.002, x=0, px=0, py=0, zeta=0, delta=dd, beta0=1)
p1 = p0.copy()

thinFringe.track(p0, _force_no_end_turn_actions=True)
PTCfringe.track(p1)

plt.plot(dd, p0.py, label='Thin fringe')
plt.plot(dd, p1.py, label='PTC fringe')
plt.legend()
plt.xlabel('delta')
plt.ylabel('py')