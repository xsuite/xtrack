import bpmeth
import xtrack as xt
import sympy

import numpy as np

b1 = "0.12"
h = "0.1"
length = 0.5

s = sympy.symbols('s')
b1_f = sympy.lambdify(s, b1, modules='numpy')

# Vector potential (bs, b1, a1, b2, a2, b3, a3, ,... as function of s)
A_magnet = bpmeth.GeneralVectorPotential(hs=h, b=(b1,))

# Make Hamiltonian for a defined length
H_magnet = bpmeth.Hamiltonian(length=length, curv=float(h), vectp=A_magnet)

# This guy is able to track an Xsuite particle!
p0 = xt.Particles(x=np.linspace(-1e-3, 1e-3, 10), energy0=10e9, mass0=xt.ELECTRON_MASS_EV)

p_silke = p0.copy()
sol = H_magnet.track(p_silke, return_sol=True)

p_xsuite = p0.copy()
bb = xt.Bend(h=float(h), length=length, k0=float(b1))

bb.model =  'bend-kick-bend'
bb.track(p_xsuite)

# I want to see the trajectory along the object
import matplotlib.pyplot as plt
plt.close('all')
plt.figure()
for ss in sol:
    plt.plot(ss.t, ss.y[0])
plt.xlabel('s [m]')
plt.ylabel('x [m]')

plt.show()
