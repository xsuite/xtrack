import bpmeth
import xtrack as xt
import sympy

import numpy as np

b1 = "0.2"
b2 = "0.1"
h = "0.1"
length = 0.5

s = sympy.symbols('s')
b1_f = sympy.lambdify(s, b1, modules='numpy')

# Vector potential (bs, b1, a1, b2, a2, b3, a3, ,... as function of s)
A_magnet = bpmeth.GeneralVectorPotential(hs=h, b=(b1,b2))

class MyHamiltonian(bpmeth.Hamiltonian):
    def get_H(self, coords):
        x, y, s = coords.x, coords.y, coords.s
        px, py, ptau = coords.px, coords.py, coords.ptau
        beta0 = coords.beta0
        h = self.curv
        one_plus_delta_allsq = 1+2*ptau/beta0+ptau**2
        sqrt = coords._m.sqrt
        term1 = ptau / beta0
        term2 = 0.5 * (px**2 + py**2) / sqrt(one_plus_delta_allsq)
        term3 = (float(b1) - h) * x + float(b1) * h * x**2 / 2
        term4 = float(b2) * (x**2 - y**2) / 2

        term_correction = float(b2) * h * (2*x**3 - 3*x*y**2) / 6

        H = term1 + term2 + term3 + term4 + term_correction
        return H

# Make Hamiltonian for a defined length
H_magnet = MyHamiltonian(length=length, curv=float(h), vectp=A_magnet)

# This guy is able to track an Xsuite particle!
p0 = xt.Particles(x=np.linspace(-1e-3, 1e-3, 10),
                  y=np.linspace(-2e-3, 2e-3, 10),
                   energy0=10e9, mass0=xt.ELECTRON_MASS_EV)

p_silke = p0.copy()
sol = H_magnet.track(p_silke, return_sol=True,
                     ivp_opt= {"rtol":1e-10, "atol":1e-12})

p_xsuite = p0.copy()
bb = xt.Bend(h=float(h), length=length, k0=float(b1), k1=float(b2))
bb.edge_entry_active = False
bb.edge_exit_active = False

bb.num_multipole_kicks = 100
bb.integrator = 'yoshida4'

bb.model =  'bend-kick-bend'
bb.model =  'mat-kick-mat'

bb.track(p_xsuite)

# Print the max difference
print(f'Max difference on x: {np.max(np.abs(p_silke.x - p_xsuite.x))}')

# I want to see the trajectory along the object
import matplotlib.pyplot as plt
plt.close('all')
plt.figure()
for ss in sol:
    plt.plot(ss.t, ss.y[0])
plt.xlabel('s [m]')
plt.ylabel('x [m]')

plt.show()
