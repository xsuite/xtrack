import numpy as np
import xtrack as xt

k0 = np.pi/200
k1 = 0.05
length = 1

k1quad = 1.6
lengthquad = 0.2

lengthdrift = 0.5

bcoeffs = np.array([[k0], [k1]])
combinedfunction = xt.FieldExpansion(length=length, a=np.array([[0]]), b=bcoeffs, bs=np.array([0]), ny=5, nstep=10)
quad1 = xt.Quadrupole(k1=k1quad, length=lengthquad)
quad2 = xt.Quadrupole(k1=-k1quad, length=lengthquad)
drift = xt.Drift(length=lengthdrift)

fodo = xt.Line(elements=[quad1, drift, combinedfunction, drift, quad2, drift, combinedfunction, drift])
fodo.set_particle_ref()
tw = fodo.twiss4d()
