import xtrack as xt
import numpy as np
import matplotlib.pyplot as plt

"""
Example for a quadrupole thin fringe and a full magnet with fringe fields
"""

# Fringe from 0 to 0.1 over length 0.05, with derivatives zero at the endpoints
# Fringe shape is specified as polynomial in s, lowest coefficient first
# First row is dipole coefficients, second row is quadrupole coefficients
bentrance=np.array([[0,0,0,0], [0,0,120,-1600]]) 
fringelength = 0.05
bmax = 0.1

entranceFringe = xt.FieldExpansion(length=fringelength, a=np.array([[0,0,0,0]]), b=bentrance, bs=np.array([0,0,0,0]), ny=10)
invDrift = xt.FieldExpansion(length=-fringelength/2, a=np.array([[0]]), b=np.array([[0]]), bs=np.array([0]), ny=5)
invQuad = xt.FieldExpansion(length=-fringelength/2, a=np.array([[0]]), b=np.array([[0], [bmax]]), bs=np.array([0]), ny=5)
thinFringe = xt.Line(elements=[invDrift, entranceFringe, invQuad])


# Full quadrupole including fringes
bexit = np.array([[0,0,0,0], [0.1,0,-120,1600]])
exitFringe = xt.FieldExpansion(length=fringelength, a=np.array([[0,0,0,0]]), b=bexit, bs=np.array([0,0,0,0]), ny=10)

magnlength = 1
bodylength = magnlength - fringelength  # Magnetic length up to center of fringe fields (symmetry)
body = xt.Quadrupole(k1=bmax, length=bodylength)  # Can also be FieldExpansion element
fringeQuad = xt.Line(elements=[entranceFringe, body, exitFringe])

# Check if does what is expected
Quad = xt.Quadrupole(k1=bmax, length=magnlength)
driftQuad = xt.Line(elements=[xt.Drift(length=fringelength/2), Quad, xt.Drift(length=fringelength/2)])

p0 = xt.Particles(x=np.linspace(-0.01, 0.01, 10))
p1 = p0.copy()

fringeQuad.track(p0)
driftQuad.track(p1)

# Both have same magnetic length, but one does not include fringe fields. The focussing is very similar.
plt.scatter(p0.x, p0.px, label='Fringe quad')
plt.scatter(p1.x, p1.px, label='Drift quad', marker='x')




