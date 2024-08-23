import xtrack as xt

elements={
    'drift0': xt.Drift(length=0.1),
    'qf1': xt.Quadrupole(k1=0.01, length=0.1),
    'drift1': xt.Drift(length=0.6),
    'qd1': xt.Quadrupole(k1=-0.01, length=0.1),
    'drift2': xt.Drift(length=0.1),
    'drift3': xt.Drift(length=0.1),
    'qd2': xt.Quadrupole(k1=-0.01, length=0.1),
    'drift4': xt.Drift(length=0.6),
    'qf2': xt.Quadrupole(k1=0.01, length=0.1),
    'drift5': xt.Drift(length=0.1),
}

line = xt.Line(elements=elements, element_names=list(elements.keys()))
line.particle_ref = xt.Particles(p0c=2e9)
tw = line.twiss4d(strengths=True)

import matplotlib.pyplot as plt
plt.close('all')
tw.plot()

plt.show()