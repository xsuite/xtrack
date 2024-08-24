import xtrack as xt
import numpy as np

elements={
    'drift0': xt.Drift(length=1.),
    'qf1': xt.Quadrupole(k1=0.027 / 2, length=1.),
    'drift1_1': xt.Drift(length=1),
    'bend1': xt.Bend(k0=3e-4, h=3e-4, length=45./2),
    'drift1_2': xt.Drift(length=1.),
    'qd1': xt.Quadrupole(k1=-0.0271 / 2, length=1.),
    'drift2': xt.Drift(length=1),
    'drift3': xt.Drift(length=1),
    'qd2': xt.Quadrupole(k1=-0.0271 / 2, length=1.),
    'drift4_1': xt.Drift(length=1),
    'bend2': xt.Bend(k0=3e-4, h=3e-4, length=45./2),
    'drift4_2': xt.Drift(length=1),
    'qf2': xt.Quadrupole(k1=0.027 / 2, length=1.),
    'drift5': xt.Drift(length=0.1),
}

line = xt.Line(elements=elements, element_names=list(elements.keys()))
line.particle_ref = xt.Particles(p0c=2e9)
tw = line.twiss4d(strengths=True)

line_half = xt.Line(elements=elements,
                    element_names=['drift0', 'qf1', 'drift1_1', 'bend1',
                                   'drift1_2', 'qd1', 'drift2'])
line_half.particle_ref = line.particle_ref.copy()
# tw_half = line_half.twiss4d(strengths=True, init='periodic_symmetric')

import matplotlib.pyplot as plt
plt.close('all')

twplt1 = tw.plot()
twplt1.ylim(left_lo=0, right_lo=0.5, right_hi=4)
# twplt2 = tw_half.plot()

plt.show()
