import xtrack as xt
import numpy as np

cell = xt.Line(
    elements={
        'drift0': xt.Drift(length=1.),
        'qf1':    xt.Quadrupole(k1=0.027/2, length=1.),
        'drift1': xt.Drift(length=1),
        'bend1':  xt.Bend(k0=3e-4, h=3e-4, length=45.),
        'drift2': xt.Drift(length=1.),
        'qd1':    xt.Quadrupole(k1=-0.0271/2, length=1.),
        'drift3': xt.Drift(length=1),
        'drift4': xt.Replica('drift3'),
        'qd2':    xt.Replica('qd1'),
        'drift5': xt.Replica('drift2'),
        'bend2':  xt.Replica('bend1'),
        'drift6': xt.Replica('drift1'),
        'qf2':    xt.Replica('qf1'),
        'drift7': xt.Replica('drift0'),
    }
)
cell.particle_ref = xt.Particles(p0c=2e9)
tw = cell.twiss4d(strengths=True)

half_cell = xt.Line(
    elements={
        'drift0': xt.Drift(length=1.),
        'qf1': xt.Quadrupole(k1=0.027/2, length=1.),
        'drift1_1': xt.Drift(length=1),
        'bend1': xt.Bend(k0=3e-4, h=3e-4, length=45.),
        'drift1_2': xt.Drift(length=1.),
        'qd1': xt.Quadrupole(k1=-0.0271/2, length=1.),
        'drift2': xt.Drift(length=1),
    }
)
half_cell.particle_ref = cell.particle_ref.copy()
tw_half = half_cell.twiss4d(strengths=True, init='periodic_symmetric')

import matplotlib.pyplot as plt
plt.close('all')

twplt1 = tw.plot()
twplt1.ylim(left_lo=0, right_lo=0.5, right_hi=4)
twplt1.left.figure.suptitle('Full cell (periodic twiss)')
twplt2 = tw_half.plot()
twplt2.ylim(left_lo=0, right_lo=0.5, right_hi=4)
twplt2.left.set_xlim(0, 100)
twplt2.left.figure.suptitle('Half cell (periodic-symmetric twiss)')
plt.show()
