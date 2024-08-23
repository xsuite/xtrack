import xtrack as xt
import numpy as np

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

line_half = xt.Line(elements=elements,
                    element_names=['drift0', 'qf1', 'drift1', 'qd1', 'drift2'])
line_half.particle_ref = line.particle_ref.copy()
line_half.build_tracker()

particle_on_co = line_half.build_particles(x=0, px=0, y=0, py=0, delta=0, zeta=0)
RR = line_half.compute_one_turn_matrix_finite_differences(
    particle_on_co=particle_on_co)['R_matrix']


inv_momenta = np.diag([1, -1, 1, -1, 1, -1])

RR_symm = inv_momenta @ np.linalg.inv(RR) @ inv_momenta @ RR

import matplotlib.pyplot as plt
plt.close('all')
tw.plot()

plt.show()