"""Slice a polynomial field without copying or refitting its coefficients.

Run with ``python -m examples.bfieldexpansion.slicing``.
All thick slices share the parent expansion and follow subsequent updates.
"""

import numpy as np
import xtrack as xt

env = xt.Environment()
env.new('field', 'BFieldExpansion', length=0.8, h=0.3, sstart=0.15,
        knc=[[0.1, 0.08, -0.03], [0.04, -0.02, 0.01]],
        ksc=[[0.02, -0.01, 0.02]], ksol=[0.15, 0.03, 0.], ny=5, nstep=160)
line = env.new_line(components=['field'])
initial = xt.Particles(p0c=1e9, x=0.01, y=0.007)
reference = initial.copy()
env.get('field').track(reference)

line.slice_thick_elements([
    xt.Strategy(xt.Uniform(4, mode='thick'), element_type=xt.BFieldExpansion)])
particles = initial.copy()
line.track(particles, _force_no_end_turn_actions=True)
for coordinate in ('x', 'px', 'y', 'py', 'zeta', 'delta', 's'):
    np.testing.assert_allclose(getattr(particles, coordinate),
                               getattr(reference, coordinate), rtol=0, atol=1e-13)

for name in line.element_names:
    if isinstance(line[name], xt.ThickSliceBFieldExpansion):
        print(f'{name}: sstart={line[name].sstart:g}, nstep={line[name].nstep}')

env.set('field', length=1.2, knc=[[0.12, 0.1, -0.03], [0.04, -0.02, 0.01]])
line.get_table(attr=True).cols['s length angle k0l k1l ksoll'].show()
