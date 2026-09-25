"""Create and update straight and curved expansions through an Environment.

Run with ``python -m examples.bfieldexpansion.environment``.
Coefficient indices are [transverse derivative order, longitudinal power].
The geometry mode and coefficient shapes remain fixed after construction.
num_phi='auto' covers later changes within those shapes, including expressions.
"""

import xtrack as xt

env = xt.Environment()
env['field'] = 0.1
env['curvature'] = 0.3
for name, h in [('straight', 0.), ('curved', 'curvature')]:
    env.new(name, 'BFieldExpansion', length=0.4, h=h, num_phi='auto',
            knc=[['field', 0., 0.], [0.02, 0., 0.]],
            ksc=[[0., 0., 0.]], ksol=[0.1, 0.02, 0.],
            knl=[0., '0.01*field'], ksl=[0.])

env.set('curved', knc=[['2*field', 0.01, 0.], [0.03, 0., 0.]], nstep=20)
env['field'] = 0.12
env['curvature'] = 0.35
env['straight'].knc[1, 0] = '3*field'
# Hard-edge inputs add knl/length and ksl/length to the field profiles.
env.set('curved', knl=[0., '0.02*field'], ksl=['0.001*field'])
# Tables use the profile integrals plus these additional hard-edge strengths.
print('Curved total normal/skew strengths:', env.get('curved').get_total_knl_ksl())

line = env.new_line(components=['straight', 'curved'])
line.get_table(attr=True).cols['element_type length angle k0l k1l ksoll'].show()
