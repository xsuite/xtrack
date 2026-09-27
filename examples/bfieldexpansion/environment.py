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
env['magnet_scale'] = 1.
for name, h in [('straight', 0.), ('curved', 'curvature')]:
    # Omitted ksc and ksl are zero-filled to the inferred coefficient shapes.
    env.new(name, 'BFieldExpansion', length=0.4, h=h, num_phi='auto',
            kscale='magnet_scale', k1='0.1*field',
            integrator='rk4', num_integration_steps=10,
            knc=[['field', 0., 0.], [0.02, 0., 0.]],
            ksol=[0.1, 0.02, 0.], knl=[0., '0.01*field'])

env.set('curved', knc=[['2*field', 0.01, 0.], [0.03, 0., 0.]],
        num_integration_steps=20)
env['field'] = 0.12
env['curvature'] = 0.35
env['straight'].knc[1, 0] = '3*field'
# Scalar strengths are additional uniform fields, independent of curvature.
env.set('curved', k0='0.2*field', k2s='0.03*field')
print('Polynomial order (fixed by knc/ksc shapes):', env.get('curved').order)
# Use the same entrance/exit misalignment maps as other Xtrack magnets.
env['x_offset'] = 1e-3
env.set('curved', shift_x='x_offset', rot_s_rad=0.02)
# Hard-edge inputs add knl/length and ksl/length to the field profiles.
env.set('curved', knl=[0., '0.02*field'], ksl=['0.001*field', 0.])
# Scale all field components, including the hard-edge inputs and solenoid.
env['magnet_scale'] = 0.8
# Tables use the profile integrals plus these additional hard-edge strengths.
print('Curved total normal/skew strengths:', env.get('curved').get_total_knl_ksl())

line = env.new_line(components=['straight', 'curved'])
line.get_table(attr=True).cols['element_type length angle k0l k1l ksoll'].show()
