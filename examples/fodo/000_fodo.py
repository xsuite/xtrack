import numpy as np

import xobjects as xo
import xtrack as xt

## Generate a simple line
line = xt.Line(
    elements=[xt.Drift(length=1.),
              xt.Multipole(knl=[0, 1.], ksl=[0,0]),
              xt.Drift(length=1.),
              xt.Multipole(knl=[0, -1.], ksl=[0,0])],
    element_names=['drift_0', 'quad_0', 'drift_1', 'quad_1'])

## Attach a reference particle to the line (optional)
## (defines the reference mass, charge and energy)
line.particle_ref = xt.Particles(p0c=6500e9, #eV
                                 q0=1, mass0=xt.PROTON_MASS_EV)

line.twiss_default['method'] = '4d'

tw = line.twiss()

# Initialize deferred expressions
line._init_var_management()

# attach knobs to the quads
line.vars['knl_f'] = 1.
line.vars['knl_d'] = -1.

line.element_refs['quad_0'].knl[1] = line.vars['knl_f']
line.element_refs['quad_1'].knl[1] = line.vars['knl_d']

# Match tunes
line.match(
    vary=xt.VaryList(['knl_f', 'knl_d'], step=1e-3),
    targets=[xt.TargetSet(qx=.18, qy=.16)])

tw_after = line.twiss()
print ('Tunes after matching:')
print(f'qx = {tw_after.qx:.3f}, qy = {tw_after.qy:.3f}')