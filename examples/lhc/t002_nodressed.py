import numpy as np

import xtrack as xt
import xobjects as xo
import sixtracktools
import pysixtrack

context = xo.ContextCpu()

#particles = xt.Particles(_context=context,
#        pysixtrack_particles=[part0_pyst, part1_pyst])
particles = xt.particles.ParticlesData(
                num_particles=2,
                s=np.array([1,2,]),
                x=np.array([7,8,]))

source, kernels, cdefs = xt.Particles.XoStruct._gen_c_api()
context.add_kernels([source], kernels, extra_cdef=cdefs)
print(
        context.kernels.ParticlesData_get_x(obj=particles, i0=0))
