import numpy as np

import xtrack as xt
import xobjects as xo
import sixtracktools
import pysixtrack

context = xo.ContextCpu()

#particles = xt.Particles(_context=context,
#        pysixtrack_particles=[part0_pyst, part1_pyst])

ParticlesData = xt.particles.ParticlesData

class ParticlesData(xo.Struct):
    num_particles = xo.Int64
    s = xo.Float64[:]
    x = xo.Float64[:]

particles = ParticlesData(
                num_particles=2,
                s=np.array([1,2,]),
                x=np.array([7,8,]))

source, kernels, cdefs = ParticlesData._gen_c_api()
context.add_kernels([source], kernels, extra_cdef=cdefs)
print(
        context.kernels.ParticlesData_get_x(obj=particles, i0=0))
