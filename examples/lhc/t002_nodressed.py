import numpy as np

import xtrack as xt
import xobjects as xo
import sixtracktools
import pysixtrack

context = xo.ContextCpu()

ParticlesData = xt.particles.ParticlesData
particles = ParticlesData(
                s=[1,2,3],
                x=[7,8,9])

source, kernels, cdefs = ParticlesData._gen_c_api()
context.add_kernels([source], kernels, extra_cdef=cdefs)
print(
        context.kernels.ParticlesData_get_x(obj=particles, i0=0))
