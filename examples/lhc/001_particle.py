import numpy as np

import xtrack as xt
import xobjects as xo
import sixtracktools
import pysixtrack

context = xo.ContextCpu()

six = sixtracktools.SixInput(".")
pyst_line = pysixtrack.Line.from_sixinput(six)
sixdump = sixtracktools.SixDump101("res/dump3.dat")

# TODO: The two particles look identical, to be checked
part0_pyst = pysixtrack.Particles(**sixdump[0::2][0].get_minimal_beam())
part1_pyst = pysixtrack.Particles(**sixdump[1::2][0].get_minimal_beam())

particles = xt.Particles(pysixtrack_particles=[part0_pyst, part1_pyst])

source, kernels, cdefs = xt.Particles.XoStruct._gen_c_api()

from xtrack.particles import scalar_vars, per_particle_vars

context.add_kernels([source], kernels, extra_cdef=cdefs)


# Checks
from xtrack.particles import scalar_vars, per_particle_vars
assert particles.num_particles == 2
for tt, vv in per_particle_vars:
    print(f'Check {vv}')
    for ii in range(particles.num_particles):
        pyval = getattr(particles, vv)[ii]
        cval = getattr(context.kernels, f'ParticlesData_get_{vv}')(
                obj=particles, i0=ii)
        assert pyval == cval
for tt, vv in scalar_vars:
    print(f'Check {vv}')
    pyval = getattr(particles, vv)
    cval = getattr(context.kernels, f'ParticlesData_get_{vv}')(
            obj=particles)
    assert pyval == cval
