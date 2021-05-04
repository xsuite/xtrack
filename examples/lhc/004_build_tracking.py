from pathlib import Path
import numpy as np
from scipy.special import factorial

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

print('Creating line...')
xtline = xt.Line(_context=context, sequence=pyst_line)

print('Build capi')
sources = []
kernels = {}
cdefs = []

# Particles
source_particles, kernels_particles, cdefs_particles = (
                                xt.Particles.XoStruct._gen_c_api())
sources.append(source_particles)
kernels.update(kernels_particles)
cdefs += cdefs_particles.split('\n')

# Local particles
sources.append(xt.particles.gen_local_particle_api())

# Elements
for cc in xtline._ElementRefClass._rtypes:
    ss, kk, dd = cc._gen_c_api()
    sources.append(ss)
    kernels.update(kk)
    cdefs += dd.split('\n')

sources.append(Path('./drift.h'))
sources.append(Path('./multipole.h'))

cdefs_norep=[]
for cc in cdefs:
    if cc not in cdefs_norep:
        cdefs_norep.append(cc)

# Compile!
context.add_kernels(sources, kernels, extra_cdef='\n\n'.join(cdefs_norep))
