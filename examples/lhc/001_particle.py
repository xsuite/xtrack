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

from xtrack.particles import reference_vars, per_particle_vars

# Generate local particle CPU

src_lines = []
src_lines.append('''typedef struct{''')
for tt, vv in ((xo.Int64, 'num_particles'),) + reference_vars:
    src_lines.append('    '+tt._c_type+' '+vv+';')
for tt, vv in per_particle_vars:
    src_lines.append('    '+tt._c_type+'* '+vv+';')
src_lines.append('    int64_t ipart;')
src_lines.append('}LocalParticle;')
src_typedef = '\n'.join(src_lines)

src_lines = []
src_lines.append('''
void Particles_to_LocalParticle(ParticlesData source, LocalParticle* dest,
                                int64_t id){''')
for tt, vv in ((xo.Int64, 'num_particles'),) + reference_vars:
    src_lines.append('  ParticlesData_get_'+vv+'(source, dest->'+vv+');')
for tt, vv in per_particle_vars:
    src_lines.append('  ParticlesData_getp1_'+vv+'(source, dest->'+vv+');')
src_lines.append('  dest->ipart = id;')
src_lines.append('}')
src_particles_to_local = '\n'.join(src_lines)


