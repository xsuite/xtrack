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

# Compile Particles c-api
source, kernels, cdefs = xt.Particles.XoStruct._gen_c_api()

source = r'''
#include <stdint.h>
#ifndef XOBJ_TYPEDEF_ParticlesData
typedef struct ParticlesData * ParticlesData;
#define XOBJ_TYPEDEF_ParticlesData
#endif

#ifndef XOBJ_TYPEDEF_Float64_N
typedef struct Float64_N * Float64_N;
#define XOBJ_TYPEDEF_Float64_N
#endif
#ifndef XOBJ_TYPEDEF_Float64_N
typedef struct Float64_N * Float64_N;
#define XOBJ_TYPEDEF_Float64_N
#endif

//double ParticlesData_get_x(const ParticlesData obj, int64_t i0){
//  int64_t offset=0;
//  offset+=(int64_t)(((char*) obj)[offset+8]);
//  offset+=16+i0*8;
//  return *(double*)((char*) obj+offset);
double ParticlesData_get_x(const ParticlesData obj, int64_t i0){
  //printf("%ld!\n", *((int64_t*)obj));
  printf("%p!\n", (void*) obj);
  return 0.;
}
'''
kernels = {'ParticlesData_get_x': kernels['ParticlesData_get_x']}


context.add_kernels([source], kernels, extra_cdef=cdefs)


context.kernels.ParticlesData_get_x(obj=particles._xobject, i0=0)
