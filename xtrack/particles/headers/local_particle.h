// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_PARTICLES_LOCAL_PARTICLE_H
#define XTRACK_PARTICLES_LOCAL_PARTICLE_H

#ifdef XTRACK_TPSA_TRACK
    #include "xtrack/tpsa/headers/local_particle.h"
#else
    #include "xtrack/particles/headers/local_particle_scalar.h"
#endif

// Hand-written helpers shared by the scalar and TPSA LocalParticle implementations.
#include "xtrack/particles/headers/local_particle_common.h"

#endif /* XTRACK_PARTICLES_LOCAL_PARTICLE_H */
