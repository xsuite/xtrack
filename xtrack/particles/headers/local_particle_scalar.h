// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_PARTICLES_LOCAL_PARTICLE_SCALAR_H
#define XTRACK_PARTICLES_LOCAL_PARTICLE_SCALAR_H

#include "xobjects/headers/common.h"
#include "xtrack/particles/rng_src/base_rng.h"
#include "xtrack/particles/rng_src/particles_rng.h"
#include "xtrack/headers/track.h"
#include "xtrack/particles/headers/local_particle_macros.h"

// LocalParticle stores scalar particle-level values inline and per-particle
// arrays as pointers into ParticlesData.
typedef struct {
    #define SCALAR_STRUCT_FIELD(TYPE, NAME) TYPE NAME;
        XT_LP_SIZE_FIELDS(SCALAR_STRUCT_FIELD)
        XT_LP_SCALAR_FIELDS(SCALAR_STRUCT_FIELD)
    #undef SCALAR_STRUCT_FIELD

    #define NUM_STRUCT_FIELD(NAME) GPUGLMEM xt_num_t* NAME;
        XT_LP_SCALAR_NUM_FIELDS(NUM_STRUCT_FIELD)
    #undef NUM_STRUCT_FIELD

    #define POINTER_STRUCT_FIELD(TYPE, NAME) GPUGLMEM TYPE* NAME;
        XT_LP_INT_FIELDS(POINTER_STRUCT_FIELD)
        XT_LP_UINT32_FIELDS(POINTER_STRUCT_FIELD)
    #undef POINTER_STRUCT_FIELD

    int64_t ipart;
    int64_t endpart;
    uint64_t track_flags;
    double line_length;
    GPUGLMEM int8_t* io_buffer;
} LocalParticle;

#define SCALAR_GETTER(TYPE, NAME)                       \
    GPUFUN                                              \
    TYPE LocalParticle_get_ ## NAME(LocalParticle* part){ \
        return part->NAME;                              \
    }

XT_LP_SIZE_FIELDS(SCALAR_GETTER)
XT_LP_SCALAR_FIELDS(SCALAR_GETTER)
#undef SCALAR_GETTER

// Per-particle get/set/add/scale accessors.
#define ACCESSOR_PREFIX GPUFUN
#define NUM_GET(PART, NAME) ((PART)->NAME[(PART)->ipart])
#define NUM_SET(PART, NAME, VALUE) ((PART)->NAME[(PART)->ipart] = (VALUE))
#define NUM_ADD(PART, NAME, VALUE) ((PART)->NAME[(PART)->ipart] += (VALUE))
#define NUM_SCALE(PART, NAME, VALUE) ((PART)->NAME[(PART)->ipart] *= (VALUE))
#define TYPED_GET(PART, NAME) ((PART)->NAME[(PART)->ipart])
#define TYPED_SET(PART, NAME, VALUE) ((PART)->NAME[(PART)->ipart] = (VALUE))
#define TYPED_ADD(PART, NAME, VALUE) ((PART)->NAME[(PART)->ipart] += (VALUE))
#define TYPED_SCALE(PART, NAME, VALUE) ((PART)->NAME[(PART)->ipart] *= (VALUE))

XT_LP_SCALAR_NUM_FIELDS(XT_LP_NUM_ACCESSORS)
XT_LP_INT_FIELDS(XT_LP_TYPED_ACCESSORS)
XT_LP_UINT32_FIELDS(XT_LP_TYPED_ACCESSORS)
#undef TYPED_SCALE
#undef TYPED_ADD
#undef TYPED_SET
#undef TYPED_GET
#undef NUM_SCALE
#undef NUM_ADD
#undef NUM_SET
#undef NUM_GET
#undef ACCESSOR_PREFIX

#ifndef XTRACK_TPSA_TRACK
/*
 * These conversion helpers are only supported by scalar tracking. TPSA tracking
 * uses TpsaParticleData and provides its own conversion helpers in the TPSA
 * LocalParticle implementation.
 */

// Conversion between the global ParticlesData structure and a LocalParticle view.
GPUFUN
void Particles_to_LocalParticle(
    ParticlesData source,
    LocalParticle* dest,
    int64_t id,
    int64_t eid
) {
    #define COPY_SCALAR_FROM_PARTICLES(TYPE, NAME) \
            dest->NAME = ParticlesData_get_ ## NAME(source);
        XT_LP_SIZE_FIELDS(COPY_SCALAR_FROM_PARTICLES)
        XT_LP_SCALAR_FIELDS(COPY_SCALAR_FROM_PARTICLES)
    #undef COPY_SCALAR_FROM_PARTICLES

    #define GET_NUM_POINTER_FROM_PARTICLES(NAME) \
            dest->NAME = ParticlesData_getp1_ ## NAME(source, 0);
        XT_LP_SCALAR_NUM_FIELDS(GET_NUM_POINTER_FROM_PARTICLES)
    #undef GET_NUM_POINTER_FROM_PARTICLES

    #define GET_POINTER_FROM_PARTICLES(TYPE, NAME) \
            dest->NAME = ParticlesData_getp1_ ## NAME(source, 0);
        XT_LP_INT_FIELDS(GET_POINTER_FROM_PARTICLES)
        XT_LP_UINT32_FIELDS(GET_POINTER_FROM_PARTICLES)
    #undef GET_POINTER_FROM_PARTICLES

    dest->ipart = id;
    dest->endpart = eid;
}

GPUFUN
void LocalParticle_to_Particles(
    LocalParticle* source,
    ParticlesData dest,
    int64_t id,
    int64_t set_scalar
) {
    if (set_scalar) {
        #define COPY_SCALAR_TO_PARTICLES(TYPE, NAME) \
                ParticlesData_set_ ## NAME(dest, LocalParticle_get_ ## NAME(source));
                XT_LP_SIZE_FIELDS(COPY_SCALAR_TO_PARTICLES)
                XT_LP_SCALAR_FIELDS(COPY_SCALAR_TO_PARTICLES)
        #undef COPY_SCALAR_TO_PARTICLES
    }

    #define COPY_NUM_FIELD_TO_PARTICLES(NAME) \
        ParticlesData_set_ ## NAME(dest, id, LocalParticle_get_ ## NAME(source));
        XT_LP_SCALAR_NUM_FIELDS(COPY_NUM_FIELD_TO_PARTICLES)
    #undef COPY_NUM_FIELD_TO_PARTICLES

    #define COPY_FIELD_TO_PARTICLES(TYPE, NAME) \
        ParticlesData_set_ ## NAME(dest, id, LocalParticle_get_ ## NAME(source));
        XT_LP_INT_FIELDS(COPY_FIELD_TO_PARTICLES)
        XT_LP_UINT32_FIELDS(COPY_FIELD_TO_PARTICLES)
    #undef COPY_FIELD_TO_PARTICLES
}

#endif /* XTRACK_TPSA_TRACK */

#endif /* XTRACK_PARTICLES_LOCAL_PARTICLE_SCALAR_H */
