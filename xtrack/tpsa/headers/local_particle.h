// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_TPSA_LOCAL_PARTICLE_H
#define XTRACK_TPSA_LOCAL_PARTICLE_H

#include "xtrack/headers/track.h"
#include "xtrack/particles/headers/local_particle_macros.h"

struct LocalParticle {
    #define INLINE_FIELD(TYPE, NAME) TYPE NAME;
        XT_LP_SIZE_FIELDS(INLINE_FIELD)
        XT_LP_SCALAR_FIELDS(INLINE_FIELD)
    #undef INLINE_FIELD

    #define NUM_FIELD(NAME) tpsa_t* NAME;
        XT_LP_TPSA_NUM_FIELDS(NUM_FIELD)
    #undef NUM_FIELD

    #define POINTER_FIELD(TYPE, NAME) TYPE* NAME;
        XT_LP_REF_MUTABLE_FIELDS(POINTER_FIELD)
        XT_LP_INT_FIELDS(POINTER_FIELD)
        XT_LP_UINT32_FIELDS(POINTER_FIELD)
    #undef POINTER_FIELD

    int64_t ipart;
    int64_t endpart;
    uint64_t track_flags;
    double line_length;
    int8_t* io_buffer;
};

// Convert between the xobject kernel argument and the unrolled LocalParticle used by tracking.
static inline void Particles_to_LocalParticle(
        TpsaParticleData source, LocalParticle* dest, int64_t id, int64_t end_id) {
    #define COPY_NUM_FROM_PARTICLES(NAME) \
        dest->NAME = (tpsa_t*)(uintptr_t)TpsaParticleData_get_ ## NAME(source);
        XT_LP_TPSA_NUM_FIELDS(COPY_NUM_FROM_PARTICLES)
    #undef COPY_NUM_FROM_PARTICLES

    #define COPY_SCALAR_FROM_PARTICLES(TYPE, NAME) \
        dest->NAME = TpsaParticleData_get_ ## NAME(source);
        XT_LP_SCALAR_FIELDS(COPY_SCALAR_FROM_PARTICLES)
    #undef COPY_SCALAR_FROM_PARTICLES

    #define GET_POINTER_FROM_PARTICLES(TYPE, NAME) \
        dest->NAME = TpsaParticleData_getp_ ## NAME(source);
        XT_LP_REF_MUTABLE_FIELDS(GET_POINTER_FROM_PARTICLES)
        XT_LP_INT_FIELDS(GET_POINTER_FROM_PARTICLES)
        XT_LP_UINT32_FIELDS(GET_POINTER_FROM_PARTICLES)
    #undef GET_POINTER_FROM_PARTICLES

    dest->_capacity = 1;
    dest->_num_active_particles = *dest->state > 0;
    dest->_num_lost_particles = *dest->state <= 0;
    dest->start_tracking_at_element = *dest->at_element;
    dest->ipart = id;
    dest->endpart = end_id;
}

static inline void LocalParticle_to_Particles(
        LocalParticle* source, TpsaParticleData dest, int64_t, int64_t) {
    TpsaParticleData_set_track_flags(dest, source->track_flags);
    TpsaParticleData_set_line_length(dest, source->line_length);
}

// Read-only inline metadata getters.
#define SCALAR_GETTER(TYPE, NAME)                                      \
    static inline TYPE LocalParticle_get_ ## NAME(LocalParticle* part) { \
        return part->NAME;                                              \
    }

XT_LP_SIZE_FIELDS(SCALAR_GETTER)
XT_LP_SCALAR_FIELDS(SCALAR_GETTER)
#undef SCALAR_GETTER

// TPSA-valued field accessors. Template overloads preserve expression-template values;
// the concrete overloads provide the same API shape as scalar tracking.
#define TEMPLATE_ACCESSORS(NAME)                                      \
    template<class A>                                                 \
    static inline void LocalParticle_add_to_ ## NAME(                 \
            LocalParticle* part, const mad::tpsa_base<A>& value) {    \
        if (!XT_LP_IS_FROZEN(NAME)) {                                 \
            mad::tpsa_ref(part->NAME) += value;                       \
        }                                                             \
    }                                                                 \
    template<class A>                                                 \
    static inline void LocalParticle_set_ ## NAME(                    \
            LocalParticle* part, const mad::tpsa_base<A>& value) {    \
        if (!XT_LP_IS_FROZEN(NAME)) {                                 \
            mad::tpsa_ref(part->NAME) = value;                        \
        }                                                             \
    }                                                                 \
    template<class A>                                                 \
    static inline void LocalParticle_scale_ ## NAME(                  \
            LocalParticle* part, const mad::tpsa_base<A>& value) {    \
        if (!XT_LP_IS_FROZEN(NAME)) {                                 \
            mad::tpsa_ref(part->NAME) *= value;                       \
        }                                                             \
    }

#define NUM_GET(PART, NAME) (1.0 * mad::tpsa_ref((PART)->NAME))
#define NUM_SET(PART, NAME, VALUE) (mad::tpsa_ref((PART)->NAME) = (VALUE))
#define NUM_ADD(PART, NAME, VALUE) (mad::tpsa_ref((PART)->NAME) += (VALUE))
#define NUM_SCALE(PART, NAME, VALUE) (mad::tpsa_ref((PART)->NAME) *= (VALUE))

XT_LP_TPSA_NUM_FIELDS(TEMPLATE_ACCESSORS)
XT_LP_TPSA_NUM_FIELDS(XT_LP_NUM_ACCESSORS)
#undef NUM_SCALE
#undef NUM_ADD
#undef NUM_SET
#undef NUM_GET
#undef TEMPLATE_ACCESSORS

// Mutable scalar and integer fields use inline storage in TPSA LocalParticle.
#define TYPED_GET(PART, NAME) (*(PART)->NAME)
#define TYPED_SET(PART, NAME, VALUE) (*(PART)->NAME = (VALUE))
#define TYPED_ADD(PART, NAME, VALUE) (*(PART)->NAME += (VALUE))
#define TYPED_SCALE(PART, NAME, VALUE) (*(PART)->NAME *= (VALUE))

XT_LP_REF_MUTABLE_FIELDS(XT_LP_TYPED_ACCESSORS)
XT_LP_INT_FIELDS(XT_LP_TYPED_ACCESSORS)
XT_LP_UINT32_FIELDS(XT_LP_TYPED_ACCESSORS)
#undef TYPED_SCALE
#undef TYPED_ADD
#undef TYPED_SET
#undef TYPED_GET

#endif /* XTRACK_TPSA_LOCAL_PARTICLE_H */
