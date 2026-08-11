// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_PARTICLES_LOCAL_PARTICLE_H
#define XTRACK_PARTICLES_LOCAL_PARTICLE_H

#include "xobjects/headers/common.h"
#include "xtrack/particles/rng_src/base_rng.h"
#include "xtrack/particles/rng_src/particles_rng.h"
#include "xtrack/headers/track.h"

// Field lists used to define the scalar LocalParticle ABI and its accessors.
#define XT_LOCAL_PARTICLE_SIZE_FIELDS(_)         \
    _(int64_t, _capacity)                        \
    _(int64_t, _num_active_particles)            \
    _(int64_t, _num_lost_particles)              \
    _(int64_t, start_tracking_at_element)

#define XT_LOCAL_PARTICLE_SCALAR_FIELDS(_)       \
    _(double, q0)                                \
    _(double, mass0)                             \
    _(double, t_sim)

#define XT_LOCAL_PARTICLE_FLOAT_FIELDS(_)        \
    _(double, p0c)                               \
    _(double, gamma0)                            \
    _(double, beta0)                             \
    _(double, s)                                 \
    _(double, zeta)                              \
    _(double, x)                                 \
    _(double, y)                                 \
    _(double, px)                                \
    _(double, py)                                \
    _(double, ptau)                              \
    _(double, delta)                             \
    _(double, rpp)                               \
    _(double, rvv)                               \
    _(double, chi)                               \
    _(double, charge_ratio)                      \
    _(double, weight)                            \
    _(double, ax)                                \
    _(double, ay)                                \
    _(double, spin_x)                            \
    _(double, spin_y)                            \
    _(double, spin_z)                            \
    _(double, anomalous_magnetic_moment)

#define XT_LOCAL_PARTICLE_INT_FIELDS(_)          \
    _(int64_t, pdg_id)                           \
    _(int64_t, particle_id)                      \
    _(int64_t, at_element)                       \
    _(int64_t, at_turn)                          \
    _(int64_t, state)                            \
    _(int64_t, parent_particle_id)

#define XT_LOCAL_PARTICLE_UINT32_FIELDS(_)       \
    _(uint32_t, _rng_s1)                         \
    _(uint32_t, _rng_s2)                         \
    _(uint32_t, _rng_s3)                         \
    _(uint32_t, _rng_s4)

#define XT_LOCAL_PARTICLE_FIELDS(_)              \
    XT_LOCAL_PARTICLE_FLOAT_FIELDS(_)            \
    XT_LOCAL_PARTICLE_INT_FIELDS(_)              \
    XT_LOCAL_PARTICLE_UINT32_FIELDS(_)

// LocalParticle stores scalar particle-level values inline and per-particle
// arrays as pointers into ParticlesData.
typedef struct {
    #define XT_LOCAL_PARTICLE_SCALAR_STRUCT_FIELD(TYPE, NAME) TYPE NAME;
        XT_LOCAL_PARTICLE_SIZE_FIELDS(XT_LOCAL_PARTICLE_SCALAR_STRUCT_FIELD)
        XT_LOCAL_PARTICLE_SCALAR_FIELDS(XT_LOCAL_PARTICLE_SCALAR_STRUCT_FIELD)
    #undef XT_LOCAL_PARTICLE_SCALAR_STRUCT_FIELD

    #define XT_LOCAL_PARTICLE_POINTER_STRUCT_FIELD(TYPE, NAME) GPUGLMEM TYPE* NAME;
        XT_LOCAL_PARTICLE_FIELDS(XT_LOCAL_PARTICLE_POINTER_STRUCT_FIELD)
    #undef XT_LOCAL_PARTICLE_POINTER_STRUCT_FIELD

    int64_t ipart;
    int64_t endpart;
    uint64_t track_flags;
    double line_length;
    GPUGLMEM int8_t* io_buffer;
} LocalParticle;

// Small scalar helpers that are not tied to a per-particle field.
GPUFUN
GPUGLMEM int8_t* LocalParticle_get_io_buffer(LocalParticle* part){
    return part->io_buffer;
}

GPUFUN
uint64_t LocalParticle_check_track_flag(LocalParticle* part, uint8_t index){
    return (part->track_flags >> index) & 1;
}

#define XT_LOCAL_PARTICLE_SCALAR_GETTER(TYPE, NAME)                 \
    GPUFUN                                                          \
    TYPE LocalParticle_get_##NAME(LocalParticle* part){             \
        return part->NAME;                                          \
    }

XT_LOCAL_PARTICLE_SIZE_FIELDS(XT_LOCAL_PARTICLE_SCALAR_GETTER)
XT_LOCAL_PARTICLE_SCALAR_FIELDS(XT_LOCAL_PARTICLE_SCALAR_GETTER)
#undef XT_LOCAL_PARTICLE_SCALAR_GETTER

// FREEZE_VAR_* is a presence-style compatibility macro. XT_FREEZE_VAR_* is the
// value-style form used by the generic accessors below.
#ifndef XT_FREEZE_VAR_p0c
#define XT_FREEZE_VAR_p0c 0
#endif
#ifndef XT_FREEZE_VAR_gamma0
#define XT_FREEZE_VAR_gamma0 0
#endif
#ifndef XT_FREEZE_VAR_beta0
#define XT_FREEZE_VAR_beta0 0
#endif
#ifndef XT_FREEZE_VAR_s
#define XT_FREEZE_VAR_s 0
#endif
#ifndef XT_FREEZE_VAR_zeta
#define XT_FREEZE_VAR_zeta 0
#endif
#ifndef XT_FREEZE_VAR_x
#define XT_FREEZE_VAR_x 0
#endif
#ifndef XT_FREEZE_VAR_y
#define XT_FREEZE_VAR_y 0
#endif
#ifndef XT_FREEZE_VAR_px
#define XT_FREEZE_VAR_px 0
#endif
#ifndef XT_FREEZE_VAR_py
#define XT_FREEZE_VAR_py 0
#endif
#ifndef XT_FREEZE_VAR_ptau
#define XT_FREEZE_VAR_ptau 0
#endif
#ifndef XT_FREEZE_VAR_delta
#define XT_FREEZE_VAR_delta 0
#endif
#ifndef XT_FREEZE_VAR_rpp
#define XT_FREEZE_VAR_rpp 0
#endif
#ifndef XT_FREEZE_VAR_rvv
#define XT_FREEZE_VAR_rvv 0
#endif
#ifndef XT_FREEZE_VAR_chi
#define XT_FREEZE_VAR_chi 0
#endif
#ifndef XT_FREEZE_VAR_charge_ratio
#define XT_FREEZE_VAR_charge_ratio 0
#endif
#ifndef XT_FREEZE_VAR_weight
#define XT_FREEZE_VAR_weight 0
#endif
#ifndef XT_FREEZE_VAR_ax
#define XT_FREEZE_VAR_ax 0
#endif
#ifndef XT_FREEZE_VAR_ay
#define XT_FREEZE_VAR_ay 0
#endif
#ifndef XT_FREEZE_VAR_spin_x
#define XT_FREEZE_VAR_spin_x 0
#endif
#ifndef XT_FREEZE_VAR_spin_y
#define XT_FREEZE_VAR_spin_y 0
#endif
#ifndef XT_FREEZE_VAR_spin_z
#define XT_FREEZE_VAR_spin_z 0
#endif
#ifndef XT_FREEZE_VAR_anomalous_magnetic_moment
#define XT_FREEZE_VAR_anomalous_magnetic_moment 0
#endif

#define XT_LOCAL_PARTICLE_IS_FROZEN(NAME) \
    XT_LOCAL_PARTICLE_IS_FROZEN_IMPL(NAME)
#define XT_LOCAL_PARTICLE_IS_FROZEN_IMPL(NAME) \
    XT_FREEZE_VAR_##NAME

// Per-particle get/set/add/scale accessors.
#define XT_LOCAL_PARTICLE_FLOAT_ACCESSORS(TYPE, NAME)                  \
    GPUFUN                                                             \
    void LocalParticle_add_to_##NAME(LocalParticle* part, TYPE value){ \
        if (!XT_LOCAL_PARTICLE_IS_FROZEN(NAME)) {                      \
            part->NAME[part->ipart] += value;                          \
        }                                                              \
    }                                                                  \
    GPUFUN                                                             \
    TYPE LocalParticle_get_##NAME(LocalParticle* part){                \
        return part->NAME[part->ipart];                                \
    }                                                                  \
    GPUFUN                                                             \
    void LocalParticle_set_##NAME(LocalParticle* part, TYPE value){    \
        if (!XT_LOCAL_PARTICLE_IS_FROZEN(NAME)) {                      \
            part->NAME[part->ipart] = value;                           \
        }                                                              \
    }                                                                  \
    GPUFUN                                                             \
    void LocalParticle_scale_##NAME(LocalParticle* part, TYPE value){  \
        if (!XT_LOCAL_PARTICLE_IS_FROZEN(NAME)) {                      \
            part->NAME[part->ipart] *= value;                          \
        }                                                              \
    }

#define XT_LOCAL_PARTICLE_INDEXED_ACCESSORS(TYPE, NAME)                \
    GPUFUN                                                             \
    void LocalParticle_add_to_##NAME(LocalParticle* part, TYPE value){ \
        part->NAME[part->ipart] += value;                              \
    }                                                                  \
    GPUFUN                                                             \
    TYPE LocalParticle_get_##NAME(LocalParticle* part){                \
        return part->NAME[part->ipart];                                \
    }                                                                  \
    GPUFUN                                                             \
    void LocalParticle_set_##NAME(LocalParticle* part, TYPE value){    \
        part->NAME[part->ipart] = value;                               \
    }                                                                  \
    GPUFUN                                                             \
    void LocalParticle_scale_##NAME(LocalParticle* part, TYPE value){  \
        part->NAME[part->ipart] *= value;                              \
    }

XT_LOCAL_PARTICLE_FLOAT_FIELDS(XT_LOCAL_PARTICLE_FLOAT_ACCESSORS)
XT_LOCAL_PARTICLE_INT_FIELDS(XT_LOCAL_PARTICLE_INDEXED_ACCESSORS)
XT_LOCAL_PARTICLE_UINT32_FIELDS(XT_LOCAL_PARTICLE_INDEXED_ACCESSORS)
#undef XT_LOCAL_PARTICLE_FLOAT_ACCESSORS
#undef XT_LOCAL_PARTICLE_INDEXED_ACCESSORS

// Conversion between the global ParticlesData structure and a LocalParticle view.
GPUFUN
void Particles_to_LocalParticle(
    ParticlesData source,
    LocalParticle* dest,
    int64_t id,
    int64_t eid
) {
    #define XT_LOCAL_PARTICLE_COPY_SCALAR_FROM_PARTICLES(TYPE, NAME) \
        dest->NAME = ParticlesData_get_##NAME(source);
        XT_LOCAL_PARTICLE_SIZE_FIELDS(XT_LOCAL_PARTICLE_COPY_SCALAR_FROM_PARTICLES)
        XT_LOCAL_PARTICLE_SCALAR_FIELDS(XT_LOCAL_PARTICLE_COPY_SCALAR_FROM_PARTICLES)
    #undef XT_LOCAL_PARTICLE_COPY_SCALAR_FROM_PARTICLES

    #define XT_LOCAL_PARTICLE_GET_POINTER_FROM_PARTICLES(TYPE, NAME) \
        dest->NAME = ParticlesData_getp1_##NAME(source, 0);
        XT_LOCAL_PARTICLE_FIELDS(XT_LOCAL_PARTICLE_GET_POINTER_FROM_PARTICLES)
    #undef XT_LOCAL_PARTICLE_GET_POINTER_FROM_PARTICLES

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
        #define XT_LOCAL_PARTICLE_COPY_SCALAR_TO_PARTICLES(TYPE, NAME) \
                ParticlesData_set_##NAME(dest, LocalParticle_get_##NAME(source));
                XT_LOCAL_PARTICLE_SIZE_FIELDS(XT_LOCAL_PARTICLE_COPY_SCALAR_TO_PARTICLES)
                XT_LOCAL_PARTICLE_SCALAR_FIELDS(XT_LOCAL_PARTICLE_COPY_SCALAR_TO_PARTICLES)
        #undef XT_LOCAL_PARTICLE_COPY_SCALAR_TO_PARTICLES
    }

    #define XT_LOCAL_PARTICLE_COPY_FIELD_TO_PARTICLES(TYPE, NAME) \
        ParticlesData_set_##NAME(dest, id, LocalParticle_get_##NAME(source));
        XT_LOCAL_PARTICLE_FIELDS(XT_LOCAL_PARTICLE_COPY_FIELD_TO_PARTICLES)
    #undef XT_LOCAL_PARTICLE_COPY_FIELD_TO_PARTICLES
}

// Swap two particles in all per-particle arrays. Used by active-particle
// reorganization.
GPUFUN
void LocalParticle_exchange(LocalParticle* part, int64_t i1, int64_t i2){
    #define XT_LOCAL_PARTICLE_EXCHANGE_FIELD(TYPE, NAME) \
        {                                                \
            TYPE temp = part->NAME[i2];                  \
            part->NAME[i2] = part->NAME[i1];             \
            part->NAME[i1] = temp;                       \
        }
        XT_LOCAL_PARTICLE_FIELDS(XT_LOCAL_PARTICLE_EXCHANGE_FIELD)
    #undef XT_LOCAL_PARTICLE_EXCHANGE_FIELD
}

// Angle convenience API. These are derived accessors over px/py/rpp/delta, but
// still belong to the mechanically generated LocalParticle field API.
#define XT_LOCAL_PARTICLE_ANGLE_API(EXACT, XX, YY)                               \
    GPUFUN                                                                       \
    double LocalParticle_get_##EXACT##XX##p(LocalParticle* part){                \
        double const p##XX = LocalParticle_get_p##XX(part);                      \
        double const rpp = LocalParticle_get_rpp(part);                          \
        return p##XX * rpp;                                                      \
    }                                                                            \
    GPUFUN                                                                       \
    void LocalParticle_set_##EXACT##XX##p(LocalParticle* part, double XX##p){    \
        double rpp = LocalParticle_get_rpp(part);                                \
        LocalParticle_set_p##XX(part, XX##p / rpp);                              \
    }                                                                            \
    GPUFUN                                                                       \
    void LocalParticle_add_to_##EXACT##XX##p(LocalParticle* part, double XX##p){ \
        LocalParticle_set_##EXACT##XX##p(                                        \
            part, LocalParticle_get_##EXACT##XX##p(part) + XX##p);               \
    }                                                                            \
    GPUFUN                                                                       \
    void LocalParticle_scale_##EXACT##XX##p(LocalParticle* part, double value){  \
        LocalParticle_set_##EXACT##XX##p(                                        \
            part, LocalParticle_get_##EXACT##XX##p(part) * value);               \
    }

#define XT_LOCAL_PARTICLE_EXACT_ANGLE_API(EXACT, XX, YY)                         \
    GPUFUN                                                                       \
    double LocalParticle_get_##EXACT##XX##p(LocalParticle* part){                \
        double const p##XX = LocalParticle_get_p##XX(part);                      \
        double const p##YY = LocalParticle_get_p##YY(part);                      \
        double const one_plus_delta = 1. + LocalParticle_get_delta(part);        \
        double const rpp = 1. / sqrt(                                            \
            one_plus_delta * one_plus_delta - px * px - py * py);                \
        return p##XX * rpp;                                                      \
    }                                                                            \
    GPUFUN                                                                       \
    void LocalParticle_set_##EXACT##XX##p(LocalParticle* part, double XX##p){    \
        double rpp = LocalParticle_get_rpp(part);                                \
        double const YY##p = LocalParticle_get_##EXACT##YY##p(part);             \
        rpp *= sqrt(1 + xp * xp + yp * yp);                                      \
        LocalParticle_set_p##XX(part, XX##p / rpp);                              \
    }                                                                            \
    GPUFUN                                                                       \
    void LocalParticle_add_to_##EXACT##XX##p(LocalParticle* part, double XX##p){ \
        LocalParticle_set_##EXACT##XX##p(                                        \
            part, LocalParticle_get_##EXACT##XX##p(part) + XX##p);               \
    }                                                                            \
    GPUFUN                                                                       \
    void LocalParticle_scale_##EXACT##XX##p(LocalParticle* part, double value){  \
        LocalParticle_set_##EXACT##XX##p(                                        \
            part, LocalParticle_get_##EXACT##XX##p(part) * value);               \
    }

XT_LOCAL_PARTICLE_ANGLE_API(, x, y)
XT_LOCAL_PARTICLE_ANGLE_API(, y, x)

GPUFUN double LocalParticle_get_exact_xp(LocalParticle* part);
GPUFUN double LocalParticle_get_exact_yp(LocalParticle* part);

XT_LOCAL_PARTICLE_EXACT_ANGLE_API(exact_, x, y)
XT_LOCAL_PARTICLE_EXACT_ANGLE_API(exact_, y, x)
#undef XT_LOCAL_PARTICLE_EXACT_ANGLE_API
#undef XT_LOCAL_PARTICLE_ANGLE_API

GPUFUN
void LocalParticle_set_xp_yp(LocalParticle* part, double xp, double yp) {
    double rpp = LocalParticle_get_rpp(part);
    LocalParticle_set_px(part, xp / rpp);
    LocalParticle_set_py(part, yp / rpp);
}

GPUFUN
void LocalParticle_add_to_xp_yp(LocalParticle* part, double xp, double yp) {
    LocalParticle_set_xp_yp(
        part,
        LocalParticle_get_xp(part) + xp,
        LocalParticle_get_yp(part) + yp
    );
}

GPUFUN
void LocalParticle_scale_xp_yp(LocalParticle* part, double value_x, double value_y) {
    LocalParticle_set_xp_yp(
        part,
        LocalParticle_get_xp(part) * value_x,
        LocalParticle_get_yp(part) * value_y
    );
}

GPUFUN
void LocalParticle_set_exact_xp_yp(LocalParticle* part, double xp, double yp) {
    double rpp = LocalParticle_get_rpp(part);
    rpp *= sqrt(1 + xp * xp + yp * yp);
    LocalParticle_set_px(part, xp / rpp);
    LocalParticle_set_py(part, yp / rpp);
}

GPUFUN
void LocalParticle_add_to_exact_xp_yp(LocalParticle* part, double xp, double yp) {
    LocalParticle_set_exact_xp_yp(
        part,
        LocalParticle_get_exact_xp(part) + xp,
        LocalParticle_get_exact_yp(part) + yp
    );
}

GPUFUN
void LocalParticle_scale_exact_xp_yp(LocalParticle* part, double value_x, double value_y) {
    LocalParticle_set_exact_xp_yp(
        part,
        LocalParticle_get_exact_xp(part) * value_x,
        LocalParticle_get_exact_yp(part) * value_y
    );
}

// Hand-written helpers layered on top of the generated-style field API.
#include "xtrack/particles/headers/local_particle_custom.h"

#undef XT_LOCAL_PARTICLE_FIELDS
#undef XT_LOCAL_PARTICLE_UINT32_FIELDS
#undef XT_LOCAL_PARTICLE_INT_FIELDS
#undef XT_LOCAL_PARTICLE_FLOAT_FIELDS
#undef XT_LOCAL_PARTICLE_SCALAR_FIELDS
#undef XT_LOCAL_PARTICLE_SIZE_FIELDS

#endif /* XTRACK_PARTICLES_LOCAL_PARTICLE_H */
