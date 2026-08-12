// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_LOCAL_PARTICLE_COMMON_H
#define XTRACK_LOCAL_PARTICLE_COMMON_H

// Canonical LocalParticle numeric field groups shared by scalar and TPSA tracking.
#define XT_LOCAL_PARTICLE_COORD_FIELDS(_)    \
    _(x)                                     \
    _(px)                                    \
    _(y)                                     \
    _(py)                                    \
    _(zeta)                                  \
    _(delta)

#define XT_LOCAL_PARTICLE_DERIVED_FIELDS(_)  \
    _(ptau)                                  \
    _(rvv)                                   \
    _(rpp)                                   \
    _(s)

#define XT_LOCAL_PARTICLE_LOCAL_FIELDS(_)    \
    _(ax)                                    \
    _(ay)

#define XT_LOCAL_PARTICLE_TPSA_NUM_FIELDS(_) \
    XT_LOCAL_PARTICLE_COORD_FIELDS(_)        \
    XT_LOCAL_PARTICLE_DERIVED_FIELDS(_)      \
    XT_LOCAL_PARTICLE_LOCAL_FIELDS(_)

// Reference fields stored as scalars for TPSA particles.
#define XT_LOCAL_PARTICLE_REF_FIELDS(_)      \
    _(q0)                                    \
    _(mass0)                                 \
    _(beta0)                                 \
    _(gamma0)                                \
    _(p0c)                                   \
    _(chi)                                   \
    _(charge_ratio)                          \
    _(weight)                                \
    _(anomalous_magnetic_moment)

// Scalar ParticlesData stores these as per-particle numeric arrays. The order
// matches the Python Particles.per_particle_vars ABI.
#define XT_LOCAL_PARTICLE_SCALAR_NUM_FIELDS(_) \
    _(p0c)                                      \
    _(gamma0)                                   \
    _(beta0)                                    \
    _(s)                                        \
    _(zeta)                                     \
    _(x)                                        \
    _(y)                                        \
    _(px)                                       \
    _(py)                                       \
    _(ptau)                                     \
    _(delta)                                    \
    _(rpp)                                      \
    _(rvv)                                      \
    _(chi)                                      \
    _(charge_ratio)                             \
    _(weight)                                   \
    _(ax)                                       \
    _(ay)                                       \
    _(spin_x)                                   \
    _(spin_y)                                   \
    _(spin_z)                                   \
    _(anomalous_magnetic_moment)

// Generic numeric accessor shape. Backend headers define the storage operations.
#define XT_LOCAL_PARTICLE_NUM_ACCESSORS(NAME)                                        \
    XT_LOCAL_PARTICLE_ACCESSOR_PREFIX                                                \
    void LocalParticle_add_to_##NAME(LocalParticle* part, xt_num_arg_t value){       \
        XT_LOCAL_PARTICLE_NUM_ADD(part, NAME, value);                                \
    }                                                                                \
    XT_LOCAL_PARTICLE_ACCESSOR_PREFIX                                                \
    xt_num_t LocalParticle_get_##NAME(LocalParticle* part){                          \
        return XT_LOCAL_PARTICLE_NUM_GET(part, NAME);                                \
    }                                                                                \
    XT_LOCAL_PARTICLE_ACCESSOR_PREFIX                                                \
    void LocalParticle_set_##NAME(LocalParticle* part, xt_num_arg_t value){          \
        XT_LOCAL_PARTICLE_NUM_SET(part, NAME, value);                                \
    }                                                                                \
    XT_LOCAL_PARTICLE_ACCESSOR_PREFIX                                                \
    void LocalParticle_scale_##NAME(LocalParticle* part, xt_num_arg_t value){        \
        XT_LOCAL_PARTICLE_NUM_SCALE(part, NAME, value);                              \
    }

#endif /* XTRACK_LOCAL_PARTICLE_COMMON_H */
