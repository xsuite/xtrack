// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_PARTICLES_LOCAL_PARTICLE_MACROS_H
#define XTRACK_PARTICLES_LOCAL_PARTICLE_MACROS_H

// Inline particle metadata. These fields are read-only through the LocalParticle API.
#define XT_LP_SIZE_FIELDS(_)             \
    _(int64_t, _capacity)                \
    _(int64_t, _num_active_particles)    \
    _(int64_t, _num_lost_particles)      \
    _(int64_t, start_tracking_at_element)

#define XT_LP_SCALAR_FIELDS(_)   \
    _(double, q0)                \
    _(double, mass0)             \
    _(double, t_sim)

// Fields represented by TPSAs during TPSA tracking.
#define XT_LP_COORD_FIELDS(_) \
    _(x)                       \
    _(px)                      \
    _(y)                       \
    _(py)                      \
    _(zeta)                    \
    _(delta)

#define XT_LP_DERIVED_FIELDS(_) \
    _(ptau)                     \
    _(rvv)                      \
    _(rpp)                      \
    _(s)

#define XT_LP_LOCAL_FIELDS(_) \
    _(ax)                       \
    _(ay)

#define XT_LP_SPIN_FIELDS(_) \
    _(spin_x)                   \
    _(spin_y)                   \
    _(spin_z)

#define XT_LP_TPSA_NUM_FIELDS(_) \
    XT_LP_COORD_FIELDS(_)        \
    XT_LP_DERIVED_FIELDS(_)      \
    XT_LP_LOCAL_FIELDS(_)        \
    XT_LP_SPIN_FIELDS(_)

// Mutable double fields remain scalar during TPSA tracking.
#define XT_LP_REF_MUTABLE_FIELDS(_)       \
    _(double, p0c)                        \
    _(double, gamma0)                     \
    _(double, beta0)                      \
    _(double, chi)                        \
    _(double, charge_ratio)               \
    _(double, weight)                     \
    _(double, anomalous_magnetic_moment)

#define XT_LP_REF_FIELDS(_) \
    _(q0)                  \
    _(mass0)               \
    _(t_sim)               \
    _(p0c)                 \
    _(gamma0)              \
    _(beta0)               \
    _(chi)                 \
    _(charge_ratio)        \
    _(weight)              \
    _(anomalous_magnetic_moment)

#define XT_LP_INT_FIELDS(_)          \
    _(int64_t, pdg_id)               \
    _(int64_t, particle_id)          \
    _(int64_t, at_element)           \
    _(int64_t, at_turn)              \
    _(int64_t, state)                \
    _(int64_t, parent_particle_id)

#define XT_LP_UINT32_FIELDS(_) \
    _(uint32_t, _rng_s1)         \
    _(uint32_t, _rng_s2)         \
    _(uint32_t, _rng_s3)         \
    _(uint32_t, _rng_s4)

// Scalar ParticlesData stores these as per-particle numeric arrays. Keep this order in
// sync with Particles.per_particle_vars.
#define XT_LP_SCALAR_NUM_FIELDS(_) \
    _(p0c)                         \
    _(gamma0)                      \
    _(beta0)                       \
    _(s)                           \
    _(zeta)                        \
    _(x)                           \
    _(y)                           \
    _(px)                          \
    _(py)                          \
    _(ptau)                        \
    _(delta)                       \
    _(rpp)                         \
    _(rvv)                         \
    _(chi)                         \
    _(charge_ratio)                \
    _(weight)                      \
    _(ax)                          \
    _(ay)                          \
    _(spin_x)                      \
    _(spin_y)                      \
    _(spin_z)                      \
    _(anomalous_magnetic_moment)

// XT_FREEZE_VAR_* is the value-style counterpart of the public presence-style
// FREEZE_VAR_* configuration. Undefined flags default to unfrozen.
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
#ifndef XT_FREEZE_VAR_pdg_id
#define XT_FREEZE_VAR_pdg_id 0
#endif
#ifndef XT_FREEZE_VAR_particle_id
#define XT_FREEZE_VAR_particle_id 0
#endif
#ifndef XT_FREEZE_VAR_at_element
#define XT_FREEZE_VAR_at_element 0
#endif
#ifndef XT_FREEZE_VAR_at_turn
#define XT_FREEZE_VAR_at_turn 0
#endif
#ifndef XT_FREEZE_VAR_state
#define XT_FREEZE_VAR_state 0
#endif
#ifndef XT_FREEZE_VAR_parent_particle_id
#define XT_FREEZE_VAR_parent_particle_id 0
#endif
#ifndef XT_FREEZE_VAR__rng_s1
#define XT_FREEZE_VAR__rng_s1 0
#endif
#ifndef XT_FREEZE_VAR__rng_s2
#define XT_FREEZE_VAR__rng_s2 0
#endif
#ifndef XT_FREEZE_VAR__rng_s3
#define XT_FREEZE_VAR__rng_s3 0
#endif
#ifndef XT_FREEZE_VAR__rng_s4
#define XT_FREEZE_VAR__rng_s4 0
#endif

#define XT_LP_IS_FROZEN(NAME) XT_LP_IS_FROZEN_IMPL(NAME)
#define XT_LP_IS_FROZEN_IMPL(NAME) XT_FREEZE_VAR_ ## NAME

// Backends provide the storage operations used below.
#define XT_LP_NUM_ACCESSORS(NAME)                                                  \
    GPUFUN                                                                        \
    void LocalParticle_add_to_ ## NAME(LocalParticle* part, xt_num_arg_t value){   \
        if (!XT_LP_IS_FROZEN(NAME)) {                                              \
            NUM_ADD(part, NAME, value);                                            \
        }                                                                          \
    }                                                                              \
    GPUFUN                                                                        \
    xt_num_t LocalParticle_get_ ## NAME(LocalParticle* part){                      \
        return NUM_GET(part, NAME);                                                \
    }                                                                              \
    GPUFUN                                                                        \
    void LocalParticle_set_ ## NAME(LocalParticle* part, xt_num_arg_t value){      \
        if (!XT_LP_IS_FROZEN(NAME)) {                                              \
            NUM_SET(part, NAME, value);                                            \
        }                                                                          \
    }                                                                              \
    GPUFUN                                                                        \
    void LocalParticle_scale_ ## NAME(LocalParticle* part, xt_num_arg_t value){    \
        if (!XT_LP_IS_FROZEN(NAME)) {                                              \
            NUM_SCALE(part, NAME, value);                                          \
        }                                                                          \
    }

#define XT_LP_TYPED_ACCESSORS(TYPE, NAME)                                          \
    GPUFUN                                                                        \
    void LocalParticle_add_to_ ## NAME(LocalParticle* part, TYPE value){           \
        if (!XT_LP_IS_FROZEN(NAME)) {                                              \
            TYPED_ADD(part, NAME, value);                                          \
        }                                                                          \
    }                                                                              \
    GPUFUN                                                                        \
    TYPE LocalParticle_get_ ## NAME(LocalParticle* part){                          \
        return TYPED_GET(part, NAME);                                              \
    }                                                                              \
    GPUFUN                                                                        \
    void LocalParticle_set_ ## NAME(LocalParticle* part, TYPE value){              \
        if (!XT_LP_IS_FROZEN(NAME)) {                                              \
            TYPED_SET(part, NAME, value);                                          \
        }                                                                          \
    }                                                                              \
    GPUFUN                                                                        \
    void LocalParticle_scale_ ## NAME(LocalParticle* part, TYPE value){            \
        if (!XT_LP_IS_FROZEN(NAME)) {                                              \
            TYPED_SCALE(part, NAME, value);                                        \
        }                                                                          \
    }

#endif /* XTRACK_PARTICLES_LOCAL_PARTICLE_MACROS_H */
