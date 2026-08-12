// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //

#ifndef XTRACK_TPSA_LOCAL_PARTICLE_H
#define XTRACK_TPSA_LOCAL_PARTICLE_H

#include "xtrack/headers/track.h"
#include "xtrack/particles/headers/local_particle_common.h"

struct LocalParticle {
#define XT_TPSA_LOCAL_PARTICLE_STRUCT_FIELD(NAME) tpsa_t *NAME;
    XT_LOCAL_PARTICLE_TPSA_NUM_FIELDS(XT_TPSA_LOCAL_PARTICLE_STRUCT_FIELD)
#undef XT_TPSA_LOCAL_PARTICLE_STRUCT_FIELD

#define XT_TPSA_LOCAL_PARTICLE_REF_FIELD(NAME) double NAME;
    XT_LOCAL_PARTICLE_REF_FIELDS(XT_TPSA_LOCAL_PARTICLE_REF_FIELD)
#undef XT_TPSA_LOCAL_PARTICLE_REF_FIELD

    int64_t state;
    int64_t at_element;
    double line_length;
    int64_t ipart, endpart, _num_active_particles, _num_lost_particles;
    uint64_t track_flags;
    int8_t* io_buffer;
};

// Convert between the xobject kernel argument and the unrolled LocalParticle used by tracking.
static inline void Particles_to_LocalParticle(
        TpsaParticleData source, LocalParticle* dest, int64_t id, int64_t end_id) {
#define XT_TPSA_LOCAL_PARTICLE_COPY_NUM_FIELD(NAME) \
    dest->NAME = (tpsa_t*)(uintptr_t)TpsaParticleData_get_##NAME(source);
    XT_LOCAL_PARTICLE_TPSA_NUM_FIELDS(XT_TPSA_LOCAL_PARTICLE_COPY_NUM_FIELD)
#undef XT_TPSA_LOCAL_PARTICLE_COPY_NUM_FIELD

#define XT_TPSA_LOCAL_PARTICLE_COPY_REF_FIELD(NAME) \
    dest->NAME = TpsaParticleData_get_##NAME(source);
    XT_LOCAL_PARTICLE_REF_FIELDS(XT_TPSA_LOCAL_PARTICLE_COPY_REF_FIELD)
#undef XT_TPSA_LOCAL_PARTICLE_COPY_REF_FIELD

    dest->state = TpsaParticleData_get_state(source);
    dest->at_element = TpsaParticleData_get_at_element(source);
    dest->ipart = id;
    dest->endpart = end_id;
}

static inline void LocalParticle_to_Particles(
        LocalParticle* source, TpsaParticleData dest, int64_t, int64_t) {
    TpsaParticleData_set_state(dest, source->state);
    TpsaParticleData_set_at_element(dest, source->at_element);
    TpsaParticleData_set_track_flags(dest, source->track_flags);
    TpsaParticleData_set_line_length(dest, source->line_length);
}

static inline int64_t LocalParticle_get__num_active_particles(LocalParticle* p){
    return p->_num_active_particles;
}

static inline uint64_t LocalParticle_check_track_flag(LocalParticle* p, uint8_t index){
    return (p->track_flags >> index) & 1;
}

static inline void LocalParticle_exchange(LocalParticle*, int64_t, int64_t){}
static inline void LocalParticle_add_to_at_turn(LocalParticle*, int64_t){}
static inline int64_t LocalParticle_get_at_turn(LocalParticle*){ return 0; }
static inline int64_t LocalParticle_get_particle_id(LocalParticle*){ return 0; }
static inline int8_t* LocalParticle_get_io_buffer(LocalParticle* p){ return p->io_buffer; }

#define XT_TPSA_LOCAL_PARTICLE_TEMPLATE_ACCESSORS(NAME)                             \
    template<class A>                                                               \
    static inline void LocalParticle_add_to_##NAME(                                 \
            LocalParticle* p, const mad::tpsa_base<A>& v){                          \
        mad::tpsa_ref(p->NAME) += v;                                                \
    }                                                                               \
    template<class A>                                                               \
    static inline void LocalParticle_set_##NAME(                                    \
            LocalParticle* p, const mad::tpsa_base<A>& v){                          \
        mad::tpsa_ref(p->NAME) = v;                                                 \
    }                                                                               \
    template<class A>                                                               \
    static inline void LocalParticle_scale_##NAME(                                  \
            LocalParticle* p, const mad::tpsa_base<A>& v){                          \
        mad::tpsa_ref(p->NAME) *= v;                                                \
    }

#define XT_LOCAL_PARTICLE_ACCESSOR_PREFIX static inline
#define XT_LOCAL_PARTICLE_NUM_GET(PART, NAME) (1.0 * mad::tpsa_ref((PART)->NAME))
#define XT_LOCAL_PARTICLE_NUM_SET(PART, NAME, VALUE) (mad::tpsa_ref((PART)->NAME) = (VALUE))
#define XT_LOCAL_PARTICLE_NUM_ADD(PART, NAME, VALUE) (mad::tpsa_ref((PART)->NAME) += (VALUE))
#define XT_LOCAL_PARTICLE_NUM_SCALE(PART, NAME, VALUE) (mad::tpsa_ref((PART)->NAME) *= (VALUE))

XT_LOCAL_PARTICLE_TPSA_NUM_FIELDS(XT_TPSA_LOCAL_PARTICLE_TEMPLATE_ACCESSORS)
XT_LOCAL_PARTICLE_TPSA_NUM_FIELDS(XT_LOCAL_PARTICLE_NUM_ACCESSORS)
#undef XT_LOCAL_PARTICLE_NUM_SCALE
#undef XT_LOCAL_PARTICLE_NUM_ADD
#undef XT_LOCAL_PARTICLE_NUM_SET
#undef XT_LOCAL_PARTICLE_NUM_GET
#undef XT_LOCAL_PARTICLE_ACCESSOR_PREFIX
#undef XT_TPSA_LOCAL_PARTICLE_TEMPLATE_ACCESSORS

#define XT_TPSA_LOCAL_PARTICLE_REF_ACCESSOR(NAME)                                  \
    static inline double LocalParticle_get_##NAME(LocalParticle* p){               \
        return p->NAME;                                                            \
    }

XT_LOCAL_PARTICLE_REF_FIELDS(XT_TPSA_LOCAL_PARTICLE_REF_ACCESSOR)
#undef XT_TPSA_LOCAL_PARTICLE_REF_ACCESSOR

static inline int64_t LocalParticle_get_state(LocalParticle* p){
    return p->state;
}

static inline void LocalParticle_set_state(LocalParticle* p, int64_t v){
    p->state = v;
}

static inline void LocalParticle_add_to_state(LocalParticle* p, int64_t v){
    p->state += v;
}

static inline int64_t LocalParticle_get_at_element(LocalParticle* p){
    return p->at_element;
}

static inline void LocalParticle_set_at_element(LocalParticle* p, int64_t v){
    p->at_element = v;
}

static inline void LocalParticle_add_to_at_element(LocalParticle* p, int64_t v){
    p->at_element += v;
}

static inline double LocalParticle_get_energy0(LocalParticle* part) {
    double const p0c = LocalParticle_get_p0c(part);
    double const m0  = LocalParticle_get_mass0(part);
    return sqrt(p0c * p0c + m0 * m0);
}

static inline void LocalParticle_update_delta(LocalParticle* part, xt_num_arg_t new_delta_value) {
    double const beta0 = LocalParticle_get_beta0(part);
    xt_num_t const irpp = new_delta_value + 1.0;
    xt_num_t const rpp = 1.0 / irpp;
    xt_num_t const ptau = (sqrt(1.0 + 2.0 * beta0 * new_delta_value
                              + new_delta_value * new_delta_value) - 1.0) / beta0;
    xt_num_t const rvv = rpp * (1.0 + beta0 * ptau);
    LocalParticle_set_delta(part, new_delta_value);
    LocalParticle_set_rvv(part, rvv);
    LocalParticle_set_rpp(part, rpp);
    LocalParticle_set_ptau(part, ptau);
}

static inline void LocalParticle_update_delta(LocalParticle* part, double value) {
    LocalParticle_update_delta(part, xt_float_or_tpsa_lift(value));
}

static inline void LocalParticle_update_ptau(LocalParticle* part, xt_num_arg_t new_ptau_value) {
    double const beta0 = LocalParticle_get_beta0(part);
    xt_num_t const delta = sqrt(1.0 + 2.0 * beta0 * new_ptau_value
                              + new_ptau_value * new_ptau_value) - 1.0;
    LocalParticle_update_delta(part, delta);
}

static inline void LocalParticle_update_ptau(LocalParticle* part, double value) {
    LocalParticle_update_ptau(part, xt_float_or_tpsa_lift(value));
}

static inline xt_num_t LocalParticle_get_pzeta(LocalParticle* part) {
    xt_num_t const ptau = LocalParticle_get_ptau(part);
    double const beta0 = LocalParticle_get_beta0(part);
    return ptau / beta0;
}

static inline void LocalParticle_update_pzeta(LocalParticle* part, xt_num_arg_t new_pzeta_value) {
    double const beta0 = LocalParticle_get_beta0(part);
    LocalParticle_update_ptau(part, beta0 * new_pzeta_value);
}

static inline void LocalParticle_add_to_energy(
        LocalParticle* part, xt_num_arg_t delta_energy, int pz_only) {
    xt_num_t ptau = LocalParticle_get_ptau(part);
    double const p0c = LocalParticle_get_p0c(part);
    double const charge_ratio = LocalParticle_get_charge_ratio(part);
    double const chi = LocalParticle_get_chi(part);
    double const mass_ratio = charge_ratio / chi;

    ptau += delta_energy / p0c / mass_ratio;

    xt_num_t const old_rpp = LocalParticle_get_rpp(part);

    LocalParticle_update_ptau(part, ptau);

    if (!pz_only) {
        xt_num_t const new_rpp = LocalParticle_get_rpp(part);
        xt_num_t const f = old_rpp / new_rpp;
        LocalParticle_scale_px(part, f);
        LocalParticle_scale_py(part, f);
    }
}

static inline void LocalParticle_add_to_energy(
        LocalParticle* part, double delta_energy, int pz_only) {
    LocalParticle_add_to_energy(part, xt_float_or_tpsa_lift(delta_energy), pz_only);
}

static inline void increment_at_element(LocalParticle* part, int64_t increment) {
    LocalParticle_add_to_at_element(part, increment);
}

static inline void increment_at_turn(LocalParticle* part, int) {
    LocalParticle_add_to_at_turn(part, 1);
    LocalParticle_set_at_element(part, 0);
    LocalParticle_set_s(part, 0.0);
}

static inline void increment_at_turn_backtrack(
        LocalParticle*, int, double, int64_t) {}

static inline void LocalParticle_kill_particle(LocalParticle* part, int64_t kill_state) {
    LocalParticle_set_x(part, 1e30);
    LocalParticle_set_px(part, 1e30);
    LocalParticle_set_y(part, 1e30);
    LocalParticle_set_py(part, 1e30);
    LocalParticle_set_zeta(part, 1e30);
    LocalParticle_update_delta(part, -1.0);
    LocalParticle_set_state(part, kill_state);
}

static inline int64_t check_is_active(LocalParticle* part) {
    return LocalParticle_get_state(part) > 0;
}

#endif
