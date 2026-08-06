// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_TRACK_H
#define XTRACK_TRACK_H

#include "xobjects/headers/common.h"
#include "xobjects/headers/atomicadd.h"
#include "xtrack/headers/constants.h"

#ifdef XTRACK_TPSA_TRACK

#include <cstdint>
#include <math.h>
#include "mad_tpsa.hpp"

using mad::tpsa;
using mad::tpsa_ref;
namespace xt_tpsa {
struct tpsa : public mad::tpsa {
    inline static thread_local tpsa_t* default_proto = nullptr;

    using mad::tpsa::tpsa;

    tpsa(double value)
        : mad::tpsa(0.0 * mad::tpsa_ref(default_proto) + value) {}

    tpsa& operator=(double value) {
        mad::tpsa::operator=(value);
        return *this;
    }

    tpsa& operator=(const tpsa& value) {
        mad::tpsa::operator=(static_cast<const mad::tpsa&>(value));
        return *this;
    }

    tpsa& operator=(tpsa&& value) {
        mad::tpsa::operator=(static_cast<const mad::tpsa&>(value));
        return *this;
    }

    template<class A>
    tpsa& operator=(const mad::tpsa_base<A>& value) {
        mad::tpsa::operator=(value);
        return *this;
    }
};

struct default_scope {
    default_scope(tpsa_t* proto) {
        tpsa::default_proto = proto;
    }
    ~default_scope() {
        tpsa::default_proto = nullptr;
    }
};
}

typedef xt_tpsa::tpsa xt_num_t;
typedef const xt_num_t& xt_num_arg_t;

static inline double xt_num_const_part(xt_num_arg_t value){
    return value[0];
}

#define XT_TPSA_REL(OP) \
  template<class A> inline bool operator OP (const mad::tpsa_base<A>& a, double b){ return a[0] OP b; } \
  template<class A> inline bool operator OP (double a, const mad::tpsa_base<A>& b){ return a OP b[0]; } \
  template<class A, class B> inline bool operator OP (const mad::tpsa_base<A>& a, const mad::tpsa_base<B>& b){ return a[0] OP b[0]; }
XT_TPSA_REL(>) XT_TPSA_REL(<) XT_TPSA_REL(>=) XT_TPSA_REL(<=) XT_TPSA_REL(==) XT_TPSA_REL(!=)
#undef XT_TPSA_REL

typedef tpsa_t XT_COORD;
typedef void* SynchrotronRadiationRecordData;

static inline double xt_float_or_tpsa_bits_to_double(uint64_t bits){
    union { uint64_t u; double d; } value;
    value.u = bits;
    return value.d;
}

static inline xt_num_t xt_float_or_tpsa_lift(double value){
    return xt_num_t(value);
}

static inline xt_num_t xt_float_or_tpsa_lift(uint64_t bits){
    return xt_float_or_tpsa_lift(xt_float_or_tpsa_bits_to_double(bits));
}

template<class A>
static inline xt_num_t xt_float_or_tpsa_lift(const mad::tpsa_base<A>& value){
    return xt_num_t(value);
}

template<class ElementData>
static inline xt_num_t xt_float_or_tpsa_get(
        ElementData, uint64_t* slot, int64_t tpsa_enabled){
    if (tpsa_enabled) {
        return 1.0 * mad::tpsa_ref((tpsa_t*)(uintptr_t)(*slot));
    }
    return xt_float_or_tpsa_lift(*slot);
}

#else

typedef double xt_num_t;
typedef const xt_num_t xt_num_arg_t;

static inline double xt_num_const_part(xt_num_arg_t value){
    return value;
}

static inline double xt_float_or_tpsa_bits_to_double(uint64_t bits){
    union { uint64_t u; double d; } value;
    value.u = bits;
    return value.d;
}

typedef unsigned char xt_float_or_tpsa_ord_t;
typedef struct xt_float_or_tpsa_desc_ xt_float_or_tpsa_desc_t;
typedef struct xt_float_or_tpsa_tpsa_ {
    const xt_float_or_tpsa_desc_t *d;
    xt_float_or_tpsa_ord_t lo, hi, mo, ao;
    int32_t uid;
    char nam[16];
    double coef[];
} xt_float_or_tpsa_tpsa_t;

static inline double xt_float_or_tpsa_get_double(uint64_t bits, uint8_t enabled){
    if (enabled) {
        return ((xt_float_or_tpsa_tpsa_t*)(uintptr_t)bits)->coef[0];
    }
    return xt_float_or_tpsa_bits_to_double(bits);
}

#endif

/*
    The particle tracking "decorators" for all the contexts.
*/

#ifdef XO_CONTEXT_CPU_SERIAL
    // We are on CPU, without OpenMP

    #define START_PER_PARTICLE_BLOCK(SRC_PART, DEST_PART) { \
            const int64_t XT_part_block_start_idx = 0; \
            const int64_t XT_part_block_end_idx = LocalParticle_get__num_active_particles((SRC_PART)); \
            for (int64_t XT_part_block_ii = XT_part_block_start_idx; XT_part_block_ii<XT_part_block_end_idx; XT_part_block_ii++) \
            { \
                LocalParticle lpart = *(SRC_PART); \
                LocalParticle* DEST_PART = &lpart; \
                part->ipart = XT_part_block_ii;

    #define END_PER_PARTICLE_BLOCK \
            } \
        }
#endif  // XO_CONTEXT_CPU_SERIAL

#ifdef XO_CONTEXT_CPU_OPENMP
    // We are on CPU with the OpenMP context switched on

    #define START_PER_PARTICLE_BLOCK(SRC_PART, DEST_PART) { \
            const int64_t _part_block_start_idx = (SRC_PART)->ipart; \
            const int64_t _part_block_end_idx = (SRC_PART)->endpart; \
            for (int64_t _part_block_idx = _part_block_start_idx; _part_block_idx < _part_block_end_idx; _part_block_idx++) \
            { \
                LocalParticle lpart = *(SRC_PART); \
                LocalParticle* DEST_PART = &lpart; \
                part->ipart = _part_block_idx; \
                \
                if (LocalParticle_get_state(DEST_PART) > 0) {

    #define END_PER_PARTICLE_BLOCK \
                } \
            } \
        }
#endif  // XO_CONTEXT_CPU_OPENMP


#ifdef XO_CONTEXT_CUDA
    // We are on a CUDA GPU

    #define START_PER_PARTICLE_BLOCK(SRC_PART, DEST_PART) { \
            LocalParticle* DEST_PART = (SRC_PART);

    #define END_PER_PARTICLE_BLOCK \
            }
#endif  // XO_CONTEXT_CUDA


#ifdef XO_CONTEXT_CL
    // We are on an OpenCL GPU

    #define START_PER_PARTICLE_BLOCK(SRC_PART, DEST_PART) { \
            LocalParticle* DEST_PART = (SRC_PART);

    #define END_PER_PARTICLE_BLOCK \
            }
#endif  // XO_CONTEXT_CL


#ifndef START_PER_PARTICLE_BLOCK
#error "Unknown context, or the expected context (XO_CONTEXT_*) flag undefined. Try updating Xobjects?"
#endif

#endif  // XTRACK_TRACK_H
