// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_TPSA_XT_TPSA_H
#define XTRACK_TPSA_XT_TPSA_H

#include <cstdint>
#include <math.h>
#include "mad_tpsa.hpp"

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

static inline double xt_num_truncate_to_double(xt_num_arg_t value){
    return value[0];
}

#define XT_TPSA_REL(OP) \
  template<class A> inline bool operator OP (const mad::tpsa_base<A>& a, double b){ return a[0] OP b; } \
  template<class A> inline bool operator OP (double a, const mad::tpsa_base<A>& b){ return a OP b[0]; } \
  template<class A, class B> inline bool operator OP (const mad::tpsa_base<A>& a, const mad::tpsa_base<B>& b){ return a[0] OP b[0]; }
XT_TPSA_REL(>) XT_TPSA_REL(<) XT_TPSA_REL(>=) XT_TPSA_REL(<=) XT_TPSA_REL(==) XT_TPSA_REL(!=)
#undef XT_TPSA_REL

typedef void* SynchrotronRadiationRecordData;

// Decode a scalar FloatOrTpsa slot stored as raw uint64_t bits.
static inline double xt_float_or_tpsa_bits_to_double(uint64_t bits){
    union { uint64_t u; double d; } value;
    value.u = bits;
    return value.d;
}

// Promote a scalar constant to the active TPSA descriptor.
static inline xt_num_t xt_float_or_tpsa_lift(double value){
    return xt_num_t(value);
}

// Decode a scalar FloatOrTpsa slot and promote it to the active TPSA descriptor.
static inline xt_num_t xt_float_or_tpsa_lift(uint64_t bits){
    return xt_float_or_tpsa_lift(xt_float_or_tpsa_bits_to_double(bits));
}

// Read a FloatOrTpsa slot as either a TPSA pointer or a lifted scalar.
static inline xt_num_t xt_float_or_tpsa_get(uint64_t* slot, int64_t tpsa_enabled){
    if (tpsa_enabled) {
        return 1.0 * mad::tpsa_ref((tpsa_t*)(uintptr_t)(*slot));
    }
    return xt_float_or_tpsa_lift(*slot);
}

#endif
