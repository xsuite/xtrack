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
    /* Subclass mad::tpsa to implement custom behaviours for Xtrack.
     *
     * Extend mad::tpsa so scalar constants can be promoted through normal C++
     * construction syntax, e.g. `tpsa a = 7.0;`, using the active prototype.
     * Do the same for constructors.
     */
    struct tpsa : public mad::tpsa {
        inline static thread_local tpsa_t* default_proto = nullptr;

        // Promote scalar constants using the active TPSA prototype.
        tpsa(double value)
            : mad::tpsa(mad::tpsa_ref(default_proto)) {
            mad::tpsa::operator=(value);
        }

        // GCC resolves literal 0 against the deleted tpsa(nullptr_t) constructor
        // instead of tpsa(double) (which is what Clang does), so let's be explicit.
        tpsa(int value)
            : tpsa(static_cast<double>(value)) {}

        // Default mad::tpsa constructor does not copy coefficients. This can be a little
        // confusing, so we override it so that tpsa(tpsa) just makes a copy.
        // For a new TPSA (in tracking) that is zero it suffices to write:
        // `xt_tpsa::tpsa my_tpsa = 0;`: this will create a zero-valued TPSA using
        // the `default_proto` as the prototype. This should be sufficient for now.
        tpsa(const tpsa& value)
            : mad::tpsa(static_cast<const mad::tpsa&>(value)) {
            mad::tpsa::operator=(static_cast<const mad::tpsa&>(value));
        }

        // Construct an xt_tpsa::tpsa as a copy of mad::tpsa
        template<class A>
        tpsa(const mad::tpsa_base<A>& value)
            : mad::tpsa(value) {
            mad::tpsa::operator=(value);
        }

        // Forwarding shim to mad::tpsa, returns our overloaded type
        tpsa& operator=(double value) {
            mad::tpsa::operator=(value);
            return *this;
        }

        // Forwarding shim to mad::tpsa, returns our overloaded type
        tpsa& operator=(const tpsa& value) {
            mad::tpsa::operator=(static_cast<const mad::tpsa&>(value));
            return *this;
        }

        // Forwarding shim to mad::tpsa, returns our overloaded type
        template<class A>
        tpsa& operator=(const mad::tpsa_base<A>& value) {
            mad::tpsa::operator=(value);
            return *this;
        }
};

// Keep this scope alive for the full lifetime of TPSA tracking, or for any
// other operation that may construct xt_tpsa::tpsa values from scalar constants.
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

// Define relational operators between TPSAs and between TPSAs and scalars.
#define XT_TPSA_REL(OP) \
  template<class A> \
  inline bool operator OP (const mad::tpsa_base<A>& a, double b){ return a[0] OP b; } \
  template<class A> \
  inline bool operator OP (double a, const mad::tpsa_base<A>& b){ return a OP b[0]; } \
  template<class A, class B> \
  inline bool operator OP (const mad::tpsa_base<A>& a, const mad::tpsa_base<B>& b){ \
      return a[0] OP b[0]; \
  }
XT_TPSA_REL(>) XT_TPSA_REL(<) XT_TPSA_REL(>=) XT_TPSA_REL(<=) XT_TPSA_REL(==) XT_TPSA_REL(!=)
#undef XT_TPSA_REL

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
        return xt_num_t(mad::tpsa_ref((tpsa_t*)(uintptr_t)(*slot)));
    }
    return xt_float_or_tpsa_lift(*slot);
}

#endif
