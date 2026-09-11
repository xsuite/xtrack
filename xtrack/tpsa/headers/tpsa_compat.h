// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_TPSA_COMPAT_H
#define XTRACK_TPSA_COMPAT_H

typedef double xt_float_or_tpsa;
typedef const xt_float_or_tpsa xt_float_or_tpsa_arg;

static inline double xt_float_or_tpsa_const_part(xt_float_or_tpsa_arg value){
    return value;
}

static inline int xt_float_or_tpsa_is_zero(xt_float_or_tpsa_arg value){
    return value == 0.0;
}

// Decode a scalar FloatOrTpsa slot stored as raw uint64_t bits.
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
    double coef[1];
} xt_float_or_tpsa_tpsa_t;

// Read a FloatOrTpsa slot as a scalar, truncating TPSA slots to coef[0].
static inline double xt_float_or_tpsa_get_double(uint64_t bits, uint8_t enabled){
    if (enabled) {
        return ((xt_float_or_tpsa_tpsa_t*)bits)->coef[0];
    }
    return xt_float_or_tpsa_bits_to_double(bits);
}

#endif
