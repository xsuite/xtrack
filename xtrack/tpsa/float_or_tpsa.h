// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_TPSA_FLOAT_OR_TPSA_H
#define XTRACK_TPSA_FLOAT_OR_TPSA_H

#include "xtrack/headers/track.h"

#ifdef XTRACK_TPSA_TRACK

#include <new>
#include <type_traits>

// Convert a double coefficient array to a short-lived XT_NUM array. Scalar
// constructors convert each entry to a constant TPSA using the active descriptor.
class xt_tpsa_lifted_array {
public:
    typedef XT_NUM value_t;

    xt_tpsa_lifted_array(const double* values, int64_t size)
        : storage_(NULL), data_(NULL), size_(0) {
        if (values == NULL || size <= 0) return;

        storage_ = new storage_t[size];
        data_ = reinterpret_cast<value_t*>(storage_);

        for (; size_ < size; size_++) {
            new (&data_[size_]) value_t(values[size_]);
        }
    }

    ~xt_tpsa_lifted_array() {
        for (int64_t ii = 0; ii < size_; ii++) {
            data_[ii].~value_t();
        }
        delete[] storage_;
    }

    const value_t* ptr() const {
        return data_;
    }

private:
    typedef typename std::aligned_storage<
        sizeof(value_t), alignof(value_t)>::type storage_t;

    xt_tpsa_lifted_array(const xt_tpsa_lifted_array&);
    xt_tpsa_lifted_array& operator=(const xt_tpsa_lifted_array&);

    storage_t* storage_;
    value_t* data_;
    int64_t size_;
};

#define XT_KICK_SIMPLE(pt, ord, invf, KN, KS, fac, kw) do { \
        xt_tpsa_lifted_array _kn((KN), (ord)+1); \
        xt_tpsa_lifted_array _ks((KS), (ord)+1); \
        kick_simple_single_particle((pt),(ord),(invf),_kn.ptr(),_ks.ptr(),(fac),(kw)); \
    } while(0)

#else

#define XT_KICK_SIMPLE(pt, ord, invf, KN, KS, fac, kw) \
        kick_simple_single_particle((pt),(ord),(invf),(KN),(KS),(fac),(kw))

#endif

#endif
