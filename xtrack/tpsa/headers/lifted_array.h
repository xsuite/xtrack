// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2026.                 //
// ######################################### //
#ifndef XTRACK_TPSA_LIFTED_ARRAY_H
#define XTRACK_TPSA_LIFTED_ARRAY_H

#include <new>
#include <type_traits>

// C++ can convert one scalar to xt_num_t, e.g. `xt_num_t x = 1.0`, but it
// cannot convert an existing `double*` array to an `xt_num_t*` array. For
// example, a call expecting `const xt_num_t*` cannot accept a `const double*`,
// even though each individual double could be converted to a constant TPSA.
//
// This class builds a short-lived xt_num_t array from a double array by
// constructing each entry separately. The xt_num_t scalar constructor uses the
// active TPSA descriptor, so each double becomes a constant TPSA. The class owns
// the temporary array and destroys it when it goes out of scope.
//
// This is a convenience class that might be able to be removed once we
// remove double* -> xt_num_t* promotions, which should be possible.
class xt_tpsa_lifted_array
{
public:
    using value_t = xt_tpsa::tpsa;

    // No copy/move constructor/assignment: it would break storage ownership
    xt_tpsa_lifted_array(const xt_tpsa_lifted_array&) = delete;
    xt_tpsa_lifted_array& operator=(const xt_tpsa_lifted_array&) = delete;
    xt_tpsa_lifted_array(xt_tpsa_lifted_array&&) = delete;
    xt_tpsa_lifted_array& operator=(xt_tpsa_lifted_array&&) = delete;

    explicit xt_tpsa_lifted_array(const double* values, int64_t size)
    {
        if (values == nullptr || size <= 0) return;

        storage = new storage_t[size];
        data = reinterpret_cast<value_t*>(storage);

        for (; this->size < size; this->size++)
        {
            new (&data[this->size]) value_t(values[this->size]);
        }
    }

    ~xt_tpsa_lifted_array()
    {
        for (int64_t ii = 0; ii < size; ii++)
        {
            data[ii].~value_t();
        }
        delete[] storage;
    }

    const value_t* ptr() const
    {
        return data;
    }

private:
    using storage_t = std::aligned_storage_t<sizeof(value_t), alignof(value_t)>;

    storage_t* storage = nullptr;
    value_t* data = nullptr;
    int64_t size = 0;
};

#endif
