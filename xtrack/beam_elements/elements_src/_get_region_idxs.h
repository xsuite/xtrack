//
// Created by simonfan on 09/12/2025.
// This header is adapted from NumPy's searchsorted implementation for performance.
// It provides a function to find the indices of regions for given values based on specified boundaries.
// It is part of the bpmeth field generator + Boris integrator sequence.

// Possibly: Instead of passing indices that set in which region the particle is,
// we can: Decide where each particle is, assign a set of parameters for each particle,
// pass that to the field evaluation function.

#include <stddef.h>

#ifndef XSUITE_GET_REGION_IDXS_H
#define XSUITE_GET_REGION_IDXS_H

static size_t searchsorted_right_branchless(const double *arr, size_t n, double value)
{
    size_t idx = 0;
    size_t step = 1ULL << (63 - __builtin_clzll(n));  // largest power of two <= n

    while (step > 0) {
        size_t next = idx + step;
        if (next <= n && value >= arr[next - 1]) {
            idx = next;
        }
        step >>= 1;
    }

    return idx;  // same as numpy.searchsorted(..., side="right")
}

static size_t searchsorted_right(const double *arr, size_t n, double value)
{
    // Fallback to binary search for small arrays
    if (n < 16) {
        size_t left = 0;
        size_t right = n;

        while (left < right) {
            size_t mid = left + (right - left) / 2;
            if (value < arr[mid]) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;  // same as numpy.searchsorted(..., side="right")
    } else {
        return searchsorted_right_branchless(arr, n, value);
    }
}

void select_region_array(const double *s_boundaries, size_t n_boundaries,
                         const double *s_val, size_t n_vals,
                         ptrdiff_t *idxs_out)
{
    for (size_t i = 0; i < n_vals; i++) {
        size_t pos = searchsorted_right(s_boundaries, n_boundaries, s_val[i]);
        idxs_out[i] = (ptrdiff_t)pos - 1;   // match Python behavior
    }
}

#endif //XSUITE_GET_REGION_IDXS_H