// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_MULTISETTER_H
#define XTRACK_MULTISETTER_H

#include "xobjects/headers/common.h"


GPUKERN
void get_values_at_offsets_float64(
    MultiSetterData data,
    GPUGLMEM int8_t* buffer,
    GPUGLMEM double* out){

    int64_t num_offsets = MultiSetterData_len_offsets(data);

    VECTORIZE_OVER(ii, num_offsets);
        int64_t offs = MultiSetterData_get_offsets(data, ii);

        double val = *((GPUGLMEM double*)(buffer + offs));
        out[ii] = val;
    END_VECTORIZE;
}

GPUKERN
void get_values_at_offsets_int64(
    MultiSetterData data,
    GPUGLMEM int8_t* buffer,
    GPUGLMEM int64_t* out){

    int64_t num_offsets = MultiSetterData_len_offsets(data);

    VECTORIZE_OVER(ii, num_offsets);
        int64_t offs = MultiSetterData_get_offsets(data, ii);

        int64_t val = *((GPUGLMEM int64_t*)(buffer + offs));
        out[ii] = val;
    END_VECTORIZE;
}

GPUKERN
void set_values_at_offsets_float64(
    MultiSetterData data,
    GPUGLMEM int8_t* buffer,
    GPUGLMEM double* input){

    int64_t num_offsets = MultiSetterData_len_offsets(data);

    VECTORIZE_OVER(ii, num_offsets);
        int64_t offs = MultiSetterData_get_offsets(data, ii);

        double val = input[ii];
        *((GPUGLMEM double*)(buffer + offs)) = val;
    END_VECTORIZE;
}

GPUKERN
void set_values_at_offsets_int64(
    MultiSetterData data,
    GPUGLMEM int8_t* buffer,
    GPUGLMEM int64_t* input){

    int64_t num_offsets = MultiSetterData_len_offsets(data);

    VECTORIZE_OVER(ii, num_offsets);
        int64_t offs = MultiSetterData_get_offsets(data, ii);

        int64_t val = input[ii];
        *((GPUGLMEM int64_t*)(buffer + offs)) = val;
    END_VECTORIZE;
}

#endif /* XTRACK_MULTISETTER_H */