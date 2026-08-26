// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_TRACK_H
#define XTRACK_TRACK_H

#include "xobjects/headers/common.h"
#include "xobjects/headers/atomicadd.h"
#include "xtrack/headers/constants.h"

// Common token-concatenation helpers used by C API templates.
#ifndef _CONCAT
    #define _CONCAT(a, b) a ## b
    #define CONCAT(a, b) _CONCAT(a, b)
    #define CONCAT3(a, b, c) CONCAT(CONCAT(a, b), c)
#endif /* !defined(_CONCAT) */

#ifdef XTRACK_TPSA_TRACK
    #define TPSA_TRACKING_BOOL 1
    #include "xtrack/tpsa/headers/xt_tpsa.h"
    #include "xtrack/tpsa/headers/lifted_array.h"
#else
    #define TPSA_TRACKING_BOOL 0
    #include "xtrack/tpsa/headers/tpsa_compat.h"
#endif


/*
    The particle tracking "decorators" for all the contexts.
*/

#ifdef XO_CONTEXT_CPU_SERIAL
    // We are on CPU, without OpenMP

    #define START_PER_PARTICLE_BLOCK(SRC_PART, DEST_PART) { \
            const int64_t XT_part_block_start_idx = 0; \
            const int64_t XT_part_block_end_idx = \
                LocalParticle_get__num_active_particles((SRC_PART)); \
            for (int64_t XT_part_block_ii = XT_part_block_start_idx; \
                 XT_part_block_ii < XT_part_block_end_idx; XT_part_block_ii++) \
            { \
                LocalParticle lpart = *(SRC_PART); \
                LocalParticle* DEST_PART = &lpart; \
                (DEST_PART)->ipart = XT_part_block_ii;

    #define END_PER_PARTICLE_BLOCK \
            } \
        }
#endif  // XO_CONTEXT_CPU_SERIAL

#ifdef XO_CONTEXT_CPU_OPENMP
    // We are on CPU with the OpenMP context switched on

    #define START_PER_PARTICLE_BLOCK(SRC_PART, DEST_PART) { \
            const int64_t _part_block_start_idx = (SRC_PART)->ipart; \
            const int64_t _part_block_end_idx = (SRC_PART)->endpart; \
            for (int64_t _part_block_idx = _part_block_start_idx; \
                 _part_block_idx < _part_block_end_idx; _part_block_idx++) \
            { \
                LocalParticle lpart = *(SRC_PART); \
                LocalParticle* DEST_PART = &lpart; \
                (DEST_PART)->ipart = _part_block_idx; \
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
#error "Unknown context or missing XO_CONTEXT_* flag. Try updating Xobjects?"
#endif

// LocalParticle to be included after defining the per-particle macros used by its API.
#include "xtrack/particles/headers/local_particle.h"

#endif  // XTRACK_TRACK_H
