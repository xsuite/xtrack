// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_TRACK_H
#define XTRACK_TRACK_H

#include "xobjects/headers/common.h"
#include "xobjects/headers/atomicadd.h"
#include "xtrack/headers/constants.h"

// Per-coordinate number type: double for normal tracking, overridable (e.g. to
// a TPSA type) by a translation unit that wants to track a non-scalar particle.
#ifndef XT_NUM
#define XT_NUM double
#endif

// Pass a coordinate as a read-only function argument. Native tracking is
// compiled as C, where call-by-value `const XT_NUM` (a double) is correct. A non-scalar
// XT_NUM (e.g. a TPSA whose copy-constructor only copies the descriptor) must be passed by const
// reference to avoid losing its value on copy. The C++ translation unit overrides this to `const XT_NUM&`.
#ifndef XT_NUM_CONST_ARG
#define XT_NUM_CONST_ARG const XT_NUM
#endif

// Per-strength number type: double for normal tracking, overridable (to a TPSA type)
// so lattice knobs can be TPSA parameters. Identity for double.
// A non-scalar XT_STRENGTH is passed by const-reference (copy-constructor caveat, as XT_NUM).
#ifndef XT_STRENGTH
#define XT_STRENGTH double
#endif
#ifndef XT_STRENGTH_CONST_ARG
#define XT_STRENGTH_CONST_ARG const XT_STRENGTH
#endif

// Strength as a mutable by-value parameter, which tapering scales in place. A non-scalar
// XT_STRENGTH cannot be copied by value, so there it becomes a const reference and
// tapering is unavailable.
#ifndef XT_STRENGTH_ARG
#define XT_STRENGTH_ARG XT_STRENGTH
#endif

// Const part of a strength, for the paths that stay double (the magnet edges).
#ifndef XT_STRENGTH_CONST
#define XT_STRENGTH_CONST(v) (v)
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
