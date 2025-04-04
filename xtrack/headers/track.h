#ifndef XTRACK_TRACK_H
#define XTRACK_TRACK_H

#include <headers/constants.h>

/*
    The particle tracking "decorators" for all the contexts.
*/

#ifdef XO_CONTEXT_CPU_SERIAL
    // We are on CPU, without OpenMP

    #define PER_PARTICLE_BLOCK(SRC_PART, DEST_PART, CODE) { \
            const int64_t XT_part_block_start_idx = 0; \
            const int64_t XT_part_block_end_idx = LocalParticle_get__num_active_particles((SRC_PART)); \
            for (int64_t XT_part_block_ii = XT_part_block_start_idx; XT_part_block_ii<XT_part_block_end_idx; XT_part_block_ii++) \
            { \
                LocalParticle lpart = *(SRC_PART); \
                LocalParticle* DEST_PART = &lpart; \
                part->ipart = XT_part_block_ii; \
                { CODE; } \
            } \
        }
#endif  // XO_CONTEXT_CPU_SERIAL

#ifdef XO_CONTEXT_CPU_OPENMP
    // We are on CPU with the OpenMP context switched on

    #define PER_PARTICLE_BLOCK(SRC_PART, DEST_PART, CODE) { \
            const int64_t XT_part_block_start_idx = (SRC_PART)->ipart; \
            const int64_t XT_part_block_end_idx = (SRC_PART)->endpart; \
            for (int64_t XT_part_block_ii = XT_part_block_start_idx; XT_part_block_ii<XT_part_block_end_idx; XT_part_block_ii++) \
            { \
                LocalParticle lpart = *(SRC_PART); \
                LocalParticle* DEST_PART = &lpart; \
                part->ipart = XT_part_block_ii; \
                \
                if (LocalParticle_get_state(DEST_PART) > 0) { \
                    CODE ; \
                } \
            } \
        }
#endif  // XO_CONTEXT_CPU_OPENMP


#if defined(XO_CONTEXT_CUDA) || defined(XO_CONTEXT_CL)
    // We are on GPU

        #define PER_PARTICLE_BLOCK(SRC_PART, DEST_PART, CODE) { \
                LocalParticle* DEST_PART = (SRC_PART); \
                CODE \
            }
#endif  // XO_CONTEXT_CUDA || XO_CONTEXT_CL


/*
    Qualifier keywords for GPU and optimisation
*/

#ifdef XO_CONTEXT_CPU // for both serial and OpenMP
    #define GPUKERN
    #define GPUFUN      static inline
    #define GPUGLMEM
    #define RESTRICT    restrict
#endif


#ifdef XO_CONTEXT_CUDA
    #define GPUKERN     __global__
    #define GPUFUN      __device__
    #define GPUGLMEM
    #define RESTRICT
#endif // XO_CONTEXT_CUDA


#ifdef XO_CONTEXT_CL
    #define GPUKERN     __kernel
    #define GPUFUN
    #define GPUGLMEM    __global
    #define RESTRICT
#endif // XO_CONTEXT_CL


/*
    Common maths-related macros
*/

#define POW2(X) ((X)*(X))
#define POW3(X) ((X)*(X)*(X))
#define POW4(X) ((X)*(X)*(X)*(X))
#define NONZERO(X) ((X) != 0.0)


#ifndef PER_PARTICLE_BLOCK
#error "Unknown context, or the expected context (XO_CONTEXT_*) flag undefined. Try updating Xobjects?"
#endif

#endif  // XTRACK_TRACK_H
