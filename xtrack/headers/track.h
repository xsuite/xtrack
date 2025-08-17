#ifndef XTRACK_TRACK_H
#define XTRACK_TRACK_H

#include <headers/constants.h>

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

    #define VECTORIZE_OVER(INDEX_NAME, COUNT) \
        for (int INDEX_NAME = 0; INDEX_NAME < (COUNT); INDEX_NAME++) {

    #define END_VECTORIZE \
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

    #define VECTORIZE_OVER(INDEX_NAME, COUNT) \
        _Pragma("omp parallel for") \
        for (int INDEX_NAME = 0; INDEX_NAME < (COUNT); INDEX_NAME++) {

    #define END_VECTORIZE \
        }

#endif  // XO_CONTEXT_CPU_OPENMP


#ifdef XO_CONTEXT_CUDA
    // We are on a CUDA GPU

    #define START_PER_PARTICLE_BLOCK(SRC_PART, DEST_PART) { \
            LocalParticle* DEST_PART = (SRC_PART);

    #define END_PER_PARTICLE_BLOCK \
            }

    #define VECTORIZE_OVER(INDEX_NAME, COUNT) { \
            int INDEX_NAME = blockDim.x * blockIdx.x + threadIdx.x; \
            if (INDEX_NAME < (COUNT)) {

    #define END_VECTORIZE \
            } \
        }
#endif  // XO_CONTEXT_CUDA


#ifdef XO_CONTEXT_CL
    // We are on an OpenCL GPU

    #define START_PER_PARTICLE_BLOCK(SRC_PART, DEST_PART) { \
            LocalParticle* DEST_PART = (SRC_PART);

    #define END_PER_PARTICLE_BLOCK \
            }

    #define VECTORIZE_OVER(INDEX_NAME, COUNT) \
        { \
            int INDEX_NAME = get_global_id(0);

    #define END_VECTORIZE \
        }
#endif  // XO_CONTEXT_CL


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


#ifndef START_PER_PARTICLE_BLOCK
#error "Unknown context, or the expected context (XO_CONTEXT_*) flag undefined. Try updating Xobjects?"
#endif

#endif  // XTRACK_TRACK_H
