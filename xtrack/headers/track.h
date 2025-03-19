#ifndef XTRACK_TRACK_H
#define XTRACK_TRACK_H

/*
    The particle tracking "decorators" for all the contexts.
*/

#ifdef XO_CONTEXT_CPU_SERIAL
    // We are on CPU, without OpenMP

    #define START_PER_PARTICLE_BLOCK(SRC_PART, DEST_PART) { \
            const int64_t XT_part_block_start_idx = 0; \
            const int64_t XT_part_block_end_idx = LocalParticle_get__num_active_particles((SRC_PART)); \
            \
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
            const int64_t XT_part_block_start_idx = (SRC_PART)->ipart; \
            const int64_t XT_part_block_end_idx = (SRC_PART)->endpart; \
            \
            \ // #pragma omp simd // TODO: currently does not work, needs investigating
            for (int64_t XT_part_block_ii = XT_part_block_start_idx; XT_part_block_ii<XT_part_block_end_idx; XT_part_block_ii++) \
            { \
                LocalParticle lpart = *(SRC_PART); \
                LocalParticle* DEST_PART = &lpart; \
                part->ipart = XT_part_block_ii; \
                \
                if (LocalParticle_get_state(DEST_PART) > 0) {

    #define END_PER_PARTICLE_BLOCK \
                } \
            } \
        }
#endif  // XO_CONTEXT_CPU_OPENMP


#if defined(XO_CONTEXT_CUDA) || defined(XO_CONTEXT_CL)
    // We are on GPU

        #define START_PER_PARTICLE_BLOCK(SRC_PART, DEST_PART) { \
                LocalParticle* DEST_PART = (SRC_PART);

    #define END_PER_PARTICLE_BLOCK \
            }
#endif  // XO_CONTEXT_CUDA || XO_CONTEXT_CL

#endif  // XTRACK_TRACK_H
