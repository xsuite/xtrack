/*
 * Template for scalar kernels that invoke a LocalParticle function once per active particle.
 * Include this file with ELEMENT_NAME, KERNEL_NAME, LOCAL_PARTICLE_FUNCTION,
 * KERNEL_EXTRA_ARGUMENTS, and KERNEL_EXTRA_ARGUMENT_VALUES defined. It deliberately has no
 * include guard.
 */

#ifndef ELEMENT_NAME
    #error "per_particle_kernel.h requires ELEMENT_NAME"
#endif

#ifndef KERNEL_NAME
    #error "per_particle_kernel.h requires KERNEL_NAME"
#endif

#ifndef LOCAL_PARTICLE_FUNCTION
    #error "per_particle_kernel.h requires LOCAL_PARTICLE_FUNCTION"
#endif

#ifndef KERNEL_EXTRA_ARGUMENTS
    #error "per_particle_kernel.h requires KERNEL_EXTRA_ARGUMENTS"
#endif

#ifndef KERNEL_EXTRA_ARGUMENT_VALUES
    #error "per_particle_kernel.h requires KERNEL_EXTRA_ARGUMENT_VALUES"
#endif

#ifndef XTRACK_TPSA_TRACK
/* Auxiliary per-particle kernels are only supported with scalar ParticlesData. */
#include "xtrack/particles/headers/local_particle.h"

#define ELEMENT_DATA CONCAT(ELEMENT_NAME, Data)

GPUKERN void KERNEL_NAME(
        ELEMENT_DATA el,
        ParticlesData particles
        KERNEL_EXTRA_ARGUMENTS,
        int64_t flag_increment_at_element,
        GPUGLMEM int8_t* io_buffer) {

    #ifdef XO_CONTEXT_CPU_OPENMP
        const int64_t capacity = ParticlesData_get__capacity(particles);
        const int num_threads = omp_get_max_threads();

        #ifndef XT_OMP_SKIP_REORGANIZE
            const int64_t num_particles_to_track =
                ParticlesData_get__num_active_particles(particles);

            {
                LocalParticle lpart;
                lpart.io_buffer = io_buffer;
                Particles_to_LocalParticle(particles, &lpart, 0, capacity);
                check_is_active(&lpart);
                count_reorganized_particles(&lpart);
                LocalParticle_to_Particles(&lpart, particles, 0, capacity);
            }
        #else
            const int64_t num_particles_to_track = capacity;
        #endif

        const int64_t chunk_size = (num_particles_to_track + num_threads - 1) / num_threads;

    #pragma omp parallel for
    for (int64_t batch_id = 0; batch_id < num_threads; batch_id++) {
        LocalParticle lpart;
        lpart.io_buffer = io_buffer;
        lpart.track_flags = 0;
        int64_t part_id = batch_id * chunk_size;
        int64_t end_id = (batch_id + 1) * chunk_size;
        if (end_id > num_particles_to_track) end_id = num_particles_to_track;

        if (part_id < capacity) {
            Particles_to_LocalParticle(particles, &lpart, part_id, end_id);
            if (check_is_active(&lpart) > 0) {
                LOCAL_PARTICLE_FUNCTION(el, &lpart KERNEL_EXTRA_ARGUMENT_VALUES);
            }
            if (check_is_active(&lpart) > 0 && flag_increment_at_element) {
                increment_at_element(&lpart, 1);
            }
        }
    }

    // On OpenMP we want to additionally by default reorganize all the particles.
    #ifndef XT_OMP_SKIP_REORGANIZE
        LocalParticle lpart;
        lpart.io_buffer = io_buffer;
        Particles_to_LocalParticle(particles, &lpart, 0, capacity);
        check_is_active(&lpart);
    #endif
    #else
        LocalParticle lpart;
        lpart.io_buffer = io_buffer;

        #if defined(XO_CONTEXT_CUDA)
            const int64_t part_id = blockDim.x * blockIdx.x + threadIdx.x;
        #elif defined(XO_CONTEXT_CL)
            const int64_t part_id = get_global_id(0);
        #else
            const int64_t part_id = 0;
        #endif

        const int64_t capacity = ParticlesData_get__capacity(particles);
        if (part_id < capacity) {
            Particles_to_LocalParticle(particles, &lpart, part_id, 0);
            if (check_is_active(&lpart) > 0) {
                LOCAL_PARTICLE_FUNCTION(el, &lpart KERNEL_EXTRA_ARGUMENT_VALUES);
            }
            if (check_is_active(&lpart) > 0 && flag_increment_at_element) {
                increment_at_element(&lpart, 1);
            }
        }
    #endif
}

#undef ELEMENT_DATA
#endif /* XTRACK_TPSA_TRACK */
