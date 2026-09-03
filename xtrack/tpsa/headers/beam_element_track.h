/*
 * Template for TPSA BeamElement.track kernels. Include this file with
 * ELEMENT_NAME and KERNEL_NAME defined; it deliberately has no include guard.
 */

#ifndef ELEMENT_NAME
    #error "beam_element_track.h requires ELEMENT_NAME"
#endif

#ifndef KERNEL_NAME
    #error "beam_element_track.h requires KERNEL_NAME"
#endif

#include "xtrack/particles/headers/local_particle.h"

#define ELEMENT_DATA CONCAT(ELEMENT_NAME, Data)

/* TPSA tracking operates on one map, so this is a serial one-particle wrapper. */
void KERNEL_NAME(
        ELEMENT_DATA el,
        TpsaParticleData particles,
        int64_t flag_increment_at_element,
        int8_t* io_buffer) {
    LocalParticle lpart;
    lpart.io_buffer = io_buffer;
    lpart.track_flags = 0;
    lpart.line_length = 0.0;

    Particles_to_LocalParticle(particles, &lpart, 0, 1);
    lpart._num_active_particles = 1;
    lpart._num_lost_particles = 0;
    xt_tpsa::default_scope xt_tpsa_default_scope(lpart.x);
    xt_tpsa::error_scope xt_tpsa_error_scope;

    try {
        LocalParticle_set_state(&lpart, 1);
        LocalParticle_set_at_element(&lpart, 0);
        LocalParticle_update_delta(&lpart, LocalParticle_get_delta(&lpart));
        LocalParticle_set_s(&lpart, 0.0);
        LocalParticle_set_ax(&lpart, 0.0);
        LocalParticle_set_ay(&lpart, 0.0);

        if (check_is_active(&lpart) > 0) {
            CONCAT(ELEMENT_NAME, _track_local_particle_with_transformations)(
                el, &lpart);
        }
        if (check_is_active(&lpart) > 0 && flag_increment_at_element) {
            increment_at_element(&lpart, 1);
        }
    } catch (const xt_tpsa::tracking_error&) {
        LocalParticle_set_state(&lpart, XT_LOST_ON_TPSA_ERROR);
    }

    LocalParticle_to_Particles(&lpart, particles, 0, 1);
}

#undef ELEMENT_DATA
