#ifndef XFIELDS_COPRESSED_PROFILE_H
#define XFIELDS_COPRESSED_PROFILE_H

#include "xtrack/headers/track.h"


GPUFUN
void CompressedProfile_track_local_particle(CompressedProfileData el,
        LocalParticle *part0) {

}

GPUFUN
void CompressedProfile_interp_result(
    CompressedProfileData el, LocalParticle *part0,
    int64_t data_shape_0,
    int64_t data_shape_1,
    int64_t data_shape_2,
    GPUGLMEM double* data,
    GPUGLMEM int64_t* i_slot_particles,
    GPUGLMEM int64_t* i_slice_particles,
    GPUGLMEM double* out
) {
	int64_t const _N_T = CompressedProfileData_get__N_T(el);
	int64_t const _first_target_slot = CompressedProfileData_get__first_target_slot(el);
	int64_t const _N_aux = CompressedProfileData_get__N_aux(el);
	int64_t const num_turns = CompressedProfileData_get_num_turns(el);

	START_PER_PARTICLE_BLOCK(part0, part);
        const int64_t ipart = part->ipart;
        const int64_t i_bunch = i_slot_particles[ipart];
        const int64_t i_slice = i_slice_particles[ipart];

        if (i_slice >= 0) {

            const int64_t i_start_in_moments_data = (_N_T+_first_target_slot - i_bunch - 1) * _N_aux;

            double rr = 0;
            for(int i_turn=0; i_turn<num_turns; i_turn++) {
                rr = rr + data[
                         i_start_in_moments_data + i_slice +
                         data_shape_2 * (i_turn + data_shape_1*(data_shape_0-1))];
            }
            out[ipart] = rr;

        }
	END_PER_PARTICLE_BLOCK;
}

#endif // XFIELDS_COPRESSED_PROFILE_H
