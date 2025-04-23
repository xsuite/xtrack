// ##################################
// Beam Size Monitor
// 
// Author: Philipp Niedermayer
// Date: 2023-08-14
// ##################################


#ifndef XTRACK_BEAM_SIZE_MONITOR_H
#define XTRACK_BEAM_SIZE_MONITOR_H

#include <headers/track.h>


GPUFUN
void BeamSizeMonitor_track_local_particle(BeamSizeMonitorData el, LocalParticle* part0){

    // get parameters
    int64_t const start_at_turn = BeamSizeMonitorData_get_start_at_turn(el);
    int64_t particle_id_start = BeamSizeMonitorData_get_particle_id_start(el);
    int64_t particle_id_stop = particle_id_start + BeamSizeMonitorData_get_num_particles(el);
    double const frev = BeamSizeMonitorData_get_frev(el);
    double const sampling_frequency = BeamSizeMonitorData_get_sampling_frequency(el);

    BeamSizeMonitorRecord record = BeamSizeMonitorData_getp_data(el);

    int64_t max_slot = BeamSizeMonitorRecord_len_count(record);

    START_PER_PARTICLE_BLOCK(part0, part);
        int64_t particle_id = LocalParticle_get_particle_id(part);
        if (particle_id_stop < 0 || (particle_id_start <= particle_id && particle_id < particle_id_stop))
        {
            // zeta is the absolute path length deviation from the reference particle: zeta = (s - beta0*c*t)
            // but without limits, i.e. it can exceed the circumference (for coasting beams)
            // as the particle falls behind or overtakes the reference particle
            double const zeta = LocalParticle_get_zeta(part);
            double const at_turn = LocalParticle_get_at_turn(part);
            double const beta0 = LocalParticle_get_beta0(part);

            // compute sample index
            int64_t slot = round(sampling_frequency * ( (at_turn-start_at_turn)/frev - zeta/beta0/C_LIGHT ));

            if (slot >= 0 && slot < max_slot){
                double x = LocalParticle_get_x(part);
                double y = LocalParticle_get_y(part);

                /*gpuglmem*/ double* count = BeamSizeMonitorRecord_getp1_count(record, slot);
                atomicAdd(count, 1);

                /*gpuglmem*/ double * x_sum = BeamSizeMonitorRecord_getp1_x_sum(record, slot);
                atomicAdd(x_sum, x);

                /*gpuglmem*/ double * y_sum = BeamSizeMonitorRecord_getp1_y_sum(record, slot);
                atomicAdd(y_sum, y);

                /*gpuglmem*/ double * x2_sum = BeamSizeMonitorRecord_getp1_x2_sum(record, slot);
                atomicAdd(x2_sum, x*x);

                /*gpuglmem*/ double * y2_sum = BeamSizeMonitorRecord_getp1_y2_sum(record, slot);
                atomicAdd(y2_sum, y*y);
            }
        }
    END_PER_PARTICLE_BLOCK;
}

#endif

