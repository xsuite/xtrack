// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_MONITORS_H
#define XTRACK_MONITORS_H

#include <headers/track.h>


GPUFUN
void ParticlesMonitor_track_local_particle(ParticlesMonitorData el,
                       LocalParticle* part0){

    int64_t const start_at_turn = ParticlesMonitorData_get_start_at_turn(el);
    int64_t const stop_at_turn = ParticlesMonitorData_get_stop_at_turn(el);
    int64_t const part_id_start = ParticlesMonitorData_get_part_id_start(el);
    int64_t const part_id_end= ParticlesMonitorData_get_part_id_end(el);
    int64_t const ebe_mode = ParticlesMonitorData_get_ebe_mode(el);
    int64_t const n_repetitions = ParticlesMonitorData_get_n_repetitions(el);
    int64_t const repetition_period = ParticlesMonitorData_get_repetition_period(el);
    ParticlesData data = ParticlesMonitorData_getp_data(el);

    int64_t n_turns_record = stop_at_turn - start_at_turn;

    START_PER_PARTICLE_BLOCK(part0, part);
        int64_t at_turn;
        if (ebe_mode){
            at_turn = LocalParticle_get_at_element(part);
        }
        else{
            #ifdef XSUITE_BACKTRACK
            return; // do not log (only ebe monitor supported for now in backtrack)
            #else
            at_turn = LocalParticle_get_at_turn(part);
            #endif
        }
        if (n_repetitions == 1){
            if (at_turn>=start_at_turn && at_turn<stop_at_turn){
                int64_t const particle_id = LocalParticle_get_particle_id(part);
                if (particle_id<part_id_end && particle_id>=part_id_start){
                    int64_t const store_at =
                        n_turns_record * (particle_id - part_id_start)
                        + at_turn - start_at_turn;
                    LocalParticle_to_Particles(part, data, store_at, 0);
                }
            }
        }
        else if (n_repetitions > 1){
            if (at_turn < start_at_turn){
                return; //only_for_context cuda opencl
                break; //only_for_context cpu_serial cpu_openmp
            }
            int64_t const i_frame = (at_turn - start_at_turn) / repetition_period;
            if (i_frame < n_repetitions
                     && at_turn >= start_at_turn + i_frame*repetition_period
                     && at_turn < stop_at_turn + i_frame*repetition_period
                 ){
                int64_t const particle_id = LocalParticle_get_particle_id(part);
                if (particle_id<part_id_end && particle_id>=part_id_start){
                    int64_t const store_at =
                        n_turns_record * (part_id_end  - part_id_start) * i_frame
                        + n_turns_record * (particle_id - part_id_start)
                        + (at_turn - i_frame * repetition_period) - start_at_turn;
                    LocalParticle_to_Particles(part, data, store_at, 0);
                }
            }
        }

        #ifdef XSUITE_RESTORE_LOSS
        LocalParticle_set_state(part, 1);
        #endif
    END_PER_PARTICLE_BLOCK;
}

#endif


