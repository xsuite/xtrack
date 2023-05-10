// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_MONITORS_H
#define XTRACK_MONITORS_H

static inline
void LocalParticle_to_Particles_Selected(
                                        LocalParticle* source,
                                        ParticlesData dest,
                                        int64_t id,
                                        int64_t set_scalar){
// if (set_scalar){
//   ParticlesData_set__capacity(dest,      LocalParticle_get__capacity(source));
//   ParticlesData_set__num_active_particles(dest,      LocalParticle_get__num_active_particles(source));
//   ParticlesData_set__num_lost_particles(dest,      LocalParticle_get__num_lost_particles(source));
//   ParticlesData_set_start_tracking_at_element(dest,      LocalParticle_get_start_tracking_at_element(source));
//   ParticlesData_set_q0(dest,      LocalParticle_get_q0(source));
//   ParticlesData_set_mass0(dest,      LocalParticle_get_mass0(source));
// }
//   ParticlesData_set_p0c(dest, id,       LocalParticle_get_p0c(source));
//   ParticlesData_set_gamma0(dest, id,       LocalParticle_get_gamma0(source));
//   ParticlesData_set_beta0(dest, id,       LocalParticle_get_beta0(source));
  ParticlesData_set_s(dest, id,       LocalParticle_get_s(source));
  ParticlesData_set_zeta(dest, id,       LocalParticle_get_zeta(source));
  ParticlesData_set_ptau(dest, id,       LocalParticle_get_ptau(source));
  ParticlesData_set_delta(dest, id,       LocalParticle_get_delta(source));
//   ParticlesData_set_rpp(dest, id,       LocalParticle_get_rpp(source));
//   ParticlesData_set_rvv(dest, id,       LocalParticle_get_rvv(source));
//   ParticlesData_set_chi(dest, id,       LocalParticle_get_chi(source));
//   ParticlesData_set_charge_ratio(dest, id,       LocalParticle_get_charge_ratio(source));
//   ParticlesData_set_weight(dest, id,       LocalParticle_get_weight(source));
//   ParticlesData_set_particle_id(dest, id,       LocalParticle_get_particle_id(source));
//   ParticlesData_set_at_element(dest, id,       LocalParticle_get_at_element(source));
//   ParticlesData_set_at_turn(dest, id,       LocalParticle_get_at_turn(source));
  ParticlesData_set_state(dest, id,       LocalParticle_get_state(source));
//   ParticlesData_set_parent_particle_id(dest, id,       LocalParticle_get_parent_particle_id(source));
//   ParticlesData_set__rng_s1(dest, id,       LocalParticle_get__rng_s1(source));
//   ParticlesData_set__rng_s2(dest, id,       LocalParticle_get__rng_s2(source));
//   ParticlesData_set__rng_s3(dest, id,       LocalParticle_get__rng_s3(source));
//   ParticlesData_set__rng_s4(dest, id,       LocalParticle_get__rng_s4(source));
  ParticlesData_set_x(dest, id,       LocalParticle_get_x(source));
  ParticlesData_set_y(dest, id,       LocalParticle_get_y(source));
  ParticlesData_set_px(dest, id,       LocalParticle_get_px(source));
  ParticlesData_set_py(dest, id,       LocalParticle_get_py(source));
}


/*gpufun*/
void ParticlesMonitor_track_local_particle(ParticlesMonitorData el,
                       LocalParticle* part0){

    #ifndef XTRACK_DISABLE_MONITOR
    printf("ParticlesMonitor_track_local_particle\n");

    int64_t const start_at_turn = ParticlesMonitorData_get_start_at_turn(el);
    int64_t const stop_at_turn = ParticlesMonitorData_get_stop_at_turn(el);
    int64_t const part_id_start = ParticlesMonitorData_get_part_id_start(el);
    int64_t const part_id_end= ParticlesMonitorData_get_part_id_end(el);
    int64_t const ebe_mode = ParticlesMonitorData_get_ebe_mode(el);
    int64_t const n_repetitions = ParticlesMonitorData_get_n_repetitions(el);
    int64_t const repetition_period = ParticlesMonitorData_get_repetition_period(el);
    ParticlesData data = ParticlesMonitorData_getp_data(el);

    int64_t n_turns_record = stop_at_turn - start_at_turn;

    //start_per_particle_block (part0->part)
    int64_t at_turn;
    if (ebe_mode){
        at_turn = LocalParticle_get_at_element(part);
    }
    else{
        at_turn = LocalParticle_get_at_turn(part);
    }
    if (n_repetitions == 1){
        if (at_turn>=start_at_turn && at_turn<stop_at_turn){
            int64_t const particle_id = LocalParticle_get_particle_id(part);
            if (particle_id<part_id_end && particle_id>=part_id_start){
                int64_t const store_at =
                    n_turns_record * (particle_id - part_id_start)
                    + at_turn - start_at_turn;
                LocalParticle_to_Particles_Selected(part, data, store_at, 0);
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



    //end_per_particle_block

    #endif //XTRACK_DISABLE_MONITOR
}

#endif
