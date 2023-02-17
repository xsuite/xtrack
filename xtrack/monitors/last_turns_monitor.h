// ##################################
// Monitor to save particle state
// for last N turns before loss
// 
// Author: Philipp Niedermayer
// Date: 2023-01-19
// ##################################

#ifndef XTRACK_LAST_TURNS_MONITOR_H
#define XTRACK_LAST_TURNS_MONITOR_H

/*gpufun*/
void LastTurnsMonitor_track_local_particle(LastTurnsMonitorData el, LocalParticle* part0){
    int64_t n_last_turns = LastTurnsMonitorData_get_n_last_turns(el);
    int64_t particle_id_start = LastTurnsMonitorData_get_particle_id_start(el);
    int64_t particle_id_stop = particle_id_start + LastTurnsMonitorData_get_num_particles(el);
    int64_t every_n_turns = LastTurnsMonitorData_get_every_n_turns(el);
    LastTurnsData data = LastTurnsMonitorData_getp_data(el);

    //start_per_particle_block (part0->part)
    
        int64_t particle_id = LocalParticle_get_particle_id(part);
        int64_t at_turn = LocalParticle_get_at_turn(part);

        // When the particle is lost, tracking is stopped automatically for it.
        // Therefore we don't need to check for particle state here.
        // But this also means, that he have to save the buffer offset each time
        // as long as the particle is still alive, since it could be the last
        // time we see it.
    
        if (at_turn >= 0 && at_turn%every_n_turns == 0 && particle_id_start <= particle_id && particle_id < particle_id_stop){

            // determine slot in rolling buffer
            int64_t offset = (at_turn / every_n_turns) % n_last_turns;
            int64_t ip = particle_id - particle_id_start;
            int64_t slot = n_last_turns * ip + offset;

            // store buffer offset
            LastTurnsData_set_lost_at_offset(data, ip, offset);

            // store data
            LastTurnsData_set_particle_id(data, slot, particle_id);
            LastTurnsData_set_at_turn(data, slot, at_turn);
            LastTurnsData_set_x(data, slot, LocalParticle_get_x(part));
            LastTurnsData_set_px(data, slot, LocalParticle_get_px(part));
            LastTurnsData_set_y(data, slot, LocalParticle_get_y(part));
            LastTurnsData_set_py(data, slot, LocalParticle_get_py(part));
            LastTurnsData_set_delta(data, slot, LocalParticle_get_delta(part));
            LastTurnsData_set_zeta(data, slot, LocalParticle_get_zeta(part));
            
        }

    //end_per_particle_block

}

#endif
