#ifndef XTRACK_MONITORS_H
#define XTRACK_MONITORS_H

/*gpufun*/
void ParticlesMonitor_track_local_particle(ParticlesMonitorData el, 
					   LocalParticle* part){

    int64_t const start_at_turn = ParticlesMonitorData_get_start_at_turn(el);
    int64_t const stop_at_turn = ParticlesMonitorData_get_stop_at_turn(el);
    int64_t const n_records = ParticlesMonitorData_get_n_records(el);
    ParticlesData data = ParticlesMonitorData_getp_data(el);

    int64_t n_turns_record = stop_at_turn - start_at_turn;

    int64_t const n_part = LocalParticle_get_num_particles(part); //only_for_context cpu_serial cpu_openmp
    for (int ii=0; ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
	part->ipart = ii;            //only_for_context cpu_serial cpu_openmp

	int64_t const at_turn = LocalParticle_get_at_turn(part);
	if (at_turn>=start_at_turn && start_at_turn<stop_at_turn){
	    int64_t const particle_id = LocalParticle_get_particle_id(part);
	    int64_t const store_at = n_turns_record*particle_id 
		    + at_turn - start_at_turn;

	    if (store_at>=0 && store_at<n_records){ //avoid memory leak in case of 
		                                    //invalid particle_id or at_turn
	        LocalParticle_to_Particles(part, data, store_at, 0);
	    }
	}

    } //only_for_context cpu_serial cpu_openmp

}

#endif
