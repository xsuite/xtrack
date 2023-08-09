// ##################################
// BPM
// 
// Author: Rahul Singh
// Date: 2023-06-10
// ##################################


#ifndef XTRACK_BEAMMONITOR_H
#define XTRACK_BEAMMONITOR_H

#if !defined( C_LIGHT )
    #define   C_LIGHT ( 299792458.0 )
#endif /* !defined( C_LIGHT ) */

/*gpufun*/
void BeamMonitor_track_local_particle(BeamMonitorData el, LocalParticle* part0){

    int64_t const start_at_turn = BeamMonitorData_get_start_at_turn(el);
    double const frev = BeamMonitorData_get_rev_frequency(el);
    double const sampling_frequency = BeamMonitorData_get_sampling_frequency(el);
    BeamMonitorRecord record = BeamMonitorData_getp_data(el);
    int64_t last_particle_id = 0;
    double x_sum = 0;
    double y_sum = 0;
    int64_t active_parts = 0;

      // This function shows errnous results with GPUs due to concurrent memory access
        //start_per_particle_block(part0->part)

    int64_t const at_turn = LocalParticle_get_at_turn(part);

    if(at_turn >= start_at_turn){
            double const zeta  = LocalParticle_get_zeta(part);
            double const beta0 = LocalParticle_get_beta0(part);

            last_particle_id = LocalParticle_get_particle_id(part);
            // Mapping zeta of all particles at each "reference particle turn" to a unique sample index in time

            int64_t i = roundf((sampling_frequency)*(-1*zeta / beta0 / C_LIGHT ));
	    int64_t slot = roundf(sampling_frequency*(at_turn-start_at_turn)/frev+i);

            // Sanity check, e.g. if the mapping from "turn of the reference particle" to the time slot makes sense
            // if (slot < 0){
            // printf("Warning: Perhaps reduce the zeta distribution width, particle cannot be at the previous turn in the first turn." 
            // " Value of time slot is %ld for this turn %ld \n",slot,at_turn-start_at_turn);
             // ignore these particles
            // }
               if (slot > (BeamMonitorRecord_len_x_cen(record))) //momentum spread of 1% can have outliers upto 5% 
            {
                    printf("Warning: Particle with ID %ld is much ahead of reference particle such that it crosses the allocated buffer,"
                 " increase safety margin of memory allocation for the samples data \n",last_particle_id);
            }

            if (slot >= 0 && slot <= BeamMonitorRecord_len_x_cen(record)){
	        BeamMonitorRecord record = BeamMonitorData_getp_data(el);
 
	        x_sum = LocalParticle_get_x(part)+BeamMonitorRecord_get_x_cen(record,slot);
	        y_sum = LocalParticle_get_y(part)+BeamMonitorRecord_get_y_cen(record,slot);

	        BeamMonitorRecord_set_x_cen(record,slot,x_sum);
	        BeamMonitorRecord_set_y_cen(record,slot,y_sum);

	        active_parts = 1+BeamMonitorRecord_get_summed_particles(record,slot);
	        BeamMonitorRecord_set_summed_particles(record,slot,active_parts);

// DEBUGGING 
//printf("Particle number %ld at turn %ld and zeta %f, absolute index %ld and active partciles %ld \n", last_particle_id,slot,zeta,i,active_parts);

	    }
            BeamMonitorRecord_set_at_turn(record, at_turn-start_at_turn, at_turn);
            BeamMonitorRecord_set_last_particle_id(record,at_turn-start_at_turn,last_particle_id);
    }    
	//end_per_particle_block
} 
#endif

