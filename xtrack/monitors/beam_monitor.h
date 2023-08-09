// ##################################
// Beam Monitor
// 
// Author: Rahul Singh, Philipp Niedermayer
// Date: 2023-06-10
// ##################################


#ifndef XTRACK_BEAMMONITOR_H
#define XTRACK_BEAMMONITOR_H

#if !defined( C_LIGHT )
    #define   C_LIGHT ( 299792458.0 )
#endif /* !defined( C_LIGHT ) */

/*gpufun*/
void BeamMonitor_track_local_particle(BeamMonitorData el, LocalParticle* part0){

    // get parameters
    int64_t const start_at_turn = BeamMonitorData_get_start_at_turn(el);
    double const frev = BeamMonitorData_get_frev(el);
    double const sampling_frequency = BeamMonitorData_get_sampling_frequency(el);
    BeamMonitorRecord record = BeamMonitorData_getp_data(el);


    //start_per_particle_block(part0->part)

        // zeta is the absolute path length deviation from the reference particle: zeta = (s - beta0*c*t)
        // but without limits, i.e. it can exceed the circumference (for coasting beams)
        // as the particle falls behind or overtakes the reference particle
        double const zeta = LocalParticle_get_zeta(part);
        double const at_turn = LocalParticle_get_at_turn(part);
        double const beta0 = LocalParticle_get_beta0(part);

        // compute sample index
        int64_t slot = roundf(sampling_frequency * ( (at_turn-start_at_turn)/frev - zeta/beta0/C_LIGHT ));
        int64_t max_slot = BeamMonitorRecord_len_count(record);

        if (slot >= 0 && slot < max_slot){
	        BeamMonitorRecord record = BeamMonitorData_getp_data(el);
 
            // TODO: this shows errnous results with GPUs due to concurrent memory access
            int64_t count = 1 + BeamMonitorRecord_get_count(record, slot);
            BeamMonitorRecord_set_count(record, slot, count);
	        double x_sum = LocalParticle_get_x(part) + BeamMonitorRecord_get_x_sum(record, slot);
	        BeamMonitorRecord_set_x_sum(record, slot, x_sum);
	        double y_sum = LocalParticle_get_y(part) + BeamMonitorRecord_get_y_sum(record, slot);
	        BeamMonitorRecord_set_y_sum(record, slot, y_sum);
	        
	    }

	//end_per_particle_block

}

#endif

