// ##################################
// Beam Monitor
// 
// Author: Rahul Singh, Philipp Niedermayer
// Date: 2023-06-10
// ##################################


#ifndef XTRACK_BEAM_POSITION_MONITOR_H
#define XTRACK_BEAM_POSITION_MONITOR_H

#if !defined( C_LIGHT )
    #define   C_LIGHT ( 299792458.0 )
#endif /* !defined( C_LIGHT ) */

/*gpufun*/
void BeamPositionMonitor_track_local_particle(BeamPositionMonitorData el, LocalParticle* part0){

    // get parameters
    int64_t const start_at_turn = BeamPositionMonitorData_get_start_at_turn(el);
    double const frev = BeamPositionMonitorData_get_frev(el);
    double const sampling_frequency = BeamPositionMonitorData_get_sampling_frequency(el);
    BeamPositionMonitorRecord record = BeamPositionMonitorData_getp_data(el);


    //start_per_particle_block(part0->part)

        // zeta is the absolute path length deviation from the reference particle: zeta = (s - beta0*c*t)
        // but without limits, i.e. it can exceed the circumference (for coasting beams)
        // as the particle falls behind or overtakes the reference particle
        double const zeta = LocalParticle_get_zeta(part);
        double const at_turn = LocalParticle_get_at_turn(part);
        double const beta0 = LocalParticle_get_beta0(part);

        // compute sample index
        int64_t slot = roundf(sampling_frequency * ( (at_turn-start_at_turn)/frev - zeta/beta0/C_LIGHT ));
        int64_t max_slot = BeamPositionMonitorRecord_len_count(record);

        if (slot >= 0 && slot < max_slot){
	        BeamPositionMonitorRecord record = BeamPositionMonitorData_getp_data(el);
 
            // TODO: this shows errnous results with GPUs due to concurrent memory access
            int64_t count = 1 + BeamPositionMonitorRecord_get_count(record, slot);
            BeamPositionMonitorRecord_set_count(record, slot, count);
	        double x_sum = LocalParticle_get_x(part) + BeamPositionMonitorRecord_get_x_sum(record, slot);
	        BeamPositionMonitorRecord_set_x_sum(record, slot, x_sum);
	        double y_sum = LocalParticle_get_y(part) + BeamPositionMonitorRecord_get_y_sum(record, slot);
	        BeamPositionMonitorRecord_set_y_sum(record, slot, y_sum);
	        
	    }

	//end_per_particle_block

}

#endif

