// ##################################
// BPM 
// 
// Author: Rahul Singh
// Date: 2023-06-10
// ##################################

#ifndef XTRACK_BPM_H
#define XTRACK_BPM_H

#if !defined( C_LIGHT )
    #define   C_LIGHT ( 299792458.0 )
#endif /* !defined( C_LIGHT ) */

/*gpufun*/
void BPM_track_local_particle(BPMData el, LocalParticle* part0){

    int64_t const start_at_turn = BPMData_get_start_at_turn(el);
    int64_t const stop_at_turn = BPMData_get_stop_at_turn(el);
    int64_t const index_in_turn = BPMData_get_samples_per_turn(el);
    int64_t const sampling_frequency = BPMData_get_sampling_frequency(el);
    BPMRecord record = BPMData_getp_data(el);
    
    int64_t const at_turn = LocalParticle_get_at_turn(part0);
    int64_t last_particle_id = 0;
    
    double x_cen = 0;
    double y_cen = 0;
    double active_parts = 0;


      //printf("Global ID is %d \n",get_global_id(0));
	//https://man.opencl.org/workItemFunctions.html
	
	//start_per_particle_block(part0->part)
    if (at_turn >= start_at_turn && at_turn <= stop_at_turn){
	double const zeta = LocalParticle_get_zeta(part);
	//printf("zeta is %f \n",zeta);
        double const beta0 = LocalParticle_get_beta0(part);
	last_particle_id = LocalParticle_get_particle_id(part);
        
	// Mapping zeta to a unique sample index within the turn
	//
        double abs_index =  (sampling_frequency)*(zeta / beta0 / C_LIGHT );
	//printf("Absolute index is %f of particle %ld \n",abs_index,particle_id);
	int64_t i = abs_index;
	if (i <0) {
		i = index_in_turn +i;
	}	
	if (abs_index/(index_in_turn) >= 1){
	i = i % index_in_turn;
	}
	if (abs_index/index_in_turn <= -1){
	i = index_in_turn + (i % index_in_turn);
	}

	// Perform the sum of positions at each sample within turn
	
                x_cen = LocalParticle_get_x(part)+BPMRecord_get_x_cen_index(record,i);
		BPMRecord_set_x_cen_index(record,i,x_cen);
                y_cen = LocalParticle_get_y(part)+BPMRecord_get_y_cen_index(record,i);
		BPMRecord_set_y_cen_index(record,i,y_cen);
                active_parts = 1+BPMRecord_get_summed_particles_index(record,i);
                BPMRecord_set_summed_particles_index(record,i,active_parts);
    }
    //end_per_particle_block
    
         int64_t slot = at_turn-start_at_turn;
         BPMRecord_set_at_turn(record, slot, at_turn);
         BPMRecord_set_last_particle_id(record,slot,last_particle_id);

	 // Concatenate the samples within a turn into a BPM time series 
	 // 
    for (int64_t j=0;j<index_in_turn;j++){
	    x_cen=BPMRecord_get_x_cen_index(record,j)/BPMRecord_get_summed_particles_index(record,j);
	    y_cen=BPMRecord_get_y_cen_index(record,j)/BPMRecord_get_summed_particles_index(record,j);
            BPMRecord_set_x_cen_index(record,j,0);
            BPMRecord_set_y_cen_index(record,j,0);
    	    int64_t i_slot = ((at_turn-start_at_turn)*index_in_turn)+j; 
    	// index_in_turn*rev_frequency = sampling_frequency 
        //printf("Turn number is %ld and particles at index %ld is %ld\n",at_turn,j,BPMRecord_get_summed_particles_index(record,j));
        if(i_slot >= 0){
            BPMRecord_set_x_cen(record, i_slot, x_cen);
            BPMRecord_set_y_cen(record, i_slot, y_cen);
	    BPMRecord_set_summed_particles(record,i_slot,BPMRecord_get_summed_particles_index(record,j));
	    BPMRecord_set_summed_particles_index(record,j,0);

        }
    }

} 
#endif
