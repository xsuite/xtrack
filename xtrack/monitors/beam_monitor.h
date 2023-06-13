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
	
    if (at_turn >= start_at_turn && at_turn <= stop_at_turn){
	//start_per_particle_block(part0->part)
	double const zeta = LocalParticle_get_zeta(part);
	//printf("zeta is %f \n",zeta);
        double const beta0 = LocalParticle_get_beta0(part);
	last_particle_id = LocalParticle_get_particle_id(part);
        
	// Mapping zeta of all particles at each "reference particle turn" to a unique sample index in time
	//
        double abs_index =  (sampling_frequency)*(zeta / beta0 / C_LIGHT );
        int64_t i = (abs_index+0.5);
	if (abs_index <0) {
		i = (abs_index-0.5);
	}	
	//if (abs_index/(index_in_turn) >= 1 || abs_index/index_in_turn <= -1){   // already in next turn or previous turn
	//n = i/index_in_turn;
        //i = i % n;}

	// Perform the sum of positions at each sample within turn
        int64_t start_index = (index_in_turn-1)/2;
	int64_t slot = start_index+(at_turn-start_at_turn)*index_in_turn+i;
           if (slot < 0){
             printf("Warning: Perhaps reduce the zeta distribution width, particle cannot be at the previous turn in the first turn." 
             " Value of absolute index is %f at this turn %ld",abs_index,at_turn-start_at_turn);
            slot =0;
             }
               if (slot > ((stop_at_turn-start_at_turn)*index_in_turn*1.05)) //momentum spread of 1% can have outliers upto 5% 
	    {
		    printf("Warning: Particle with ID %ld is much ahead of reference particle,"
                 " increase safety margin of memory allocation for the samples data \n",last_particle_id);
	    }
                x_cen = LocalParticle_get_x(part)+BPMRecord_get_x_cen(record,slot);
		BPMRecord_set_x_cen(record,slot,x_cen);
		y_cen = LocalParticle_get_y(part)+BPMRecord_get_y_cen(record,slot);
                BPMRecord_set_y_cen(record,slot,y_cen);
                active_parts = 1+BPMRecord_get_summed_particles(record,slot);
                BPMRecord_set_summed_particles(record,slot,active_parts);
    //end_per_particle_block
         BPMRecord_set_at_turn(record, at_turn-start_at_turn, at_turn);
         BPMRecord_set_last_particle_id(record,at_turn-start_at_turn,last_particle_id);
        }

} 
#endif

