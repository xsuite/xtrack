#ifndef XTRACK_TRACKER_H
#define XTRACK_TRACKER_H

#ifdef XTRACK_GLOBAL_POSLIMIT

/*gpufun*/
void global_aperture_check(LocalParticle* part0){


    //start_per_particle_block (part0->part)
        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);

	int64_t const is_alive = (int64_t)(
                      (x >= -XTRACK_GLOBAL_POSLIMIT) &&
		      (x <=  XTRACK_GLOBAL_POSLIMIT) &&
		      (y >= -XTRACK_GLOBAL_POSLIMIT) &&
		      (y <=  XTRACK_GLOBAL_POSLIMIT) );

	// I assume that if I am in the function is because
    	if (!is_alive){
           LocalParticle_set_state(part, 0);
	}
    //end_per_particle_block


}
#endif

/*gpufun*/
void increment_at_element(LocalParticle* part0){

   //start_per_particle_block (part0->part)
        LocalParticle_add_to_at_element(part, 1);
   //end_per_particle_block


}

/*gpufun*/
void increment_at_turn(LocalParticle* part0, int flag_reset_s){

    //start_per_particle_block (part0->part)
	LocalParticle_add_to_at_turn(part, 1);
	LocalParticle_set_at_element(part, 0);
    if (flag_reset_s>0){
        LocalParticle_set_s(part, 0.);
    }
    //end_per_particle_block
}


// check_is_active has different implementation on CPU and GPU

#define CPUIMPLEM //only_for_context cpu_serial cpu_openmp

#ifdef CPUIMPLEM

/*gpufun*/
int64_t check_is_active(LocalParticle* part) {
    int64_t ipart=0;
    while (ipart < part->_num_active_particles){
        if (part->state[ipart]<1){
            LocalParticle_exchange(
                part, ipart, part->_num_active_particles-1);
            part->_num_active_particles--; 
            part->_num_lost_particles++; 
        }
	else{
	    ipart++;
	}
    }

    if (part->_num_active_particles==0){
        return 0;//All particles lost
    } else {
        return 1; //Some stable particles are still present
    }
}

#else

/*gpufun*/
int64_t check_is_active(LocalParticle* part) {
    return LocalParticle_get_state(part)>0;
};

#endif

#undef CPUIMPLEM //only_for_context cpu_serial cpu_openmp

#endif
