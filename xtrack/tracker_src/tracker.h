#ifndef XTRACK_TRACKER_H
#define XTRACK_TRACKER_H

#define CPUIMPLEM //only_for_context cpu_serial cpu_openmp

#ifdef CPUIMPLEM

/*gpufun*/
int64_t check_is_not_lost(LocalParticle* part) {
    int64_t ipart=0;
    while (ipart < part->num_particles){
        if (part->state[ipart]<1){
            LocalParticle_exchange(part, ipart, part->num_particles-1);
            part->num_particles--; 
        }
	else{
	    ipart++;
	}
    }

    if (part->num_particles==0){
        return 0;//All particles lost
    } else {
        return 1; //Some stable particles are still present
    }
}

#else

/*gpufun*/
int64_t check_is_not_lost(LocalParticle* part) {
    return LocalParticle_get_state(part);
};

#endif

#undef CPUIMPLEM //only_for_context cpu_serial cpu_openmp

#endif
