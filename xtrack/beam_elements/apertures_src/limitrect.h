
#ifndef XTRACK_LimitRect_H
#define XTRACK_LimitRect_H

/*gpufun*/
void LimitRect_track_local_particle(LimitRectData el, LocalParticle* part){

    double const min_x = LimitRectData_get_min_x(el);
    double const max_x = LimitRectData_get_max_x(el);
    double const min_y = LimitRectData_get_min_y(el);
    double const max_y = LimitRectData_get_max_y(el);

    double const n_part = LocalParticle_get_num_particles(part); //only_for_context cpu_serial cpu_openmp
    for (int ii=0; ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
	part->ipart = ii;            //only_for_context cpu_serial cpu_openmp


        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);

	int64_t const is_alive = (int64_t)(
                      (x >= min_x) &&
		      (x <= max_x) &&
		      (y >= min_y) &&
		      (y <= max_y) );

	// I assume that if I am in the function is because
    	if (!is_alive){
           LocalParticle_set_state(part, 0);
	}

    } //only_for_context cpu_serial cpu_openmp

}

#endif
