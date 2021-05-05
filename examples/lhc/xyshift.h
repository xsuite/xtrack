#ifndef XTRACK_XYSHIFT_H
#define XTRACK_XYSHIFT_H

void XYShift_track_local_particle(XYShiftData el, LocalParticle* part){

    double const minus_dx = -(XYShiftData_get_dx(el));
    double const minus_dy = -(XYShiftData_get_dy(el));

    double const n_part = LocalParticle_get_num_particles(part); 
    for (int ii=0; ii<n_part; ii++){ //only_for_context cpu_serial cpu_openmp
	part->ipart = ii;            //only_for_context cpu_serial cpu_openmp

    	LocalParticle_add_to_x(part, minus_dx );
    	LocalParticle_add_to_y(part, minus_dy );
    } //only_for_context cpu_serial cpu_openmp
}

#endif
