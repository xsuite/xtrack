#ifndef XTRACK_XYSHIFT_H
#define XTRACK_XYSHIFT_H

/*gpufun*/
void XYShift_track_local_particle(XYShiftData el, LocalParticle* part0){

    double const minus_dx = -(XYShiftData_get_dx(el));
    double const minus_dy = -(XYShiftData_get_dy(el));

    //start_per_particle_block (part0->part)
    	LocalParticle_add_to_x(part, minus_dx );
    	LocalParticle_add_to_y(part, minus_dy );
    //end_per_particle_block
}

#endif
