// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_XYSHIFT_H
#define XTRACK_XYSHIFT_H


/*gpufun*/
void XYShift_single_particle(LocalParticle* part, double dx, double dy){

    LocalParticle_add_to_x(part, -dx );
    LocalParticle_add_to_y(part, -dy );
 
}


/*gpufun*/
void XYShift_track_local_particle(XYShiftData el, LocalParticle* part0){

    double const dx = XYShiftData_get_dx(el);
    double const dy = XYShiftData_get_dy(el);

    //start_per_particle_block (part0->part)
        XYShift_single_particle(part, dx, dy);
    //end_per_particle_block

}


#endif /* XTRACK_XYSHIFT_H */
