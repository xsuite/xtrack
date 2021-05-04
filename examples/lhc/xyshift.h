#ifndef XTRACK_XYSHIFT_H
#define XTRACK_XYSHIFT_H

void XYShift_track_local_particle(XYShiftData el, LocalParticle* part){

    double const minus_dx = -(XYShiftData_get_dx(el));
    double const minus_dy = -(XYShiftData_get_dy(el));

    LocalParticle_add_to_x(part, minus_dx );
    LocalParticle_add_to_y(part, minus_dy );

}

#endif
