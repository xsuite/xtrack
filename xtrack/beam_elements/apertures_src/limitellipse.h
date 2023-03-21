// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_LIMITELLIPSE_H
#define XTRACK_LIMITELLIPSE_H

/*gpufun*/
void LimitEllipse_track_local_particle(LimitEllipseData el, LocalParticle* part0){


    double const a_squ = LimitEllipseData_get_a_squ(el);
    double const b_squ = LimitEllipseData_get_b_squ(el);
    double const a_b_squ = LimitEllipseData_get_a_b_squ(el);

    //start_per_particle_block (part0->part)
        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);

	double const temp = x*x*b_squ + y*y*a_squ;

	int64_t const is_alive = (int64_t)( temp <= a_b_squ );

	// I assume that if I am in the function is because
    	if (!is_alive){
           LocalParticle_set_state(part, XT_LOST_ON_APERTURE);
	}

    //end_per_particle_block

}

#endif
