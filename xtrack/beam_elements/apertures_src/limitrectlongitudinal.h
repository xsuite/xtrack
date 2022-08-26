// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_LIMITRECTLONGITUDINAL_H
#define XTRACK_LIMITRECTLONGITUDINAL_H

/*gpufun*/
void LimitRectLongitudinal_track_local_particle(LimitRectLongitudinalData el, LocalParticle* part0){

    double const min_zeta = LimitRectLongitudinalData_get_min_zeta(el);
    double const max_zeta = LimitRectLongitudinalData_get_max_zeta(el);
    double const min_pzeta = LimitRectLongitudinalData_get_min_pzeta(el);
    double const max_pzeta = LimitRectLongitudinalData_get_max_pzeta(el);

    //start_per_particle_block (part0->part)

        double const zeta = LocalParticle_get_zeta(part);
        double const pzeta = LocalParticle_get_pzeta(part);

	int64_t const is_alive = (int64_t)(
                      (zeta >= min_zeta) &&
		      (zeta <= max_zeta) &&
		      (pzeta >= min_pzeta) &&
		      (pzeta <= max_pzeta) );

	// I assume that if I am in the function is because
    	if (!is_alive){
           LocalParticle_set_state(part, 0);
	}

    //end_per_particle_block

}

#endif
