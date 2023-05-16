// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_REFERENCEENERGYINCREASE_H
#define XTRACK_REFERENCEENERGYINCREASE_H

/*gpufun*/
void ReferenceEnergyIncrease_track_local_particle(ReferenceEnergyIncreaseData el,
		                                  LocalParticle* part0){

    double Delta_p0c = ReferenceEnergyIncreaseData_get_Delta_p0c(el);

    #ifdef XSUITE_BACKTRACK
        Delta_p0c = -Delta_p0c;
    #endif

    //start_per_particle_block (part0->part)
	LocalParticle_update_p0c(part,
		LocalParticle_get_p0c(part) + Delta_p0c);
    //end_per_particle_block

}
#endif
