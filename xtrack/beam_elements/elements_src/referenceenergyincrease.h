#ifndef XTRACK_REFERENCEENERGYINCREASE_H
#define XTRACK_REFERENCEENERGYINCREASE_H

/*gpufun*/
void ReferenceEnergyIncrease_track_local_particle(ReferenceEnergyIncreaseData el,
		                                  LocalParticle* part0){

    double const Delta_p0c = ReferenceEnergyIncreaseData_get_Delta_p0c(el);

    //start_per_particle_block (part0->part)
	LocalParticle_update_p0c(part, 
		LocalParticle_get_p0c(part) + Delta_p0c);
    //end_per_particle_block

}
#endif
