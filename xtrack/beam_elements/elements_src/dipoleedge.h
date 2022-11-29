// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_DIPOLEEDGE_H
#define XTRACK_DIPOLEEDGE_H

/*gpufun*/
void DipoleEdge_track_local_particle(DipoleEdgeData el, LocalParticle* part0){

    double r21 = DipoleEdgeData_get_r21(el);
    double r43 = DipoleEdgeData_get_r43(el);

    //start_per_particle_block (part0->part)
	double const x = LocalParticle_get_x(part);
	double const y = LocalParticle_get_y(part);

    #ifdef XTRACK_DIPOLEEDGE_TAPER
        double const delta_taper = LocalParticle_get_delta(part);
        r21 = r21 * (1 + delta_taper);
        r43 = r43 * (1 + delta_taper);
    #endif

	LocalParticle_add_to_px(part, r21*x);
	LocalParticle_add_to_py(part, r43*y);

    //end_per_particle_block

}

#endif
