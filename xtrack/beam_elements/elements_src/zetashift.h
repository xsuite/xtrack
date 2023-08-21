// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_ZETASHIFT_H
#define XTRACK_ZETASHIFT_H

/*gpufun*/
void ZetaShift_track_local_particle(ZetaShiftData el, LocalParticle* part0){


    double dzeta = ZetaShiftData_get_dzeta(el);
    #ifdef XSUITE_BACKTRACK
        dzeta = -dzeta;
    #endif

    //start_per_particle_block (part0->part)

        LocalParticle_add_to_zeta(part, -dzeta);

    //end_per_particle_block

}

#endif
