// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_ZETASHIFT_H
#define XTRACK_ZETASHIFT_H

/*gpufun*/
void ZetaShift_track_local_particle(ZetaShiftData el, LocalParticle* part0){

    //start_per_particle_block (part0->part)
    	double const dzeta = ZetaShiftData_get_dzeta(el);

        double const beta0 = LocalParticle_get_beta0(part);
        double const beta = LocalParticle_get_rvv(part)*beta0;
    	double const x  = LocalParticle_get_x(part);
    	double const y  = LocalParticle_get_y(part);
    	double const px = LocalParticle_get_px(part);
    	double const py = LocalParticle_get_py(part);
    	double const zeta = LocalParticle_get_zeta(part);
    	double const pt = LocalParticle_get_pzeta(part)*beta0;

        double pz = sqrt(1.0 + 2.0*pt/beta + pt*pt - px*px - py*py);
        double x_hat = x + dzeta*px/pz;
        double y_hat = y + dzeta*py/pz;
        double zeta_hat = zeta - dzeta*(1.0/beta+pt)/pz;

    	LocalParticle_set_x(part, x_hat);
    	LocalParticle_set_y(part, y_hat);
        LocalParticle_set_zeta(part,zeta_hat);

    //end_per_particle_block

}

#endif
