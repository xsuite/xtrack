// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_XROTATION_H
#define XTRACK_XROTATION_H

/*gpufun*/
void XRotation_track_local_particle(XRotationData el, LocalParticle* part0){

    //start_per_particle_block (part0->part)
    	double const cos_angle = XRotationData_get_cos_angle(el);
    	double const sin_angle = XRotationData_get_sin_angle(el);
    	double const tan_angle = XRotationData_get_tan_angle(el);

        double const beta0 = LocalParticle_get_beta0(part);
        double const beta = LocalParticle_get_rvv(part)*beta0;
    	double const x  = LocalParticle_get_x(part);
    	double const y  = LocalParticle_get_y(part);
    	double const px = LocalParticle_get_px(part);
    	double const py = LocalParticle_get_py(part);
    	double const t = LocalParticle_get_zeta(part)/beta0;
    	double const pt = LocalParticle_get_pzeta(part)*beta0;

        double pz = sqrt(1.0 + 2.0*pt/beta + pt*pt - px*px - py*py);
        double ptt = 1.0 - tan_angle*py/pz;
        double x_hat = x + tan_angle*y*px/(pz*ptt);
        double y_hat = y/(cos_angle*ptt);
        double py_hat = cos_angle*py + sin_angle*pz;
        double t_hat = t - tan_angle*y*(1.0/beta+pt)/(pz*ptt);

    	LocalParticle_set_x(part, x_hat);
    	LocalParticle_set_y(part, y_hat);
    	LocalParticle_set_py(part, py_hat);
        LocalParticle_set_zeta(part,t_hat*beta0);

    //end_per_particle_block

}

#endif
