// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_YROTATION_H
#define XTRACK_YROTATION_H


/*gpufun*/
void YRotation_single_particle(LocalParticle* part, double sin_angle, double cos_angle, double tan_angle){

    double const beta0 = LocalParticle_get_beta0(part);
    double const beta = LocalParticle_get_rvv(part)*beta0;
    double const x  = LocalParticle_get_x(part);
    double const y  = LocalParticle_get_y(part);
    double const px = LocalParticle_get_px(part);
    double const py = LocalParticle_get_py(part);
    double const t = LocalParticle_get_zeta(part)/beta0;
    double const pt = LocalParticle_get_pzeta(part)*beta0;

    double pz = sqrt(1.0 + 2.0*pt/beta + pt*pt - px*px - py*py);
    double ptt = 1.0 - tan_angle*px/pz;
    double x_hat = x/(cos_angle*ptt);
    double px_hat = cos_angle*px + sin_angle*pz;
    double y_hat = y + tan_angle*x*py/(pz*ptt);
    double t_hat = t - tan_angle*x*(1.0/beta+pt)/(pz*ptt);

    LocalParticle_set_x(part, x_hat);
    LocalParticle_set_px(part, px_hat);
    LocalParticle_set_y(part, y_hat);
    LocalParticle_set_zeta(part,t_hat*beta0);

}


/*gpufun*/
void YRotation_track_local_particle(YRotationData el, LocalParticle* part0){

    double const sin_angle = YRotationData_get_sin_angle(el);
    double const cos_angle = YRotationData_get_cos_angle(el);
    double const tan_angle = YRotationData_get_tan_angle(el);

    //start_per_particle_block (part0->part)
        YRotation_single_particle(part, sin_angle, cos_angle, tan_angle);
    //end_per_particle_block

}


#endif /* XTRACK_YROTATION_H */
