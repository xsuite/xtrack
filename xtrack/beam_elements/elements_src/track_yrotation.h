// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //
#ifndef XTRACK_TRACK_YROTATION_H
#define XTRACK_TRACK_YROTATION_H

#include <headers/track.h>


GPUFUN
void YRotation_single_particle(LocalParticle* part, double sin_angle, double cos_angle, double tan_angle){

    double const beta0 = LocalParticle_get_beta0(part);
    double const x  = LocalParticle_get_x(part);
    double const y  = LocalParticle_get_y(part);
    double const px = LocalParticle_get_px(part);
    double const py = LocalParticle_get_py(part);
    double const t = LocalParticle_get_zeta(part)/beta0;
    double const pt = LocalParticle_get_pzeta(part)*beta0;

    double pz = sqrt(1.0 + 2.0*pt/beta0 + pt*pt - px*px - py*py);
    double ptt = 1.0 - tan_angle*px/pz;
    double x_hat = x/(cos_angle*ptt);
    double px_hat = cos_angle*px + sin_angle*pz;
    double y_hat = y + tan_angle*x*py/(pz*ptt);
    double t_hat = t - tan_angle*x*(1.0/beta0+pt)/(pz*ptt);

    LocalParticle_set_x(part, x_hat);
    LocalParticle_set_px(part, px_hat);
    LocalParticle_set_y(part, y_hat);
    LocalParticle_set_zeta(part,t_hat*beta0);

}

#endif /* XTRACK_TRACK_YROTATION_H */
