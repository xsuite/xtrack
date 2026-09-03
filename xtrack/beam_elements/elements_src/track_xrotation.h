// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //
#ifndef XTRACK_TRACK_XROTATION_H
#define XTRACK_TRACK_XROTATION_H

#include "xtrack/headers/track.h"


GPUFUN
void XRotation_single_particle(LocalParticle* part, double sin_angle, double cos_angle, double tan_angle)
{
    double const beta0 = LocalParticle_get_beta0(part);
    xt_num_t const x  = LocalParticle_get_x(part);
    xt_num_t const y  = LocalParticle_get_y(part);
    xt_num_t const px = LocalParticle_get_px(part);
    xt_num_t const py = LocalParticle_get_py(part);
    xt_num_t const t = LocalParticle_get_zeta(part)/beta0;
    xt_num_t const pt = LocalParticle_get_pzeta(part)*beta0;

    xt_num_t pz = sqrt(1.0 + 2.0*pt/beta0 + pt*pt - px*px - py*py);
    xt_num_t ptt = 1.0 - tan_angle*py/pz;
    xt_num_t y_hat = y/(cos_angle*ptt);
    xt_num_t py_hat = cos_angle*py + sin_angle*pz;
    xt_num_t x_hat = x + tan_angle*y*px/(pz*ptt);
    xt_num_t t_hat = t - tan_angle*y*(1.0/beta0+pt)/(pz*ptt);

    /* Spin tracking is disabled by the synrad compile flag */
    #if !defined(XTRACK_TPSA_TRACK) && !defined(XTRACK_MULTIPOLE_NO_SYNRAD)
        // Rotate spin
        double const spin_y_0 = LocalParticle_get_spin_y(part);
        double const spin_z_0 = LocalParticle_get_spin_z(part);
        if ((spin_y_0 != 0) || (spin_z_0 != 0)){
            double const spin_y_1 = cos_angle*spin_y_0 + sin_angle*spin_z_0;
            double const spin_z_1 = -sin_angle*spin_y_0 + cos_angle*spin_z_0;
            LocalParticle_set_spin_y(part, spin_y_1);
            LocalParticle_set_spin_z(part, spin_z_1);
        }
    #endif

    LocalParticle_set_x(part, x_hat);
    LocalParticle_set_py(part, py_hat);
    LocalParticle_set_y(part, y_hat);
    LocalParticle_set_zeta(part, t_hat*beta0);
}

#endif /* XTRACK_TRACK_XROTATION_H */
