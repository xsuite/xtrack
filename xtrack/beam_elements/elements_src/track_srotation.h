// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_SROTATION_H
#define XTRACK_TRACK_SROTATION_H

#include "xtrack/headers/track.h"


GPUFUN
void SRotation_single_particle(LocalParticle* part, double sin_z, double cos_z)
{
    xt_float_or_tpsa const x  = LocalParticle_get_x(part);
    xt_float_or_tpsa const y  = LocalParticle_get_y(part);
    xt_float_or_tpsa const px = LocalParticle_get_px(part);
    xt_float_or_tpsa const py = LocalParticle_get_py(part);

    xt_float_or_tpsa const x_hat  =  cos_z * x  + sin_z * y;
    xt_float_or_tpsa const y_hat  = -sin_z * x  + cos_z * y;
    xt_float_or_tpsa const px_hat =  cos_z * px + sin_z * py;
    xt_float_or_tpsa const py_hat = -sin_z * px + cos_z * py;

    /* Spin tracking is disabled by the synrad compile flag */
    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        // Rotate spin
        xt_float_or_tpsa const spin_x_0 = LocalParticle_get_spin_x(part);
        xt_float_or_tpsa const spin_y_0 = LocalParticle_get_spin_y(part);
        if (!xt_float_or_tpsa_is_zero(spin_x_0) || !xt_float_or_tpsa_is_zero(spin_y_0)){
            xt_float_or_tpsa const spin_x_1 = cos_z*spin_x_0 + sin_z*spin_y_0;
            xt_float_or_tpsa const spin_y_1 = -sin_z*spin_x_0 + cos_z*spin_y_0;
            LocalParticle_set_spin_x(part, spin_x_1);
            LocalParticle_set_spin_y(part, spin_y_1);
        }
    #endif

    LocalParticle_set_x(part, x_hat);
    LocalParticle_set_y(part, y_hat);
    LocalParticle_set_px(part, px_hat);
    LocalParticle_set_py(part, py_hat);
}

#endif
