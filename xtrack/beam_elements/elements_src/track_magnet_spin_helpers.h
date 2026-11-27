// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MAGNET_SPIN_HELPERS_H
#define XTRACK_TRACK_MAGNET_SPIN_HELPERS_H

GPUFUN
void direction_of_motion(
    double const px,
    double const py,
    double const delta,
    double* iv_x,
    double* iv_y,
    double* iv_s){

    double iix = px / (1. + delta);
    double iiy = py / (1. + delta);
    double iis = sqrt(1 - iix * iix - iiy * iiy);

    *iv_x = iix;
    *iv_y = iiy;
    *iv_s = iis;
}

GPUFUN
void separate_par_perp_components(
    double const Bx,
    double const By,
    double const Bz,
    double const ix,
    double const iy,
    double const iz,
    double* B_par_x,
    double* B_par_y,
    double* B_par_z,
    double* B_perp_x,
    double* B_perp_y,
    double* B_perp_z
){

    double B_par = Bx * ix + By * iy + Bz * iz;
    double const BB_par_x = B_par * ix;
    double const BB_par_y = B_par * iy;
    double const BB_par_z = B_par * iz;

    double const BB_perp_x = Bx - BB_par_x;
    double const BB_perp_y = By - BB_par_y;
    double const BB_perp_z = Bz - BB_par_z;

    *B_par_x = BB_par_x;
    *B_par_y = BB_par_y;
    *B_par_z = BB_par_z;
    *B_perp_x = BB_perp_x;
    *B_perp_y = BB_perp_y;
    *B_perp_z = BB_perp_z;

}

GPUFUN
double compute_b_perp_mod(
    double const kin_px,
    double const kin_py,
    double const delta,
    double const Bx,
    double const By,
    double const Bz){

    double iv_x, iv_y, iv_z;
    direction_of_motion(kin_px, kin_py, delta,
                        &iv_x, &iv_y, &iv_z);

    double B_par_x, B_par_y, B_par_z;
    double B_perp_x, B_perp_y, B_perp_z;
    separate_par_perp_components(
        Bx,
        By,
        Bz,
        iv_x,
        iv_y,
        iv_z,
        &B_par_x,
        &B_par_y,
        &B_par_z,
        &B_perp_x,
        &B_perp_y,
        &B_perp_z);

    return sqrt(B_perp_x*B_perp_x + B_perp_y*B_perp_y + B_perp_z*B_perp_z);

}

#endif
