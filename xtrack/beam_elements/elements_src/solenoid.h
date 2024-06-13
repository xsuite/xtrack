// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2024.                 //
// ######################################### //

#ifndef XTRACK_SOLENOID_H
#define XTRACK_SOLENOID_H


/*gpufun*/
void Solenoid_track_local_particle(SolenoidData el, LocalParticle* part0) {
    // Parameters
    double length = SolenoidData_get_length(el);
    double ks = SolenoidData_get_ks(el);
    int64_t radiation_flag = SolenoidData_get_radiation_flag(el);
    double factor_knl_ksl = 1;

    #ifdef XSUITE_BACKTRACK
        length = -length;
        factor_knl_ksl = -1;
    #endif

    #ifndef XTRACK_SOLENOID_NO_SYNRAD
        double dp_record_entry = 0.;
        double dpx_record_entry = 0.;
        double dpy_record_entry = 0.;
        double dp_record_exit = 0.;
        double dpx_record_exit = 0.;
        double dpy_record_exit = 0.;
    #endif

    int64_t num_multipole_kicks = SolenoidData_get_num_multipole_kicks(el);
    const int64_t order = SolenoidData_get_order(el);
    const double inv_factorial_order = SolenoidData_get_inv_factorial_order(el);
    /*gpuglmem*/ const double *knl = SolenoidData_getp1_knl(el, 0);
    /*gpuglmem*/ const double *ksl = SolenoidData_getp1_ksl(el, 0);
    const double slice_length = length / (num_multipole_kicks + 1);
    const double kick_weight = 1. / num_multipole_kicks;

    //start_per_particle_block (part0->part)
    #ifndef XTRACK_SOLENOID_NO_SYNRAD
        double const old_px = LocalParticle_get_px(part);
        double const old_py = LocalParticle_get_py(part);
        double const old_ax = LocalParticle_get_ax(part);
        double const old_ay = LocalParticle_get_ay(part);
        double const old_zeta = LocalParticle_get_zeta(part);
    #endif

    for (int ii = 0; ii < num_multipole_kicks; ii++) {
        Solenoid_thick_track_single_particle(part, slice_length, ks, radiation_flag);

        track_multipolar_kick_bend(
                    part, order, inv_factorial_order, knl, ksl, factor_knl_ksl,
                    kick_weight, 0, 0, 0, 0);
    }

    Solenoid_thick_track_single_particle(part, slice_length, ks, radiation_flag);

    #ifndef XTRACK_SOLENOID_NO_SYNRAD
        if (radiation_flag > 0 && length > 0){
            Solenoid_apply_radiation_single_particle(
                part, length, radiation_flag,
                old_px, old_py, old_ax, old_ay, old_zeta,
                &dp_record_entry, &dpx_record_entry, &dpy_record_entry
            );
        }
    #endif
    //end_per_particle_block
}


// /*gpufun*/
// void Solenoid_thin_track_single_particle(
//     LocalParticle* part,
//     double length,
//     double ks,
//     double ksi
// ) {
//     const double sk = ks / 2;
//     const double skl = ksi / 2;
//     const double beta0 = LocalParticle_get_beta0(part);

//     // Particle coordinates
//     const double x = LocalParticle_get_x(part);
//     const double px = LocalParticle_get_px(part);
//     const double y = LocalParticle_get_y(part);
//     const double py = LocalParticle_get_py(part);
//     const double t = LocalParticle_get_zeta(part) / beta0;
//     const double pt = LocalParticle_get_ptau(part);
//     const double delta = LocalParticle_get_delta(part);

//     // Useful quantities
//     const double psigf = pt / beta0;
//     const double betas = beta0;

//     const double onedp = 1 + delta;
//     const double fppsig = (1 + (betas * betas) * psigf) / onedp;

//     // Set up C, S, Q, R, Z
//     const double cosTh = cos(skl / onedp);
//     const double sinTh = sin(skl / onedp);
//     const double Q = -skl * sk / onedp;
//     const double Z = fppsig / (onedp * onedp) * skl;
//     const double R = Z * sk;

//     const double pxf = px + x * Q;
//     const double pyf = py + y * Q;
//     const double sigf = t * betas - 0.5 * (x * x + y * y) * R;

//     // Final angles after solenoid
//     const double pxf_ =  pxf * cosTh + pyf * sinTh;
//     const double pyf_ = -pxf * sinTh + pyf * cosTh;

//     // Calculate new coordinates
//     const double new_x = x  * cosTh  +  y  * sinTh;
//     const double new_px = pxf_;
//     const double new_y = -x  * sinTh  +  y  * cosTh;
//     const double new_py = pyf_;
//     const double new_zeta = sigf + (x * new_py - y * new_px) * Z;

//     LocalParticle_set_x(part, new_x);
//     LocalParticle_set_px(part, new_px);
//     LocalParticle_set_y(part, new_y);
//     LocalParticle_set_py(part, new_py);
//     LocalParticle_set_zeta(part, new_zeta);

// }

#endif // XTRACK_SOLENOID_H