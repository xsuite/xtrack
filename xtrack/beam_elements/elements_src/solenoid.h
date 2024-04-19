// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_SOLENOID_H
#define XTRACK_SOLENOID_H


/*gpufun*/
void Solenoid_track_local_particle(SolenoidData el, LocalParticle* part0) {
    // Parameters
    double length = SolenoidData_get_length(el);
    double ks = SolenoidData_get_ks(el);
    int64_t radiation_flag = SolenoidData_get_radiation_flag(el);

    #ifdef XSUITE_BACKTRACK
        length = -length;
    #endif

    #ifndef XTRACK_SOLENOID_NO_SYNRAD
    double dp_record_entry = 0.;
    double dpx_record_entry = 0.;
    double dpy_record_entry = 0.;
    double dp_record_exit = 0.;
    double dpx_record_exit = 0.;
    double dpy_record_exit = 0.;
    #endif

    //start_per_particle_block (part0->part)
    Solenoid_thick_track_single_particle(part, length, ks, radiation_flag,
                    &dp_record_entry, &dpx_record_entry, &dpy_record_entry,
                    &dp_record_exit, &dpx_record_exit, &dpy_record_exit);
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