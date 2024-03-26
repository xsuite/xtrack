// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_SOLENOID_H
#define XTRACK_SOLENOID_H

#define IS_ZERO(X) (fabs(X) < 1e-9)

/*gpufun*/
void Solenoid_thin_track_single_particle(LocalParticle*, double, double, double);

/*gpufun*/
void Solenoid_thick_track_single_particle(LocalParticle*, double, double, int64_t,
                                          double*, double*, double*,
                                          double*, double*, double*);


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




/*gpufun*/
void Solenoid_thick_track_single_particle(
    LocalParticle* part,
    double length,
    double ks,
    int64_t radiation_flag,
    double* dp_record_entry, double* dpx_record_entry, double* dpy_record_entry,
    double* dp_record_exit, double* dpx_record_exit, double* dpy_record_exit
) {
    const double sk = ks / 2;

    if (IS_ZERO(sk)) {
        Drift_single_particle_exact(part, length);
        LocalParticle_set_ax(part, 0);
        LocalParticle_set_ay(part, 0);
        return;
    }

    if (IS_ZERO(length)){
        return;
    }

    const double skl = sk * length;

    // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double px = LocalParticle_get_px(part);
    const double y = LocalParticle_get_y(part);
    const double py = LocalParticle_get_py(part);
    const double delta = LocalParticle_get_delta(part);
    const double rvv = LocalParticle_get_rvv(part);

    // set up constants
    const double pk1 = px + sk * y;
    const double pk2 = py - sk * x;
    const double ptr2 = pk1 * pk1 + pk2 * pk2;
    const double one_plus_delta = 1 + delta;
    const double one_plus_delta_sq = one_plus_delta * one_plus_delta;
    const double pz = sqrt(one_plus_delta_sq - ptr2);

    // set up constants
    const double cosTh = cos(skl / pz);
    const double sinTh = sin(skl / pz);

    const double si = sin(skl / pz) / sk;
    const double rps[4] = {
        cosTh * x + sinTh * y,
        cosTh * px + sinTh * py,
        cosTh * y - sinTh * x,
        cosTh * py - sinTh * px
    };
    const double new_x = cosTh * rps[0] + si * rps[1];
    const double new_px = cosTh * rps[1] - sk * sinTh * rps[0];
    const double new_y = cosTh * rps[2] + si * rps[3];
    const double new_py = cosTh * rps[3] - sk * sinTh * rps[2];
    const double add_to_zeta = length * (1 - one_plus_delta / (pz * rvv));

    // Update ax and ay (Wolsky Eq. 3.114 and Eq. 2.74)
    double const p0c = LocalParticle_get_p0c(part);
    double const q0 = LocalParticle_get_q0(part);
    double const P0_J = p0c * QELEM / C_LIGHT;
    double const brho = P0_J / QELEM / q0;
    double const Bz = brho * ks;
    double const new_ax = -0.5 * Bz * new_y * q0 * QELEM / P0_J;
    double const new_ay = 0.5 * Bz * new_x * q0 * QELEM / P0_J;

    #ifndef XTRACK_SOLENOID_NO_SYNRAD
        double l_path, curv;
        if (radiation_flag > 0 && length > 0){

            double const old_ax = LocalParticle_get_ax(part);
            double const old_ay = LocalParticle_get_ay(part);

            double const old_px_mech = px - old_ax;
            double const old_py_mech = py - old_ay;

            double const new_px_mech = new_px - new_ax;
            double const new_py_mech = new_py - new_ay;

            double const dpx = new_px_mech - old_px_mech;
            double const dpy = new_py_mech - old_py_mech;

            curv = sqrt(dpx*dpx + dpy*dpy) / length;

            // Path length for radiation
            double const dzeta = add_to_zeta;
            double const rvv = LocalParticle_get_rvv(part);
            l_path = rvv * (length - dzeta);

            // LocalParticle_add_to_px(part, -old_ax);
            // LocalParticle_add_to_py(part, -old_ay);
            // if (radiation_flag == 1){
            //     synrad_average_kick(part, curv, l_path / 2,
            //             dp_record_entry, dpx_record_entry, dpy_record_entry);
            // }
            // else if (radiation_flag == 2){
            //     synrad_emit_photons(part, curv, l_path / 2, NULL, NULL);
            // }
            // LocalParticle_add_to_px(part, old_ax);
            // LocalParticle_add_to_py(part, old_ay);
        }
    #endif

    LocalParticle_set_x(part, new_x);
    LocalParticle_add_to_px(part, new_px - px);
    LocalParticle_set_y(part, new_y);
    LocalParticle_add_to_py(part, new_py - py);
    LocalParticle_add_to_zeta(part, add_to_zeta);
    LocalParticle_add_to_s(part, length);
    LocalParticle_set_ax(part, new_ax);
    LocalParticle_set_ay(part, new_ay);

    #ifndef XTRACK_SOLENOID_NO_SYNRAD

        if (radiation_flag > 0 && length > 0){
            LocalParticle_add_to_px(part, -new_ax);
            LocalParticle_add_to_py(part, -new_ay);
            if (radiation_flag == 1){
                synrad_average_kick(part, curv, l_path,
                    dp_record_exit, dpx_record_exit, dpy_record_exit);
            }
            else if (radiation_flag == 2){
                synrad_emit_photons(part, curv, l_path, NULL, NULL);
            }
            LocalParticle_add_to_px(part, new_ax);
            LocalParticle_add_to_py(part, new_ay);
        }
    #endif
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