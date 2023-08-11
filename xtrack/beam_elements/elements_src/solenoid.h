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
void Solenoid_thick_track_single_particle(LocalParticle*, double, double);


/*gpufun*/
void Solenoid_track_local_particle(SolenoidData el, LocalParticle* part0) {
    // Parameters
    const double length = SolenoidData_get_length(el);
    const double ks = SolenoidData_get_ks(el);
    const double ksi = SolenoidData_get_ksi(el);

    if (IS_ZERO(length)) {
        //start_per_particle_block (part0->part)
        Solenoid_thin_track_single_particle(part, length, ks, ksi);
        //end_per_particle_block
    }
    else {
        //start_per_particle_block (part0->part)
        Solenoid_thick_track_single_particle(part, length, ks);
        //end_per_particle_block
    }
}


/*gpufun*/
void Solenoid_thin_track_single_particle(
    LocalParticle* part,
    double length,
    double ks,
    double ksi
) {
    const double sk = ks / 2;
    const double skl = ksi / 2;
    const double beta0 = LocalParticle_get_beta0(part);

    // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double px = LocalParticle_get_px(part);
    const double y = LocalParticle_get_y(part);
    const double py = LocalParticle_get_py(part);
    const double t = LocalParticle_get_zeta(part) / beta0;
    const double pt = LocalParticle_get_ptau(part);
    const double delta = LocalParticle_get_delta(part);

    // Useful quantities
    const double psigf = pt / beta0;
    const double betas = beta0;

    const double onedp = 1 + delta;
    const double fppsig = (1 + (betas * betas) * psigf) / onedp;

    // Set up C, S, Q, R, Z
    const double cosTh = cos(skl / onedp);
    const double sinTh = sin(skl / onedp);
    const double Q = -skl * sk / onedp;
    const double Z = fppsig / (onedp * onedp) * skl;
    const double R = Z * sk;

    const double pxf = px + x * Q;
    const double pyf = py + y * Q;
    const double sigf = t * betas - 0.5 * (x * x + y * y) * R;

    // Final angles after solenoid
    const double pxf_ =  pxf * cosTh + pyf * sinTh;
    const double pyf_ = -pxf * sinTh + pyf * cosTh;

    // Calculate new coordinates
    const double new_x = x  * cosTh  +  y  * sinTh;
    const double new_px = pxf_;
    const double new_y = -x  * sinTh  +  y  * cosTh;
    const double new_py = pyf_;
    const double new_zeta = sigf + (x * pyf - y * pxf) * Z;

    LocalParticle_set_x(part, new_x);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_y(part, new_y);
    LocalParticle_set_py(part, new_py);
    LocalParticle_set_zeta(part, new_zeta);
}


/*gpufun*/
void Solenoid_thick_track_single_particle(
    LocalParticle* part,
    double length,
    double ks
) {
    const double sk = ks / 2;  // *other_bv in mad, what is bv?

    if (IS_ZERO(sk)) {
        Drift_single_particle(part, length);
        return;
    }

    const double skl = sk * length;
    const double beta0 = LocalParticle_get_beta0(part);
    const double gamma0 = LocalParticle_get_gamma0(part);

    // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double px = LocalParticle_get_px(part);
    const double y = LocalParticle_get_y(part);
    const double py = LocalParticle_get_py(part);
    const double pt = LocalParticle_get_ptau(part);
    const double delta = LocalParticle_get_delta(part);
    const double rvv = LocalParticle_get_rvv(part);

    // Useful quantities
    const double betas = beta0;
    const double gammas = gamma0; //1 / sqrt(1 - beta0 * beta0);

    // set up constants
    const double pk1 = px + sk * y;
    const double pk2 = py - sk * x;
    const double ptr2 = pk1 * pk1 + pk2 * pk2;
    const double onedp = sqrt(1 + 2 * pt * (1 / beta0) + pt * pt - ptr2);

    // set up constants
    const double cosTh = cos(skl / onedp);
    const double sinTh = sin(skl / onedp);
    // const double omega = sk / onedp;  // unused

    const double si = sin(skl / onedp) * length / skl;
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
    const double add_to_zeta = beta0 * length * \
        (pt * (pt + 2 / beta0) / (gammas * gammas) - ptr2) / \
        (betas * onedp * (1 + onedp + betas * pt));

    LocalParticle_set_x(part, new_x);
    LocalParticle_set_px(part, new_px);
    LocalParticle_set_y(part, new_y);
    LocalParticle_set_py(part, new_py);
    LocalParticle_add_to_zeta(part, add_to_zeta);
    LocalParticle_add_to_s(part, length);
}

#endif // XTRACK_SOLENOID_H