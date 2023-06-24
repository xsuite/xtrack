// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_FRINGE_H
#define XTRACK_FRINGE_H

#define POW2(X) ((X)*(X))
#define POW3(X) ((X)*(X)*(X))
#define POW4(X) ((X)*(X)*(X)*(X))

/*gpufun*/
void Fringe_Gianni_single_particle(
        LocalParticle* part,  // LocalParticle to track
        const double fint,    // Fringe field integral
        const double hgap,    // Half gap
        const double k0       // Dipole strength
) {

    const double rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const double y = LocalParticle_get_y(part);
    const double px = LocalParticle_get_px(part);
    const double py = LocalParticle_get_py(part);
    const double delta = LocalParticle_get_delta(part);

    const double one_plus_delta = delta + 1.0;

    const double pz_sq = POW2(one_plus_delta) - POW2(px) - POW2(py);
    const double pz = sqrt(pz_sq);
    const double xp = px / pz;
    const double yp = py / pz;

    const double dpz_dpx = -xp;
    const double dpz_dpy = -yp;
    const double dpz_ddelta = one_plus_delta / pz;

    const double dxp_dpx =    -px/pz_sq * dpz_dpx     + 1/pz;
    const double dxp_dpy =    -px/pz_sq * dpz_dpy;
    const double dxp_ddelta = -px/pz_sq * dpz_ddelta;

    const double dyp_dpx =    -py/pz_sq * dpz_dpx;
    const double dyp_dpy =    -py/pz_sq * dpz_dpy     + 1/pz;
    const double dyp_ddelta = -py/pz_sq * dpz_ddelta;

    const double phi0 = xp / (1 + POW2(yp));
    const double dphi0_dxp = 1 / (1 + POW2(yp));
    const double dphi0_dyp = -2 * xp * yp / POW2(1 + POW2(yp));

    const double phi1 = 1 + 2 * POW2(xp) + POW2(xp) * POW2(yp);
    const double dphi1_dxp = 4 * xp + 2 * POW2(yp) * xp;
    const double dphi1_dyp = 2 * POW2(xp) * yp;

    const double dphi0_dpx = dphi0_dxp * dxp_dpx + dphi0_dyp * dyp_dpx;
    const double dphi0_dpy = dphi0_dxp * dxp_dpy + dphi0_dyp * dyp_dpy;
    const double dphi0_ddelta = dphi0_dxp * dxp_ddelta + dphi0_dyp * dyp_ddelta;

    const double dphi1_dpx = dphi1_dxp * dxp_dpx + dphi1_dyp * dyp_dpx;
    const double dphi1_dpy = dphi1_dxp * dxp_dpy + dphi1_dyp * dyp_dpy;
    const double dphi1_ddelta = dphi1_dxp * dxp_ddelta + dphi1_dyp * dyp_ddelta;

    const double g = 2 * hgap;
    const double Phi = k0 * atan(phi0) - g * k0 * k0 * fint * pz * phi1;
    const double dPhi_dpx = k0 * 1 / (1 + POW2(phi0)) * dphi0_dpx
                    - g * k0 * k0 * fint * (pz * dphi1_dpx + phi1 * dpz_dpx);
    const double dPhi_dpy = k0 * 1 / (1 + POW2(phi0)) * dphi0_dpy
                    - g * k0 * k0 * fint * (pz * dphi1_dpy + phi1 * dpz_dpy);
    const double dPhi_ddelta = k0 * 1 / (1 + POW2(phi0)) * dphi0_ddelta
                    - g * k0 * k0 * fint * (pz * dphi1_ddelta + phi1 * dpz_ddelta);

    const double new_y = 2 * y / (1 + sqrt(1 - 2 * dPhi_dpy * y));
    const double delta_x = dPhi_dpx * POW2(new_y) / 2;
    const double delta_py = -Phi * new_y;
    const double delta_zeta = - 1 / (2 * rvv) * dPhi_ddelta * POW2(new_y);

    LocalParticle_add_to_x(part, delta_x);
    LocalParticle_add_to_py(part, delta_py);
    LocalParticle_add_to_zeta(part, delta_zeta);
    LocalParticle_set_y(part, new_y);

}

/*gpufun*/
void Fringe_single_particle(
        LocalParticle* part,  // LocalParticle to track
        const double fint,    // Fringe field integral
        const double hgap,    // Half gap
        const double k0       // Dipole strength
) {
    const double rvv = LocalParticle_get_rvv(part);
    // Particle coordinates
    const double y = LocalParticle_get_y(part);
    const double px = LocalParticle_get_px(part);
    const double py = LocalParticle_get_py(part);
    const double delta = LocalParticle_get_delta(part);

    // Translate input variables
    const double g = 2 * hgap;
    const double K = fint;
    const double b0 = k0;

    // Useful constants
    const double one_plus_delta = (1 + delta);
    const double one_plus_delta_sq = one_plus_delta * one_plus_delta;

    const double pz = sqrt(one_plus_delta_sq - POW2(px) - POW2(py));
    const double x_prime = px / pz;
    const double y_prime = py / pz;

    const double px2 = POW2(px);
    const double px3 = px2 * px;
    const double py2 = POW2(py);
    const double pz2 = POW2(pz);
    const double pz3 = pz2 * pz;
    const double pz5 = pz2 * pz3;
    const double kk = g * POW2(b0) * K;

    // Phi and its derivatives
    const double Phi = (b0 * x_prime)/(1 + POW2(y_prime)) - \
        kk * ((one_plus_delta_sq - py2)/pz3 + px2/pz2 * ((one_plus_delta_sq - px2)/(pz3)));
    const double dPhi_dpx = 2 * kk * (px3/pz5 - (one_plus_delta_sq - px2)*px/pz5);
    const double dPhi_dpy = 2 * kk * py/pz3;
    const double dPhi_ddelta = -2 * kk * (one_plus_delta * px2/pz5 + one_plus_delta/pz3);

    // Map
    const double new_y = (2 * y) / (1 + sqrt(1 - 2 * dPhi_dpy * y));
    const double delta_x = (dPhi_dpx * POW2(new_y)) / 2;
    const double delta_py = -Phi * new_y;
    const double delta_ell = -(dPhi_ddelta * POW2(new_y)) / 2;

    // Update the particle
    LocalParticle_add_to_x(part, delta_x);
    LocalParticle_set_y(part, new_y);
    LocalParticle_add_to_py(part, delta_py);
    LocalParticle_add_to_zeta(part, -delta_ell / rvv);

}


// From MAD-NG, refactored from PTC: https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/master/libs/ptc/src/Sh_def_kind.f90#L4936

/*gpufun*/
void MadNG_Fringe_single_particle(
        LocalParticle* part,  // LocalParticle to track
        const double fint,    // Fringe field integral
        const double hgap,    // Half gap
        const double k0       // Dipole strength
) {
    if (fabs(k0) < 10e-10) {
        return;
    }
    const double rvv = LocalParticle_get_rvv(part);
    const double beta0 = LocalParticle_get_beta0(part);

    // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double px = LocalParticle_get_px(part);
    const double y = LocalParticle_get_y(part);
    const double py = LocalParticle_get_py(part);
    const double t = LocalParticle_get_zeta(part) / beta0;
    const double pt = LocalParticle_get_ptau(part);

    const double fh = hgap * fint;
    const double fsad = (fh > 10e-10) ? 1./(72 * fh) : 0;
    const double k0w = k0;

    const double _beta = 1. / (beta0 * rvv);
    const double b0 = k0w * LocalParticle_get_charge_ratio(part) * LocalParticle_get_q0(part);

    const double dpp = 1. + 2*_beta*pt + POW2(pt);
    const double pz = sqrt(dpp - POW2(px) - POW2(py));
    const double _pz = 1./pz;
    const double relp = 1./sqrt(dpp);
    const double tfac = -(_beta + pt);

    const double c2 = b0*fh*2;
    const double c3 = POW2(b0)*fsad*relp;

    const double xp = px/pz;
    const double yp = py/pz;
    const double xyp = xp*yp;
    const double yp2 = 1.+POW2(yp);
    const double xp2 = POW2(xp);
    const double _yp2 = 1./yp2;

    const double fi0 = atan((xp*_yp2)) - c2*(1 + xp2*(1+yp2))*pz;
    const double co2 = b0/POW2(cos(fi0));
    const double co1 = co2/(1 + POW2(xp*_yp2))*_yp2;
    const double co3 = co2*c2;

    const double fi1 =    co1          - co3*2*xp*(1+yp2)*pz;
    const double fi2 = -2*co1*xyp*_yp2 - co3*2*xp*xyp    *pz;
    const double fi3 =                 - co3*(1 + xp2*(1+yp2));

    const double kx = fi1*(1+xp2)*_pz   + fi2*xyp*_pz       - fi3*xp;
    const double ky = fi1*xyp*_pz       + fi2*yp2*_pz       - fi3*yp;
    const double kz = fi1*tfac*xp*POW2(_pz) + fi2*tfac*yp*POW2(_pz) - fi3*tfac*_pz;

    const double new_y = 2 * y / (1 + sqrt(1 - 2 * ky * y));
    const double new_x  = x  + 0.5*kx*POW2(y);
    const double new_py = py              - 4*c3*POW3(y)             - b0*tan(fi0)*y;
    const double new_t = t + 0.5*kz*POW2(y) +   c3*POW4(y)*POW2(relp)*tfac;
    const double new_zeta = new_t * beta0;

    LocalParticle_set_x(part, new_x);
    LocalParticle_set_y(part, new_y);
    LocalParticle_set_py(part, new_py);
    LocalParticle_set_zeta(part, new_zeta);
}


/*gpufun*/
void Fringe_track_local_particle(
        FringeData el,
        LocalParticle* part0
) {
    // Parameters
    const double fint = FringeData_get_fint(el);
    const double hgap = FringeData_get_hgap(el);
    const double k = FringeData_get_k(el);

    //start_per_particle_block (part0->part)
        #ifdef XTRACK_FRINGE_GIANNI
           Fringe_Gianni_single_particle(part, fint, hgap, k);
        #else
            MadNG_Fringe_single_particle(part, fint, hgap, k);
        #endif
    //end_per_particle_block
}

#endif // XTRACK_FRINGE_H