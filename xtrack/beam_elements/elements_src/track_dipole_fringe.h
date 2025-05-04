// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACK_DIPOLE_FRINGE_H
#define XTRACK_TRACK_DIPOLE_FRINGE_H

#include <headers/track.h>

#ifndef XTRACK_FRINGE_FROM_PTC

// MAD-NG implementation
// https://github.com/MethodicalAcceleratorDesign/MAD/blob/d3cabd9cdebde62ebedb51bab61ac033b9159489/src/madl_dynmap.mad#L1864

GPUFUN
void DipoleFringe_single_particle(
        LocalParticle* part,  // LocalParticle to track
        const double fint,    // Fringe field integral
        const double hgap,    // Half gap
        const double k0       // Dipole strength
) {
    if (fabs(k0) < 10e-10) {
        return;
    }

    const double beta0 = LocalParticle_get_beta0(part);

    // Particle coordinates
    const double x = LocalParticle_get_x(part);
    const double px = LocalParticle_get_px(part);
    const double y = LocalParticle_get_y(part);
    const double py = LocalParticle_get_py(part);
    const double t = LocalParticle_get_zeta(part) / beta0;
    const double pt = LocalParticle_get_ptau(part);
    const double delta = LocalParticle_get_delta(part);

    const double fh = hgap * fint;
    const double fsad = (fh > 10e-10) ? 1./(72 * fh) : 0;
    const double k0w = k0 * LocalParticle_get_chi(part);

    const double _beta = 1. / beta0 ;
    const double b0 = k0w; // MAD does something with the charge (to be checked)

    const double dpp = POW2(1. + delta);
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
    const double new_x  = x  + 0.5 * kx * POW2(new_y);
    const double new_py = py - 4 * c3 * POW3(new_y) - b0 * tan(fi0) * new_y;
    const double new_t = t + 0.5 * kz * POW2(new_y) + c3 * POW4(new_y) * POW2(relp) * tfac;

    const double new_zeta = new_t * beta0;

    LocalParticle_set_x(part, new_x);
    LocalParticle_set_y(part, new_y);
    LocalParticle_set_py(part, new_py);
    LocalParticle_set_zeta(part, new_zeta);
}

#endif // no XTRACK_FRINGE_FROM_PTC


#ifdef XTRACK_FRINGE_FROM_PTC
// The following is ported from PTC:
//https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/master/libs/ptc/src/Sh_def_kind.f90#L4936

GPUFUN
void DipoleFringe_single_particle(
        LocalParticle* part,  // LocalParticle to track
        const double fint,    // Fringe field integral
        const double hgap,    // Half gap
        const double k0       // Dipole strength
) {
    if (fabs(k0) < 10e-10) {
        return;
    }

    const double b = k0; // PTC naming convention

    double fsad=0.0;
    if(fint*hgap != 0.){
      fsad=1./(fint*hgap*2)/36.0;
    }

    // Particle coordinates
    const double beta0 = LocalParticle_get_beta0(part);
    const double x = LocalParticle_get_x(part);
    const double px = LocalParticle_get_px(part);
    const double y = LocalParticle_get_y(part);
    const double py = LocalParticle_get_py(part);
    const double ptau = LocalParticle_get_ptau(part);
    const double delta = LocalParticle_get_delta(part);

    const double pz = sqrt((1.0 + delta)*(1.0 + delta) - px * px - py * py);
    const double time_fac = 1/beta0 + ptau;
    const double rel_p = sqrt(1. + 2*ptau/beta0 + POW2(ptau));
    const double c3 = b * b * fsad / rel_p;

    const double xp = px / pz;
    const double yp = py / pz;

    const double D_1_1 = (1.0 + xp * xp) / pz;
    const double D_2_1 =  xp * yp / pz;
    const double D_3_1 = -xp;
    const double D_1_2 = xp * yp / pz;
    const double D_2_2 = (1.0 + yp * yp)/pz;
    const double D_3_2 = -yp;
    const double D_1_3 = -time_fac * xp / (pz * pz);
    const double D_2_3 = -time_fac * yp / (pz * pz);
    const double D_3_3 =  time_fac / pz;

    double fi0 = atan((xp / (1.0 + yp * yp)))-b * fint * hgap * 2.0 * ( 1.0 + xp * xp *(2.0 + yp * yp)) * pz;
    const double co2 = b / cos(fi0) / cos(fi0);
    const double co1 = co2 / (1.0 + POW2(xp / POW2(1.0 + yp * yp)));

    const double fi_1 = co1 / (1.0 + yp*yp) - co2 * b * fint * hgap * 2.0*(2.0 * xp * (2.0 + yp * yp) * pz);
    const double fi_2 = -co1 * 2.0 * xp * yp / POW2(1.0 + yp * yp) - co2 * b * fint * hgap * 2.0 * (2.0 * xp* xp * yp) * pz;
    const double fi_3 = -co2 * b * fint * hgap * 2.0 * (1.0 + xp * xp * (2.0 + yp*yp));

    fi0 = b * tan(fi0);

    double BB = 0;
    BB = fi_1 * D_1_2 + BB;
    BB = fi_2 * D_2_2 + BB;
    BB = fi_3 * D_3_2 + BB;

    const double new_y = 2.0 * y / (1.0 + sqrt(1.0 - 2.0 * BB * y));
    double new_py = py - fi0 * new_y;

    BB = 0;
    BB = fi_1 * D_1_1 + BB;
    BB = fi_2 * D_2_1 + BB;
    BB = fi_3 * D_3_1 + BB;
    const double new_x = x + 0.5 * BB * new_y * new_y;

    BB = 0;
    BB = fi_1 * D_1_3 + BB;
    BB = fi_2 * D_2_3 + BB;
    BB = fi_3 * D_3_3 + BB;
    double d_tau = -0.5 * BB * new_y * new_y;

    new_py = new_py - 4 * c3 * POW3(new_y);
    d_tau = d_tau + c3 * POW4(new_y) / POW2(rel_p) * time_fac;

    LocalParticle_set_x(part, new_x);
    LocalParticle_set_y(part, new_y);
    LocalParticle_set_py(part, new_py);
    LocalParticle_add_to_zeta(part, -d_tau * beta0); // PTC uses tau = ct
}
#endif // XTRACK_FRINGE_FROM_PTC


// The following is derived from https://cds.cern.ch/record/2857004
// still to be checked

// GPUFUN
// void DipoleFringe_single_particle(
//         LocalParticle* part,  // LocalParticle to track
//         const double fint,    // Fringe field integral
//         const double hgap,    // Half gap
//         const double k0       // Dipole strength
// ) {
//
//     if (fabs(k0) < 10e-10) {
//         return;
//     }
//
//     const double rvv = LocalParticle_get_rvv(part);
//     // Particle coordinates
//     const double y = LocalParticle_get_y(part);
//     const double px = LocalParticle_get_px(part);
//     const double py = LocalParticle_get_py(part);
//     const double delta = LocalParticle_get_delta(part);

//     const double one_plus_delta = delta + 1.0;

//     const double pz_sq = POW2(one_plus_delta) - POW2(px) - POW2(py);
//     const double pz = sqrt(pz_sq);
//     const double xp = px / pz;
//     const double yp = py / pz;

//     const double dpz_dpx = -xp;
//     const double dpz_dpy = -yp;
//     const double dpz_ddelta = one_plus_delta / pz;

//     const double dxp_dpx =    -px/pz_sq * dpz_dpx     + 1/pz;
//     const double dxp_dpy =    -px/pz_sq * dpz_dpy;
//     const double dxp_ddelta = -px/pz_sq * dpz_ddelta;

//     const double dyp_dpx =    -py/pz_sq * dpz_dpx;
//     const double dyp_dpy =    -py/pz_sq * dpz_dpy     + 1/pz;
//     const double dyp_ddelta = -py/pz_sq * dpz_ddelta;

//     const double phi0 = xp / (1 + POW2(yp));
//     const double dphi0_dxp = 1 / (1 + POW2(yp));
//     const double dphi0_dyp = -2 * xp * yp / POW2(1 + POW2(yp));

//     const double phi1 = 1 + 2 * POW2(xp) + POW2(xp) * POW2(yp);
//     const double dphi1_dxp = 4 * xp + 2 * POW2(yp) * xp;
//     const double dphi1_dyp = 2 * POW2(xp) * yp;

//     const double dphi0_dpx = dphi0_dxp * dxp_dpx + dphi0_dyp * dyp_dpx;
//     const double dphi0_dpy = dphi0_dxp * dxp_dpy + dphi0_dyp * dyp_dpy;
//     const double dphi0_ddelta = dphi0_dxp * dxp_ddelta + dphi0_dyp * dyp_ddelta;

//     const double dphi1_dpx = dphi1_dxp * dxp_dpx + dphi1_dyp * dyp_dpx;
//     const double dphi1_dpy = dphi1_dxp * dxp_dpy + dphi1_dyp * dyp_dpy;
//     const double dphi1_ddelta = dphi1_dxp * dxp_ddelta + dphi1_dyp * dyp_ddelta;

//     const double g = 2 * hgap;
//     const double Phi = k0 * atan(phi0) - g * k0 * k0 * fint * pz * phi1;
//     const double dPhi_dpx = k0 * 1 / (1 + POW2(phi0)) * dphi0_dpx
//                     - g * k0 * k0 * fint * (pz * dphi1_dpx + phi1 * dpz_dpx);
//     const double dPhi_dpy = k0 * 1 / (1 + POW2(phi0)) * dphi0_dpy
//                     - g * k0 * k0 * fint * (pz * dphi1_dpy + phi1 * dpz_dpy);
//     const double dPhi_ddelta = k0 * 1 / (1 + POW2(phi0)) * dphi0_ddelta
//                     - g * k0 * k0 * fint * (pz * dphi1_ddelta + phi1 * dpz_ddelta);

//     // printf("dPhi_dpy = %e\n", dPhi_dpy);
//     // printf("Phi = %e\n", Phi);

//     const double new_y = 2 * y / (1 + sqrt(1 - 2 * dPhi_dpy * y));
//     const double delta_x = dPhi_dpx * POW2(new_y) / 2;
//     const double delta_py = -Phi * new_y;
//     const double delta_zeta = - 1 / (2 * rvv) * dPhi_ddelta * POW2(new_y);

//     LocalParticle_add_to_x(part, delta_x);
//     LocalParticle_add_to_py(part, delta_py);
//     LocalParticle_add_to_zeta(part, delta_zeta);
//     LocalParticle_set_y(part, new_y);

// }

#endif // XTRACK_TRACK_DIPOLE_FRINGE_H