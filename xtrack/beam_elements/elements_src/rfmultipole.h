// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //

#ifndef XTRACK_RFMULTIPOLE_H
#define XTRACK_RFMULTIPOLE_H

#include <headers/track.h>


GPUFUN
void RFMultipole_track_local_particle(RFMultipoleData el, LocalParticle* part0){

    GPUGLMEM double const* knl = RFMultipoleData_getp1_knl(el, 0);
    GPUGLMEM double const* ksl = RFMultipoleData_getp1_ksl(el, 0);
    GPUGLMEM double const* pn = RFMultipoleData_getp1_pn(el, 0);
    GPUGLMEM double const* ps = RFMultipoleData_getp1_ps(el, 0);
    int64_t const order = RFMultipoleData_get_order(el);
    double const frequency = RFMultipoleData_get_frequency(el);
    double voltage = RFMultipoleData_get_voltage(el);
    double const lag = RFMultipoleData_get_lag(el);

    #ifdef XSUITE_BACKTRACK
        voltage = -voltage;
    #endif

    START_PER_PARTICLE_BLOCK(part0, part);
        double const k = frequency * ( 2.0 * PI / C_LIGHT);

        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const beta0  = LocalParticle_get_beta0(part);
        double const zeta   = LocalParticle_get_zeta(part);
        double const q      = LocalParticle_get_q0(part)
                            * LocalParticle_get_charge_ratio(part);
        double const ktau   = k * zeta / beta0;

        double dpx = 0.0;
        double dpy = 0.0;
        double dptr = 0.0;
        double zre = 1.0;
        double zim = 0.0;

        double factorial = 1.0;
        for (int64_t kk = 0; kk <= order; kk++)
        {

            if (kk>0){
                factorial *= kk;
            }

            double const pn_kk = DEG2RAD * pn[kk] - ktau;
            double const ps_kk = DEG2RAD * ps[kk] - ktau;

            double bal_n_kk = knl[kk]/factorial;
            double bal_s_kk = ksl[kk]/factorial;

            #ifdef XSUITE_BACKTRACK
                bal_n_kk = -bal_n_kk;
                bal_s_kk = -bal_s_kk;
            #endif

            double const cn = cos(pn_kk);
            double const cs = cos(ps_kk);
            double const sn = sin(pn_kk);
            double const ss = sin(ps_kk);

            dpx += cn * (bal_n_kk * zre) - cs * (bal_s_kk * zim);
            dpy += cs * (bal_s_kk * zre) + cn * (bal_n_kk * zim);

            double const zret = zre * x - zim * y;
            zim = zim * x + zre * y;
            zre = zret;

            dptr += sn * (bal_n_kk * zre) - ss * (bal_s_kk * zim);
        }

        double const cav_energy = q * voltage * sin(lag * DEG2RAD - ktau);
        double const p0c = LocalParticle_get_p0c(part);
        double const rfmultipole_energy = - q * ( (k * p0c) * dptr );

        double const chi    = LocalParticle_get_chi(part);

        double const px_kick = - chi * dpx;
        double const py_kick =   chi * dpy;
        double const energy_kick = cav_energy + rfmultipole_energy;

        LocalParticle_add_to_px(part, px_kick);
        LocalParticle_add_to_py(part, py_kick);
        LocalParticle_add_to_energy(part, energy_kick, 1);
    END_PER_PARTICLE_BLOCK;
}

#endif
