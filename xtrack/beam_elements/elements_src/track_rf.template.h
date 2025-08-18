// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACK_RF_H
#define XTRACK_TRACK_RF_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet_drift.h>
#include <beam_elements/elements_src/track_magnet_configure.h>

#ifndef VSWAP
    #define VSWAP(a, b) { double tmp = a; a = b; b = tmp; }
#endif


GPUFUN
void track_rf_kick_single_particle(
    LocalParticle* part,
    double voltage,
    double frequency,
    double lag,
    double transverse_voltage,
    double transverse_lag,
    int64_t absolute_time,
    int64_t order,
    double factor_knl_ksl,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    GPUGLMEM const double* pn,
    GPUGLMEM const double* ps,
    double weight
){

    double phase0 = 0;

    if (absolute_time == 1) {
        double const t_sim = LocalParticle_get_t_sim(part);
        int64_t const at_turn = LocalParticle_get_at_turn(part);
        phase0 += 2 * PI * at_turn * frequency * t_sim;
    }

    double const beta0 = LocalParticle_get_beta0(part);
    double const zeta  = LocalParticle_get_zeta(part);
    double const q = fabs(LocalParticle_get_q0(part)) * LocalParticle_get_charge_ratio(part);
    double const tau = zeta / beta0;

    double const energy_kick = weight * q * voltage
        * sin(phase0 + DEG2RAD * lag - (2.0 * PI) / C_LIGHT * frequency * tau);

    double rfmultipole_energy_kick = 0;
    if (order >= 0) {

        double dpx = 0.0;
        double dpy = 0.0;
        double dptr = 0.0;
        double zre = 1.0;
        double zim = 0.0;
        double factorial = 1.0;

        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const p0c = LocalParticle_get_p0c(part);

        for (int64_t iter = 0; iter <= order+1; iter++)
        {

            int64_t kk = iter;
            double knl_kk, ksl_kk, pn0_kk, ps0_kk;

            if (iter == order + 1){
                // last iteration used for transverse_voltage, transverse_lag
                kk = 0;
                knl_kk = transverse_voltage / p0c;
                ksl_kk = 0;
                pn0_kk = transverse_lag;
                ps0_kk = 0;
                factorial = 1.0;
            }
            else
            {
                knl_kk = knl[kk];
                ksl_kk = ksl[kk];
                pn0_kk = pn[kk];
                ps0_kk = ps[kk];
            }

            double const pn_kk = phase0 + DEG2RAD * pn0_kk - (2.0 * PI) / C_LIGHT * frequency * tau;
            double const ps_kk = phase0 + DEG2RAD * ps0_kk - (2.0 * PI) / C_LIGHT * frequency * tau;

            double bal_n_kk = factor_knl_ksl * knl_kk / factorial * weight;
            double bal_s_kk = factor_knl_ksl * ksl_kk / factorial * weight;

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

        rfmultipole_energy_kick = - q * ( (frequency * ( 2.0 * PI / C_LIGHT) * p0c) * dptr );
        double const chi    = LocalParticle_get_chi(part);

        double const px_kick = - chi * dpx;
        double const py_kick =   chi * dpy;

        LocalParticle_add_to_px(part, px_kick);
        LocalParticle_add_to_py(part, py_kick);

    }


    #ifdef XTRACK_CAVITY_PRESERVE_ANGLE
    LocalParticle_add_to_energy(part, energy_kick + rfmultipole_energy_kick, 0);
    #else
    LocalParticle_add_to_energy(part, energy_kick + rfmultipole_energy_kick, 1);
    #endif

}


GPUFUN
void track_rf_body_single_particle(
    LocalParticle* part,
    double length,
    double voltage,
    double frequency,
    double lag,
    double transverse_voltage,
    double transverse_lag,
    int64_t absolute_time,
    int64_t order,
    double factor_knl_ksl,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    GPUGLMEM const double* pn,
    GPUGLMEM const double* ps,
    const int64_t num_kicks,
    const int8_t drift_model,
    const int8_t integrator
) {

    #define RF_KICK(part, weight) \
        track_rf_kick_single_particle(\
            part, voltage, frequency, lag,\
            transverse_voltage, transverse_lag,\
            absolute_time, order, \
            factor_knl_ksl, knl, ksl, pn, ps, (weight)\
        )

    #define RF_DRIFT(part, dlength) \
        track_magnet_drift_single_particle(\
            part, (dlength), 0., 0., 0., 0.,\
            0., 0., drift_model\
        )

    // No radiation implemented for RF elements for now
    #define WITH_RF_RADIATION(ll, code)\
        {\
            code;\
        }

    INTEGRATION_CODE[[
        INTEGRATOR=integrator,
        DRIFT_FUNCTION=RF_DRIFT,
        KICK_FUNCTION=RF_KICK,
        RADIATION_MACRO=WITH_RF_RADIATION,
        PART=part,
        LENGTH=length,
        NUM_KICKS=num_kicks
    ]]


    #undef RF_KICK
    #undef RF_DRIFT
    #undef WITH_RF_RADIATION

}

GPUFUN
void track_rf_particles(
    double const weight,
    LocalParticle* part0,
    double length,
    double voltage,
    double frequency,
    double lag,
    double transverse_voltage,
    double transverse_lag,
    int64_t absolute_time,
    int64_t order,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    GPUGLMEM const double* pn,
    GPUGLMEM const double* ps,
    int64_t num_kicks,
    int8_t model,
    int8_t default_model,
    int8_t integrator,
    int8_t default_integrator,
    int64_t radiation_flag,
    int64_t radiation_flag_parent,
    double lag_taper,
    int64_t body_active,
    int64_t edge_entry_active,
    int64_t edge_exit_active
) {

    double factor_knl_ksl = 1.0;

    // Backtracking
    #ifdef XSUITE_BACKTRACK
        const double body_length = -length * weight;
        double factor_knl_ksl_body = -factor_knl_ksl * weight;
        VSWAP(edge_entry_active, edge_exit_active);
        voltage = -voltage;
    #else
        const double body_length = length * weight;
        double factor_knl_ksl_body = factor_knl_ksl * weight;
    #endif

    if (body_active){

        if (integrator == 0){
            integrator = default_integrator;
        }
        if (model == 0){
            model = default_model;
        }

        // Adjust the number of kicks based on the weight
        if (weight != 1.0 && num_kicks > 0){
            num_kicks = (int64_t) ceil(num_kicks * weight);
        }

        // Compute the number of kicks for auto mode
        if (num_kicks == 0) { // num_multipole_kicks = 0 means auto mode
            num_kicks = 1;
        }

        double k0_drift, k1_drift, h_drift, ks_drift;
        double k0_kick, k1_kick, h_kick;
        double k0_h_correction, k1_h_correction;
        int8_t kick_rot_frame;
        int8_t drift_model;
        configure_tracking_model(
            model,
            0, // k0
            0, // k1
            0, // h
            0, // ks
            &k0_drift,
            &k1_drift,
            &h_drift,
            &ks_drift,
            &k0_kick,
            &k1_kick,
            &h_kick,
            &k0_h_correction,
            &k1_h_correction,
            &kick_rot_frame,
            &drift_model
        );


        START_PER_PARTICLE_BLOCK(part0, part);
            track_rf_body_single_particle(
                part,
                body_length,
                voltage,
                frequency,
                lag + lag_taper,
                transverse_voltage,
                transverse_lag,
                absolute_time,
                order,
                factor_knl_ksl_body,
                knl,
                ksl,
                pn,
                ps,
                num_kicks,
                drift_model,
                integrator
            );
        END_PER_PARTICLE_BLOCK;


    }
}

#endif