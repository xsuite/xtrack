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
    const uint8_t kill_energy_kick
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

    double const energy_kick = q * voltage
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

        for (int64_t kk = 0; kk <= order; kk++)
        {

            if (kk>0){
                factorial *= kk;
            }

            double const pn_kk = phase0 + DEG2RAD * pn[kk] - (2.0 * PI) / C_LIGHT * frequency * tau;
            double const ps_kk = phase0 + DEG2RAD * ps[kk] - (2.0 * PI) / C_LIGHT * frequency * tau;

            double bal_n_kk = factor_knl_ksl * knl[kk]/factorial;
            double bal_s_kk = factor_knl_ksl * ksl[kk]/factorial;

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

    if (transverse_voltage != 0) {
        double dpx = 0.0;
        double dpy = 0.0;
        double dptr = 0.0;
        double zre = 1.0;
        double zim = 0.0;

        double const x = LocalParticle_get_x(part);
        double const y = LocalParticle_get_y(part);
        double const p0c = LocalParticle_get_p0c(part);

        double const pn_kk = phase0 + DEG2RAD * (transverse_lag + 90.) - (2.0 * PI) / C_LIGHT * frequency * tau;
        double const k0l = transverse_voltage / p0c;

        double bal_n_kk = k0l;

        double const cn = cos(pn_kk);
        double const sn = sin(pn_kk);

        dpx += cn * (bal_n_kk * zre);
        dpy += cn * (bal_n_kk * zim);

        double const zret = zre * x - zim * y;
        zim = zim * x + zre * y;
        zre = zret;

        dptr += sn * (bal_n_kk * zre);

        rfmultipole_energy_kick += - q * ( (frequency * ( 2.0 * PI / C_LIGHT) * p0c) * dptr );
        double const chi    = LocalParticle_get_chi(part);

        double const px_kick = - chi * dpx;
        double const py_kick =   chi * dpy;

        LocalParticle_add_to_px(part, px_kick);
        LocalParticle_add_to_py(part, py_kick);

    }


    if (!kill_energy_kick) {
        #ifdef XTRACK_CAVITY_PRESERVE_ANGLE
        LocalParticle_add_to_energy(part, energy_kick + rfmultipole_energy_kick, 0);
        #else
        LocalParticle_add_to_energy(part, energy_kick + rfmultipole_energy_kick, 1);
        #endif
    }

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
    const int8_t integrator,
    const uint8_t kill_energy_kick
) {

    #define RF_KICK(part, kick_weight) \
        track_rf_kick_single_particle(\
            part, voltage * (kick_weight), frequency, lag,\
            transverse_voltage * (kick_weight), transverse_lag,\
            absolute_time, order, \
            factor_knl_ksl * (kick_weight), knl, ksl, pn, ps,\
            kill_energy_kick\
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

    
    // START GENERATED INTEGRATION CODE

    if (integrator == 1){ // TEAPOT

        WITH_RF_RADIATION(length,
            const double kick_weight = 1. / num_kicks;
            double edge_drift_weight = 0.5;
            double inside_drift_weight = 0;
            if (num_kicks > 1) {
                edge_drift_weight = 1. / (2 * (1 + num_kicks));
                inside_drift_weight = (
                    ((double) num_kicks)
                        / ((double)(num_kicks*num_kicks) - 1));
            }

            RF_DRIFT(part, edge_drift_weight*length);
            for (int i_kick=0; i_kick<num_kicks - 1; i_kick++) {
                RF_KICK(part, kick_weight);
                RF_DRIFT(part, inside_drift_weight*length);
            }
            RF_KICK(part, kick_weight);
            RF_DRIFT(part, edge_drift_weight*length);
        )

    }
    else if (integrator==3){ // uniform

        const double kick_weight = 1. / num_kicks;
        const double drift_weight = kick_weight;

        for (int i_kick=0; i_kick<num_kicks; i_kick++) {
            WITH_RF_RADIATION(drift_weight*length,
                RF_DRIFT(part, 0.5*drift_weight*length);
                RF_KICK(part, kick_weight);
                RF_DRIFT(part, 0.5*drift_weight*length);
            )
        }

    }
    else if (integrator==2){ // YOSHIDA 4

        const int64_t n_kicks_yoshida = 7;
        const int64_t num_slices = (num_kicks / n_kicks_yoshida
                                + (num_kicks % n_kicks_yoshida != 0));

        const double slice_length = length / (num_slices);
        const double kick_weight = 1. / num_slices;
        const double d_yoshida[] =
                     // From MAD-NG
                     {3.922568052387799819591407413100e-01,
                      5.100434119184584780271052295575e-01,
                      -4.710533854097565531482416645304e-01,
                      6.875316825251809316199569366290e-02};
                    //  {0x1.91abc4988937bp-2, 0x1.052468fb75c74p-1, // same in hex
                    //  -0x1.e25bd194051b9p-2, 0x1.199cec1241558p-4 };
                    //  {1/8.0, 1/8.0, 1/8.0, 1/8.0}; // Uniform, for debugging
        const double k_yoshida[] =
                     // From MAD-NG
                     {7.845136104775599639182814826199e-01,
                      2.355732133593569921359289764951e-01,
                      -1.177679984178870098432412305556e+00,
                      1.315186320683906284756403692882e+00};
                    //  {0x1.91abc4988937bp-1, 0x1.e2743579895b4p-3, // same in hex
                    //  -0x1.2d7c6f7933b93p+0, 0x1.50b00cfb7be3ep+0 };
                    //  {1/7.0, 1/7.0, 1/7.0, 1/7.0}; // Uniform, for debugging

            for (int ii = 0; ii < num_slices; ii++) {
                WITH_RF_RADIATION(slice_length,
                    RF_DRIFT(part, slice_length * d_yoshida[0]);
                    RF_KICK(part, kick_weight * k_yoshida[0]);
                    RF_DRIFT(part, slice_length * d_yoshida[1]);
                    RF_KICK(part, kick_weight * k_yoshida[1]);
                    RF_DRIFT(part, slice_length * d_yoshida[2]);
                    RF_KICK(part, kick_weight * k_yoshida[2]);
                    RF_DRIFT(part, slice_length * d_yoshida[3]);
                    RF_KICK(part, kick_weight * k_yoshida[3]);
                    RF_DRIFT(part, slice_length * d_yoshida[3]);
                    RF_KICK(part, kick_weight * k_yoshida[2]);
                    RF_DRIFT(part, slice_length * d_yoshida[2]);
                    RF_KICK(part, kick_weight * k_yoshida[1]);
                    RF_DRIFT(part, slice_length * d_yoshida[1]);
                    RF_KICK(part, kick_weight * k_yoshida[0]);
                    RF_DRIFT(part, slice_length * d_yoshida[0]);
                ) // WITH_RF_RADIATION
            }
    } // integrator if

    // END GENERATED INTEGRATION CODE



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
    uint8_t kill_energy_kick = LocalParticle_check_track_flag(
                    part0, XS_FLAG_KILL_CAVITY_KICK);

    // Backtracking
    double body_length;
    double factor_knl_ksl_body;

    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        body_length = -length;
        factor_knl_ksl_body = -factor_knl_ksl;
        VSWAP(edge_entry_active, edge_exit_active);
        voltage = -voltage;
        transverse_voltage = -transverse_voltage;
    } else {
        body_length = length;
        factor_knl_ksl_body = factor_knl_ksl;
    }

    if (body_active){

        if (integrator == 0){
            integrator = default_integrator;
        }
        if (model == 0){
            model = default_model;
        }
        if (model==-1){ // kick only
            integrator = 3; // uniform
            num_kicks = 1;
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
                body_length * weight,
                voltage * weight,
                frequency,
                lag + lag_taper,
                transverse_voltage * weight,
                transverse_lag,
                absolute_time,
                order,
                factor_knl_ksl_body * weight,
                knl,
                ksl,
                pn,
                ps,
                num_kicks,
                drift_model,
                integrator,
                kill_energy_kick
            );
        END_PER_PARTICLE_BLOCK;


    }
}

#endif