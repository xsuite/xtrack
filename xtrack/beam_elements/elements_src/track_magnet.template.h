// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MAGNET_H
#define XTRACK_TRACK_MAGNET_H

#include <headers/track.h>
#include <beam_elements/elements_src/track_magnet_kick.h>
#include <beam_elements/elements_src/track_magnet_drift.h>
#include <beam_elements/elements_src/track_magnet_edge.h>
#include <beam_elements/elements_src/track_magnet_configure.h>

#ifndef XTRACK_MULTIPOLE_NO_SYNRAD
#include <beam_elements/elements_src/track_magnet_radiation.h>
#endif

#define H_TOLERANCE (1e-8)

#ifndef VSWAP
    #define VSWAP(a, b) { double tmp = a; a = b; b = tmp; }
#endif


GPUFUN
void track_magnet_body_single_particle(
    LocalParticle* part,
    const double length,
    const int64_t order,
    const double inv_factorial_order,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    const double factor_knl_ksl,
    const int64_t num_multipole_kicks,
    const int8_t kick_rot_frame,
    const int8_t drift_model,
    const int8_t integrator,
    const double k0_drift,
    const double k1_drift,
    const double ks_drift,
    const double h_drift,
    const double k0_kick,
    const double k1_kick,
    const double h_kick,
    const double hxl,
    const double k0_h_correction,
    const double k1_h_correction,
    const double k2,
    const double k3,
    const double k0s,
    const double k1s,
    const double k2s,
    const double k3s,
    const double dks_ds,
    const double x0_solenoid,
    const double y0_solenoid,
    const int64_t radiation_flag,
    const int64_t spin_flag,
    SynchrotronRadiationRecordData radiation_record,
    double* dp_record_exit,
    double* dpx_record_exit,
    double* dpy_record_exit
) {

    #define MAGNET_KICK(part, weight) \
        track_magnet_kick_single_particle(\
            part, length, order, inv_factorial_order, \
            knl, ksl, factor_knl_ksl, (weight), \
            k0_kick, k1_kick, k2, k3, k0s, k1s, k2s, k3s, h_kick,\
            hxl, k0_h_correction, k1_h_correction, kick_rot_frame\
        )

    #define MAGNET_DRIFT(part, dlength) \
        track_magnet_drift_single_particle(\
            part, (dlength), k0_drift, k1_drift, ks_drift, h_drift,\
            x0_solenoid, y0_solenoid, drift_model\
        )

    #ifdef XTRACK_MULTIPOLE_NO_SYNRAD
        #define WITH_RADIATION(ll, code)\
        {\
            code;\
        }
    #else
        #define WITH_RADIATION(ll, code) \
        { \
            double const old_x = LocalParticle_get_x(part); \
            double const old_y = LocalParticle_get_y(part); \
            double const old_zeta = LocalParticle_get_zeta(part); \
            double const old_kin_px = LocalParticle_get_px(part) - LocalParticle_get_ax(part);\
            double const old_kin_py = LocalParticle_get_py(part) - LocalParticle_get_ay(part);\
            code; \
            if ((radiation_flag || spin_flag) && length > 0){ \
                double h_for_rad = h_kick + hxl / length; \
                if (fabs(h_drift) > 0){ h_for_rad = h_drift; } \
                double Bx_T, By_T, Bz_T; \
                double const p0c = LocalParticle_get_p0c(part); \
                double const q0 = LocalParticle_get_q0(part); \
                double const new_x = LocalParticle_get_x(part); \
                double const new_y = LocalParticle_get_y(part); \
                double const new_kin_px = LocalParticle_get_px(part) - LocalParticle_get_ax(part);\
                double const new_kin_py = LocalParticle_get_py(part) - LocalParticle_get_ay(part);\
                double const mean_x = 0.5 * (old_x + new_x); \
                double const mean_y = 0.5 * (old_y + new_y); \
                double const mean_kin_px = 0.5 * (old_kin_px + new_kin_px); \
                double const mean_kin_py = 0.5 * (old_kin_py + new_kin_py); \
                evaluate_field_from_strengths( \
                    p0c, \
                    q0, \
                    mean_x, \
                    mean_y, \
                    length, \
                    order, \
                    inv_factorial_order, \
                    knl, \
                    ksl, \
                    factor_knl_ksl, \
                    k0_drift + k0_kick, \
                    k1_drift + k1_kick, \
                    k2, \
                    k3, \
                    k0s, \
                    k1s, \
                    k2s, \
                    k3s, \
                    ks_drift, \
                    dks_ds, \
                    x0_solenoid, \
                    y0_solenoid, \
                    &Bx_T, \
                    &By_T, \
                    &Bz_T \
                ); \
                double const dzeta = LocalParticle_get_zeta(part) - old_zeta; \
                double const rvv = LocalParticle_get_rvv(part); \
                double l_path = rvv * (ll - dzeta); \
                if (spin_flag){ \
                    magnet_spin( \
                        part, \
                        Bx_T, \
                        By_T, \
                        Bz_T, \
                        h_for_rad, \
                        ll, \
                        l_path); \
                } \
                if (radiation_flag){ \
                    double const B_perp_T = compute_b_perp_mod( \
                        mean_kin_px, \
                        mean_kin_py, \
                        LocalParticle_get_delta(part), \
                        Bx_T, \
                        By_T, \
                        Bz_T); \
                    magnet_radiation( \
                        part, \
                        B_perp_T, \
                        ll, \
                        l_path, \
                        radiation_flag, \
                        radiation_record, \
                        dp_record_exit, dpx_record_exit, dpy_record_exit \
                    ); \
                } \
            }\
        }
    #endif

    if (num_multipole_kicks == 0 && k0_kick == 0 && k1_kick == 0 && h_kick == 0) { //only drift
        WITH_RADIATION(length,
            MAGNET_DRIFT(part, length);
        )
    }
    else{

        INTEGRATION_CODE[[
          INTEGRATOR=integrator,
          DRIFT_FUNCTION=MAGNET_DRIFT,
          KICK_FUNCTION=MAGNET_KICK,
          RADIATION_MACRO=WITH_RADIATION,
          PART=part,
          LENGTH=length,
          NUM_KICKS=num_multipole_kicks
        ]]

    }

    #undef MAGNET_KICK
    #undef MAGNET_DRIFT
    #undef WITH_RADIATION

}

GPUFUN
void track_magnet_particles(
    double const weight,
    LocalParticle* part0,
    double length,
    int64_t order,
    double inv_factorial_order,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    int64_t num_multipole_kicks,
    int8_t model,
    int8_t default_model,
    int8_t integrator,
    int8_t default_integrator,
    int64_t radiation_flag,
    int64_t radiation_flag_parent,
    SynchrotronRadiationRecordData radiation_record,
    double delta_taper,
    double h,
    double hxl,
    double k0,
    double k1,
    double k2,
    double k3,
    double k0s,
    double k1s,
    double k2s,
    double k3s,
    double ks,
    double dks_ds,
    double x0_solenoid,
    double y0_solenoid,
    int64_t rbend_model, // -1: not used, 0: auto, 1: curved body, 2: straight body
    double rbend_shift,
    int64_t body_active,
    int64_t edge_entry_active,
    int64_t edge_exit_active,
    int64_t edge_entry_model,
    int64_t edge_exit_model,
    double edge_entry_angle,
    double edge_exit_angle,
    double edge_entry_angle_fdown,
    double edge_exit_angle_fdown,
    double edge_entry_fint,
    double edge_exit_fint,
    double edge_entry_hgap,
    double edge_exit_hgap
) {

    double factor_knl_ksl = 1.0;

    // Used only for rbend-straight-body
    double rbend_half_angle = 0.;
    double cos_rbha = 0.;
    double sin_rbha = 0.;
    double length_curved = 0.;
    double dx_rb = 0.;


    if (rbend_model == 0){
        // auto mode, curved body
        rbend_model = 1;
    }

    if (rbend_model == 1){
        // curved body
        double const angle = h * length;
        edge_entry_angle += angle / 2.0;
        edge_exit_angle += angle / 2.0;
    }
    else if (rbend_model == 2){
        // straight body
        double sinc_rbha;
        rbend_half_angle = h * length / 2;
        if (fabs(rbend_half_angle) > 1e-10){
            sin_rbha = sin(rbend_half_angle);
            cos_rbha = cos(rbend_half_angle);
            sinc_rbha = sin_rbha / rbend_half_angle;
        }
        else {
            sinc_rbha = 1.;
            cos_rbha = 1.;
        }
        length_curved = length;
        length = length_curved * sinc_rbha;
        if (fabs(rbend_half_angle) > 1e-10){
            // shift by half the sagitta
            dx_rb = 0.5 / h * (1 - cos_rbha) + rbend_shift;
        };
        h = 0; // treat magnet as straight
        // We are entering the fringe with an angle, the linear fringe
        // needs to come from the  expansion around the right angle
        edge_entry_angle_fdown += rbend_half_angle;
        edge_exit_angle_fdown += rbend_half_angle;
    }

    double core_length, core_length_curved,factor_knl_ksl_body,
           factor_knl_ksl_edge,factor_backtrack_edge;
    // Backtracking
    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        core_length = -length * weight;
        core_length_curved = -length_curved * weight;
        factor_knl_ksl_body = -factor_knl_ksl * weight;
        factor_knl_ksl_edge = factor_knl_ksl; // Edge has a specific factor for backtracking
        factor_backtrack_edge = -1.;
        hxl = -hxl;
        VSWAP(edge_entry_active, edge_exit_active);
        VSWAP(edge_entry_model, edge_exit_model);
        VSWAP(edge_entry_angle, edge_exit_angle);
        VSWAP(edge_entry_angle_fdown, edge_exit_angle_fdown);
        VSWAP(edge_entry_fint, edge_exit_fint);
        VSWAP(edge_entry_hgap, edge_exit_hgap);
        rbend_half_angle = -rbend_half_angle;
        sin_rbha = -sin_rbha;
    }
    else {
        core_length = length * weight;
        core_length_curved = length_curved * weight;
        factor_knl_ksl_body = factor_knl_ksl * weight;
        factor_knl_ksl_edge = factor_knl_ksl;
        factor_backtrack_edge = 1.;
    }

    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        if (radiation_flag == 10){ // from parent
            radiation_flag = radiation_flag_parent;
        }
    #endif

    // Tapering
    if (LocalParticle_check_track_flag(part0, XS_FLAG_SR_TAPER)){
        part0->ipart = 0;
        delta_taper = LocalParticle_get_delta(part0); // I can use part0 because
                                                      // there is only one particle
                                                      // when doing the tapering
    }

    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        if (radiation_flag){
            // knl and ksl are scaled by the called functions below using factor_knl_ksl
            factor_knl_ksl_body *= (1. + delta_taper);
            factor_knl_ksl_edge *= (1. + delta_taper);
            // k0, k1, k2, k3, k0s, k1s, k2s, k3s are scaled directly here
            k0 *= (1 + delta_taper);
            k1 *= (1 + delta_taper);
            k2 *= (1 + delta_taper);
            k3 *= (1 + delta_taper);
            k0s *= (1 + delta_taper);
            k1s *= (1 + delta_taper);
            k2s *= (1 + delta_taper);
            k3s *= (1 + delta_taper);
        }
    #endif

    if (edge_entry_active){

        if (rbend_model == 2){
            // straight body --> curvature in the edges
            START_PER_PARTICLE_BLOCK(part0, part);
                YRotation_single_particle(part, -sin_rbha, cos_rbha, -sin_rbha/cos_rbha);
                LocalParticle_add_to_x(part, -dx_rb);
            END_PER_PARTICLE_BLOCK;
        }

        double knorm[] = {k0, k1, k2, k3};
        double kskew[] = {k0s, k1s, k2s, k3s};

        track_magnet_edge_particles(
            part0,
            edge_entry_model,
            0, // is_exit
            edge_entry_hgap,
            knorm,
            kskew,
            3, // k_order,
            knl,
            ksl,
            factor_knl_ksl_edge,
            order,
            ks,
            x0_solenoid,
            y0_solenoid,
            length,
            edge_entry_angle,
            edge_entry_angle_fdown,
            edge_entry_fint,
            factor_backtrack_edge
        );
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
            num_multipole_kicks = 1;
        }

        // Adjust the number of multipole kicks based on the weight
        if (weight != 1.0 && num_multipole_kicks > 0){
            num_multipole_kicks = (int64_t) ceil(num_multipole_kicks * weight);
        }

        // Compute the number of kicks for auto mode
        if (num_multipole_kicks == 0) { // num_multipole_kicks = 0 means auto mode
            // If there are active kicks the number of kicks is guessed. Otherwise,
            // only the drift is performed.
            if (!kick_is_inactive(order, knl, ksl, k0, k1, k2, k3, k0s, k1s, k2s, k3s, h)){
                if (fabs(h) < 1e-8){
                    num_multipole_kicks = 1; // straight magnet, one multipole kick in the middle
                }
                else{
                    double b_circum = 2 * 3.14159 / fabs(h);
                    num_multipole_kicks = fabs(core_length) / b_circum / 0.5e-3; // 0.5 mrad per kick (on average)
                    if (num_multipole_kicks < 1){
                        num_multipole_kicks = 1;
                    }
                }
            }
        }

        double k0_drift=0, k1_drift=0, h_drift=0, ks_drift=0;
        double k0_kick=0, k1_kick=0, h_kick=0;
        double k0_h_correction=0, k1_h_correction=0;
        int8_t kick_rot_frame=0;
        int8_t drift_model=0;
        configure_tracking_model(
            model,
            k0,
            k1,
            h,
            ks,
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

        double dp_record_exit, dpx_record_exit, dpy_record_exit;

        START_PER_PARTICLE_BLOCK(part0, part);
            track_magnet_body_single_particle(
                part, core_length, order, inv_factorial_order,
                knl, ksl,
                factor_knl_ksl_body,
                num_multipole_kicks, kick_rot_frame, drift_model, integrator,
                k0_drift, k1_drift, ks_drift, h_drift,
                k0_kick, k1_kick, h_kick, hxl,
                k0_h_correction, k1_h_correction,
                k2, k3, k0s, k1s, k2s, k3s,
                dks_ds,
                x0_solenoid, y0_solenoid,
                radiation_flag,
                1, // spin_flag
                radiation_record,
                &dp_record_exit, &dpx_record_exit, &dpy_record_exit
            );
        END_PER_PARTICLE_BLOCK;

        if (rbend_model == 2){
            // straight body --> correct s to match the curved frame
            double const ds = (core_length_curved - core_length);
            START_PER_PARTICLE_BLOCK(part0, part);
                LocalParticle_add_to_s(part, ds);
                LocalParticle_add_to_zeta(part, ds);
            END_PER_PARTICLE_BLOCK;
        }
    }

    if (edge_exit_active){
        double knorm[] = {k0, k1, k2, k3};
        double kskew[] = {k0s, k1s, k2s, k3s};

        track_magnet_edge_particles(
            part0,
            edge_exit_model,
            1, // is_exit
            edge_exit_hgap,
            knorm,
            kskew,
            3, // k_order,
            knl,
            ksl,
            factor_knl_ksl_edge,
            order,
            ks,
            x0_solenoid,
            y0_solenoid,
            length,
            edge_exit_angle,
            edge_exit_angle_fdown,
            edge_exit_fint,
            factor_backtrack_edge
        );

        if (rbend_model == 2){
            // straight body --> curvature in the edges
            START_PER_PARTICLE_BLOCK(part0, part);
                LocalParticle_add_to_x(part, dx_rb); // shift by half sagitta
                YRotation_single_particle(part, -sin_rbha, cos_rbha, -sin_rbha/cos_rbha);
            END_PER_PARTICLE_BLOCK;
        }
    }

}

#endif