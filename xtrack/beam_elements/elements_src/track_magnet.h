// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRACK_MAGNET_H
#define XTRACK_TRACK_MAGNET_H

#define H_TOLERANCE (1e-8)

#ifndef VSWAP
    #define VSWAP(a, b) { double tmp = a; a = b; b = tmp; }
#endif


/*gpufun*/
void configure_tracking_model(
    int64_t model,
    double k0,
    double k1,
    double h,
    double* k0_drift,
    double* k1_drift,
    double* h_drift,
    double* k0_kick,
    double* k1_kick,
    double* h_kick,
    double* k0_h_correction,
    double* k1_h_correction,
    int8_t* kick_rot_frame,
    int8_t* out_drift_model
){

    // model = 0 or 1 : adaptive
    // model = 2: bend-kick-bend
    // model = 3: rot-kick-rot
    // model = 4: mat-kick-mat (previously called `expanded`)
    // model = 5: drift-kick-drift-exact
    // model = 6: drift-kick-drift-expanded
    // model = -1: kick only (not exposed in python)

    if (model==0 || model==1){
        model = 3;
    }

    int8_t h_is_zero = (fabs(h) < H_TOLERANCE);
    int64_t drift_model = 0;

    if (model == 2){ // bend-kick-bend
        if (h_is_zero){
            drift_model = 5; // bend without h
        }
        else{
            drift_model = 4; // bend with h
        }
    }
    else if(model == 3){ // rot-kick-rot
        if (h_is_zero){
            drift_model = 1; // drift exact
        }
        else{
            drift_model = 2; // polar drift
        }
    }
    else if(model == 4){ // mat-kick-mat
        drift_model = 3; // expanded
    }
    else if(model == 5){ // drift-kick-drift-exact
        drift_model = 1; // drift exact
    }
    else if(model == 6){ // drift-kick-drift-expanded
        drift_model = 0; // drift expanded
    }
    else {
        drift_model = -1;
    }

    if (drift_model == -1 || drift_model == 0 || drift_model == 1){ // drift expanded or drift exact
        *k0_drift = 0.0;
        *k1_drift = 0.0;
        *h_drift = 0.0;
        *k0_kick = k0;
        *k1_kick = k1;
        *h_kick = h;
        *k0_h_correction = k0;
        *k1_h_correction = k1;
        *kick_rot_frame = 1;
    }
    else if (drift_model == 2){ // polar drift
        *k0_drift = 0.0;
        *k1_drift = 0.0;
        *h_drift = h;
        *k0_kick = k0;
        *k1_kick = k1;
        *h_kick = h;
        *k0_h_correction = k0;
        *k1_h_correction = k1;
        *kick_rot_frame = 0;
    }
    else if (drift_model == 3){ // expanded dipole-quadrupole
        *k0_drift = k0;
        *k1_drift = k1;
        *h_drift = h;
        *k0_kick = 0.0;
        *k1_kick = 0.0;
        *h_kick = h;
        *k0_h_correction = 0.;
        *k1_h_correction = k1;
        *kick_rot_frame = 0;
    }
    else if (drift_model == 4){ // bend with h
        *k0_drift = k0;
        *k1_drift = 0.0;
        *h_drift = h;
        *k0_kick = 0.0;
        *k1_kick = k1;
        *h_kick = h;
        *k0_h_correction = 0.;
        *k1_h_correction = k1;
        *kick_rot_frame = 0;
    }
    else if (drift_model == 5){ // bend without h
        *k0_drift = k0;
        *k1_drift = 0.0;
        *h_drift = 0.0;
        *k0_kick = 0.0;
        *k1_kick = k1;
        *h_kick = 0.0;
        *k0_h_correction = 0.;
        *k1_h_correction = 0.;
        *kick_rot_frame = 0;
    }

    *out_drift_model = drift_model;
}


/*gpufun*/
void track_magnet_body_single_particle(
    LocalParticle* part,
    const double length,
    const int64_t order,
    const double inv_factorial_order,
    /*gpuglmem*/ const double* knl,
    /*gpuglmem*/ const double* ksl,
    const double factor_knl_ksl,
    const int64_t num_multipole_kicks,
    const int8_t kick_rot_frame,
    const int8_t drift_model,
    const int8_t integrator,
    const double k0_drift,
    const double k1_drift,
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
    const int64_t radiation_flag,
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
            part, (dlength), k0_drift, k1_drift, h_drift, drift_model\
        )

    #ifdef XTRACK_MULTIPOLE_NO_SYNRAD
        #define WITH_RADIATION(ll, code)\
        {\
            code;\
        }
    #else
        #define WITH_RADIATION(ll, code) \
        { \
            const double old_px = LocalParticle_get_px(part); \
            const double old_py = LocalParticle_get_py(part); \
            const double old_ax = LocalParticle_get_ax(part); \
            const double old_ay = LocalParticle_get_ay(part); \
            const double old_zeta = LocalParticle_get_zeta(part); \
            code; \
            if (radiation_flag && length > 0){ \
                double h_for_rad = h_kick + hxl / length; \
                if (fabs(h_drift) > 0){ h_for_rad = h_drift; } \
                magnet_apply_radiation_single_particle( \
                    part, \
                    (ll), \
                    /*hx*/h_for_rad, \
                    /*hy*/0., \
                    radiation_flag, \
                    old_px, old_py, \
                    old_ax, old_ay, \
                    old_zeta, \
                    radiation_record, \
                    dp_record_exit, dpx_record_exit, dpy_record_exit);\
            }\
        }
    #endif

    if (num_multipole_kicks == 0 && k0_kick == 0 && k1_kick == 0 && h_kick == 0) { //only drift
        WITH_RADIATION(length,
            MAGNET_DRIFT(part, length);
        )
    }
    else if (integrator == 1){ // TEAPOT

        WITH_RADIATION(length,
            const double kick_weight = 1. / num_multipole_kicks;
            double edge_drift_weight = 0.5;
            double inside_drift_weight = 0;
            if (num_multipole_kicks > 1) {
                edge_drift_weight = 1. / (2 * (1 + num_multipole_kicks));
                inside_drift_weight = (
                    ((double) num_multipole_kicks)
                        / ((double)(num_multipole_kicks*num_multipole_kicks) - 1));
            }

            MAGNET_DRIFT(part, edge_drift_weight*length);
            for (int i_kick=0; i_kick<num_multipole_kicks - 1; i_kick++) {
                MAGNET_KICK(part, kick_weight);
                MAGNET_DRIFT(part, inside_drift_weight*length);
            }
            MAGNET_KICK(part, kick_weight);
            MAGNET_DRIFT(part, edge_drift_weight*length);
        )

    }
    else if (integrator==3){ // uniform

        const double kick_weight = 1. / num_multipole_kicks;
        const double drift_weight = kick_weight;

        for (int i_kick=0; i_kick<num_multipole_kicks; i_kick++) {
            WITH_RADIATION(drift_weight*length,
                MAGNET_DRIFT(part, 0.5*drift_weight*length);
                MAGNET_KICK(part, kick_weight);
                MAGNET_DRIFT(part, 0.5*drift_weight*length);
            )
        }

    }
    else if (integrator==0 || integrator==2){ // YOSHIDA 4

        const int64_t n_kicks_yoshida = 7;
        const int64_t num_slices = (num_multipole_kicks / n_kicks_yoshida
                                + (num_multipole_kicks % n_kicks_yoshida != 0));

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
                WITH_RADIATION(slice_length,
                    MAGNET_DRIFT(part, slice_length * d_yoshida[0]);
                    MAGNET_KICK(part, kick_weight * k_yoshida[0]);
                    MAGNET_DRIFT(part, slice_length * d_yoshida[1]);
                    MAGNET_KICK(part, kick_weight * k_yoshida[1]);
                    MAGNET_DRIFT(part, slice_length * d_yoshida[2]);
                    MAGNET_KICK(part, kick_weight * k_yoshida[2]);
                    MAGNET_DRIFT(part, slice_length * d_yoshida[3]);
                    MAGNET_KICK(part, kick_weight * k_yoshida[3]);
                    MAGNET_DRIFT(part, slice_length * d_yoshida[3]);
                    MAGNET_KICK(part, kick_weight * k_yoshida[2]);
                    MAGNET_DRIFT(part, slice_length * d_yoshida[2]);
                    MAGNET_KICK(part, kick_weight * k_yoshida[1]);
                    MAGNET_DRIFT(part, slice_length * d_yoshida[1]);
                    MAGNET_KICK(part, kick_weight * k_yoshida[0]);
                    MAGNET_DRIFT(part, slice_length * d_yoshida[0]);
                ) // WITH_RADIATION
            }
    } // integrator if

    #undef MAGNET_KICK
    #undef MAGNET_DRIFT
    #undef WITH_RADIATION

}

/*gpufun*/
void track_magnet_particles(
    LocalParticle* part0,
    double length,
    int64_t order,
    double inv_factorial_order,
    /*gpuglmem*/ const double* knl,
    /*gpuglmem*/ const double* ksl,
    double factor_knl_ksl,
    int64_t num_multipole_kicks,
    int8_t model,
    int8_t integrator,
    int64_t radiation_flag,
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

    // Backtracking
    #ifdef XSUITE_BACKTRACK
        const double core_length = -length;
        double factor_knl_ksl_body = -factor_knl_ksl;
        double factor_knl_ksl_edge = factor_knl_ksl; // Edge has a specific factor for backtracking
        const double factor_backtrack_edge = -1.;
        hxl = -hxl;
        VSWAP(edge_entry_active, edge_exit_active);
        VSWAP(edge_entry_model, edge_exit_model);
        VSWAP(edge_entry_angle, edge_exit_angle);
        VSWAP(edge_entry_angle_fdown, edge_exit_angle_fdown);
        VSWAP(edge_entry_fint, edge_exit_fint);
        VSWAP(edge_entry_hgap, edge_exit_hgap)
    #else
        const double core_length = length;
        double factor_knl_ksl_body = factor_knl_ksl;
        double factor_knl_ksl_edge = factor_knl_ksl;
        const double factor_backtrack_edge = 1.;
    #endif

    // Tapering
    #ifdef XTRACK_MULTIPOLE_TAPER // Computing the tapering
        part0->ipart = 0;
        delta_taper = LocalParticle_get_delta(part0); // I can use part0 because
                                                      // there is only one particle
                                                      // when doing the tapering
    #endif

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
                num_multipole_kicks = fabs(length) / b_circum / 0.5e-3; // 0.5 mrad per kick (on average)
                if (num_multipole_kicks < 1){
                    num_multipole_kicks = 1;
                }
            }
        }
    }

    double k0_drift, k1_drift, h_drift;
    double k0_kick, k1_kick, h_kick;
    double k0_h_correction, k1_h_correction;
    int8_t kick_rot_frame;
    int8_t drift_model;
    configure_tracking_model(
        model,
        k0,
        k1,
        h,
        &k0_drift,
        &k1_drift,
        &h_drift,
        &k0_kick,
        &k1_kick,
        &h_kick,
        &k0_h_correction,
        &k1_h_correction,
        &kick_rot_frame,
        &drift_model
    );

    double dp_record_exit, dpx_record_exit, dpy_record_exit;


    if (edge_entry_active){

        double kn[] = {k0, k1, k2, k3};
        double ks[] = {k0s, k1s, k2s, k3s};

        track_magnet_edge_particles(
            part0,
            edge_entry_model,
            0, // is_exit
            edge_entry_hgap,
            kn,
            ks,
            3, // k_order,
            knl,
            ksl,
            factor_knl_ksl_edge,
            order,
            length,
            edge_entry_angle,
            edge_entry_angle_fdown,
            edge_entry_fint,
            factor_backtrack_edge
        );
    }

    //start_per_particle_block (part0->part)
        track_magnet_body_single_particle(
            part, core_length, order, inv_factorial_order,
            knl, ksl,
            factor_knl_ksl_body,
            num_multipole_kicks, kick_rot_frame, drift_model, integrator,
            k0_drift, k1_drift, h_drift,
            k0_kick, k1_kick, h_kick, hxl,
            k0_h_correction, k1_h_correction,
            k2, k3, k0s, k1s, k2s, k3s,
            radiation_flag, radiation_record,
            &dp_record_exit, &dpx_record_exit, &dpy_record_exit
        );
    //end_per_particle_block

    if (edge_exit_active){
        double kn[] = {k0, k1, k2, k3};
        double ks[] = {k0s, k1s, k2s, k3s};

        track_magnet_edge_particles(
            part0,
            edge_exit_model,
            1, // is_exit
            edge_exit_hgap,
            kn,
            ks,
            3, // k_order,
            knl,
            ksl,
            factor_knl_ksl_edge,
            order,
            length,
            edge_exit_angle,
            edge_exit_angle_fdown,
            edge_exit_fint,
            factor_backtrack_edge
        );
    }

}

#endif