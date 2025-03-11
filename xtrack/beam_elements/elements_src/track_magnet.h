// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRACK_MAGNET_H
#define XTRACK_TRACK_MAGNET_H

H_TOLERANCE = 1e-8

/*gpufun*/
void configure_tracking_model(
    int64_t model,
    double k0,
    double k1,
    double h,
    double* out_k0_drift,
    double* out_k1_drift,
    double* out_h_drift,
    double* out_k0_kick,
    double* out_k1_kick,
    double* out_h_kick,
    uint8_t* out_kick_rot_frame,
    uint8_t* out_drift_model
){


    // model = 0 or 1 : adaptive
    // model = 2: bend-kick-bend
    // model = 3: rot-kick-rot
    // model = 4: mat-kick-mat (previously called `expanded`)
    // model = 5: drift-kick-drift-exact
    // model = 6: drift-kick-drift-expanded

    if (model==0 || model==1){
        model = 3;
    }

    uint8_t h_is_zero = (fabs(h) < H_TOLERANCE);
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

    double k0_drift, k1_drift, h_drift;
    double k0_kick, k1_kick, h_kick;
    uint8_t kick_rot_frame;

    if (drift_model == 0 || drift_model == 1){ // drift expanded or drift exact
        k0_drift = 0.0;
        k1_drift = 0.0;
        h_drift = 0.0;
        k0_kick = k0;
        k1_kick = k1;
        h_kick = h;
        kick_rot_frame = 1;
    }
    else if (drift_model == 2){ // polar drift
        k0_drift = 0.0;
        k1_drift = 0.0;
        h_drift = 0.0;
        k0_kick = k0;
        k1_kick = k1;
        h_kick = h;
        kick_rot_frame = 0;
    }
    else if (drift_model == 3){ // expanded dipole-quadrupole
        k0_drift = k0;
        k1_drift = k1;
        h_drift = h;
        k0_kick = 0.0;
        k1_kick = 0.0;
        h_kick = 0.0;
        kick_rot_frame = 0;
    }
    else if (drift_model == 4){ // bend with h
        k0_drift = k0;
        k1_drift = k1;
        h_drift = h;
        k0_kick = 0.0;
        k1_kick = 0.0;
        h_kick = 0.0;
        kick_rot_frame = 0;
    }
    else if (drift_model == 5){ // bend without h
        k0_drift = k0;
        k1_drift = k1;
        h_drift = 0.0;
        k0_kick = 0.0;
        k1_kick = 0.0;
        h_kick = 0.0;
        kick_rot_frame = 0;
    }

    *out_k0_drift = k0_drift;
    *out_k1_drift = k1_drift;
    *out_h_drift = h_drift;
    *out_k0_kick = k0_kick;
    *out_k1_kick = k1_kick;
    *out_h_kick = h_kick;
    *out_kick_rot_frame = kick_rot_frame;
    *out_drift_model = drift_model;

}


/*gpufun*/
void track_magnet_body_single_particle(
    LocalParticle* part,
    double length,
    int64_t order,
    double inv_factorial_order,
    /*gpuglmem*/ const double* knl,
    /*gpuglmem*/ const double* ksl,
    double const factor_knl_ksl,
    int64_t num_multipole_kicks,
    uint8_t kick_rot_frame,
    uint8_t drift_model,
    double k0_drift,
    double k1_drift,
    double h_drift,
    double k0_kick,
    double k1_kick,
    double h_kick,
    double k2,
    double k3,
    double k0s,
    double k1s,
    double k2s,
    double k3s,
) {

    #define MAGNET_KICK(part, weight) \
        track_magnet_kick_single_particle(\
            part, length * (weight), order, inv_factorial_order, \
            knl, ksl, factor_knl_ksl, (weight), \
            k0_kick, k1_kick, k2, k3, k0s, k1s, k2s, k3s, h_kick, kick_rot_frame\
        )

    #define MAGNET_DRIFT(part, weight) \
        track_magnet_drift_single_particle(\
            part, length * (weight), k0_drift, k1_drift, h_drift, drift_model\
        )

    if (num_multipole_kicks == 0) { // auto mode
        num_multipole_kicks = 1;
    }
    const double kick_weight = 1. / num_multipole_kicks;
    double edge_drift_weight = 0.5;
    double inside_drift_weight = 0;
    if (num_multipole_kicks > 1) {
        edge_drift_weight = 1. / (2 * (1 + num_multipole_kicks));
        inside_drift_weight = (
            ((float) num_multipole_kicks)
                / ((float)(num_multipole_kicks*num_multipole_kicks) - 1));
    }

    // TEAPOT body
    MAGNET_DRIFT(part, edge_drift_weight);
    for (int i_kick=0; i_kick<num_multipole_kicks - 1; i_kick++) {
        MAGNET_KICK(part, kick_weight);
        MAGNET_DRIFT(part, inside_drift_weight);
    }
    MAGNET_KICK(part, kick_weight);
    MAGNET_DRIFT(part, edge_drift_weight);

    #undef MAGNET_KICK
    #undef MAGNET_DRIFT

}

/*gpufun*/
void track_magnet_body_particles(
    LocalParticle* part0,
    double length,
    int64_t order,
    double inv_factorial_order,
    /*gpuglmem*/ const double* knl,
    /*gpuglmem*/ const double* ksl,
    double const factor_knl_ksl,
    int64_t num_multipole_kicks,
    uint8_t model,
    double h,
    double k0,
    double k1,
    double k2,
    double k3,
    double k0s,
    double k1s,
    double k2s,
    double k3s,
) {

    double k0_drift, k1_drift, h_drift;
    double k0_kick, k1_kick, h_kick;
    uint8_t kick_rot_frame;
    uint8_t drift_model;

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
        &kick_rot_frame,
        &drift_model
    );

    //start_per_particle_block (part0->part)
        track_magnet_body_single_particle(
            part, length, order, inv_factorial_order,
            knl, ksl, factor_knl_ksl,
            num_multipole_kicks, kick_rot_frame, drift_model,
            k0_drift, k1_drift, h_drift,
            k0_kick, k1_kick, h_kick,
            k2, k3, k0s, k1s, k2s, k3s
        );
    //end_per_particle_block

}

#endif