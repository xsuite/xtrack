// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_TRACK_MAGNET_H
#define XTRACK_TRACK_MAGNET_H

H_TOLERANCE = 1e-8

/*gpufun*/
void track_magnet_drift_single_particle(
    LocalParticle* part,
    double length,
    int64_t order,
    double inv_factorial_order,
    /*gpuglmem*/ const double* knl,
    /*gpuglmem*/ const double* ksl,
    double const factor_knl_ksl,
    double kick_weight,
    double k0,
    double k1,
    double k2,
    double k3,
    double k0s,
    double k1s,
    double k2s,
    double k3s,
    double h,
    int64_t model
) {


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







}



#endif