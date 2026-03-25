#ifndef XTRACK_TRACK_MAGNET_CONFIGURE_H
#define XTRACK_TRACK_MAGNET_CONFIGURE_H

#define H_TOLERANCE (1e-8)

GPUFUN
void configure_tracking_model(
    int64_t model,
    double k0,
    double k1,
    double h,
    double ks,
    double* k0_drift,
    double* k1_drift,
    double* h_drift,
    double* ks_drift,
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
    // model = -2: sol-kick-sol (not exposed in python)

    if (model==1){
        model = 3; // backward compatibility
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
    else if(model == -1){ // kick only
        drift_model = -1;
    }
    else if(model == -2){ // sol-kick-sol
        drift_model = 6; // solenoid
    }
    else{
        // This should never happen, but just in case
        drift_model = 99999999;
    }

    if (drift_model == -1 || drift_model == 0 || drift_model == 1){ // drift expanded, drift exact, kick only
        *k0_drift = 0.0;
        *k1_drift = 0.0;
        *h_drift = 0.0;
        *ks_drift = 0.0;
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
        *ks_drift = 0.0;
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
        *ks_drift = 0.0;
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
        *ks_drift = 0.0;
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
        *ks_drift = 0.0;
        *k0_kick = 0.0;
        *k1_kick = k1;
        *h_kick = 0.0;
        *k0_h_correction = 0.;
        *k1_h_correction = 0.;
        *kick_rot_frame = 0;
    }
    else if (drift_model == 6){ // solenoid
        *k0_drift = 0.0;
        *k1_drift = 0.0;
        *h_drift = 0.0;
        *ks_drift = ks;
        *k0_kick = k0;
        *k1_kick = k1;
        *h_kick = h;
        *k0_h_correction = k0;
        *k1_h_correction = k1;
        *kick_rot_frame = 1;
    }


    *out_drift_model = drift_model;
}

#undef H_TOLERANCE

#endif // XTRACK_TRACK_MAGNET_CONFIGURE_H