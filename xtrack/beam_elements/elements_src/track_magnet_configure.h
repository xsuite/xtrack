#ifndef XTRACK_TRACK_MAGNET_CONFIGURE_H
#define XTRACK_TRACK_MAGNET_CONFIGURE_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_magnet_structs.h"
#include "xtrack/beam_elements/elements_src/track_magnet_drift.h"
#include "xtrack/beam_elements/elements_src/default_magnet_config.h"

#define H_TOLERANCE (1e-8)


// Not XT_INLINE (nor MagnetKick_prepare): forced, the inlining grows the
// element frame on GPU and slows it (solenoid +6 %); the compiler inlines it
GPUFUN
MagnetSplit configure_tracking_model(
    int64_t model,
    xt_float_or_tpsa_arg k0,
    xt_float_or_tpsa_arg k1,
    double h,
    xt_float_or_tpsa_arg ks
){
    MagnetSplit split;

    // The models and drift maps: default_magnet_config.h

    if (model == MODEL_FULL) {
        model = MODEL_ROT_KICK_ROT; // backward compatibility
    }

    int8_t h_is_zero = (fabs(h) < H_TOLERANCE);
    int64_t drift_model = DRIFT_MAP_EXPANDED;
    split.do_drift_correction = 0;

    if (model == MODEL_BEND_KICK_BEND) {
        if (h_is_zero){
            drift_model = DRIFT_MAP_BEND_STRAIGHT;
        }
        else{
            drift_model = DRIFT_MAP_BEND;
        }
    }
    else if (model == MODEL_ROT_KICK_ROT) {
        if (h_is_zero){
            drift_model = DRIFT_MAP_EXACT;
        }
        else{
             drift_model = DRIFT_MAP_BEND_YOSHIDA4;
        }
    }
    else if (model == MODEL_MAT_KICK_MAT) {
        drift_model = DRIFT_MAP_DIPOLE_QUAD_EXPANDED;
    }
    else if (model == MODEL_MAT_KICK_MAT_EXACT) {
        drift_model = DRIFT_MAP_DIPOLE_QUAD_EXPANDED;
        split.do_drift_correction = 1;
    }
    else if (model == MODEL_DRIFT_KICK_DRIFT_EXACT) {
        drift_model = DRIFT_MAP_EXACT;
    }
    else if (model == MODEL_DRIFT_KICK_DRIFT_EXPANDED) {
        drift_model = DRIFT_MAP_EXPANDED;
    }
    else if (model == MODEL_KICK_ONLY) {
        drift_model = DRIFT_MAP_NONE;
    }
    else if (model == MODEL_SOL_KICK_SOL) {
        drift_model = DRIFT_MAP_SOLENOID;
    }
    else if (model == MODEL_ROT_KICK_ROT_LOW_ORDER) {
        if (h_is_zero){
            drift_model = DRIFT_MAP_EXACT;
        }
        else{
            drift_model = DRIFT_MAP_POLAR;
        }
    }
    else if (model == MODEL_ROT_KICK_ROT_HIGH_ORDER) {
        if (h_is_zero){
            drift_model = DRIFT_MAP_EXACT;
        }
        else{
            drift_model = DRIFT_MAP_BEND_YOSHIDA6;
        }
    }
    else{
        // This should never happen, but just in case
        drift_model = 99999999;
    }

    if (drift_model == DRIFT_MAP_NONE || drift_model == DRIFT_MAP_EXPANDED || drift_model == DRIFT_MAP_EXACT) {
        split.drift.k0 = 0.0;
        split.drift.k1 = 0.0;
        split.drift.h = 0.0;
        split.drift.ks = 0.0;
        split.k0_kick = k0;
        split.k1_kick = k1;
        split.h_kick = h;
        split.k0_h_correction = k0;
        split.k1_h_correction = k1;
        split.kick_rot_frame = 1;
    }
    else if (drift_model == DRIFT_MAP_POLAR) {
        split.drift.k0 = 0.0;
        split.drift.k1 = 0.0;
        split.drift.h = h;
        split.drift.ks = 0.0;
        split.k0_kick = k0;
        split.k1_kick = k1;
        split.h_kick = h;
        split.k0_h_correction = k0;
        split.k1_h_correction = k1;
        split.kick_rot_frame = 0;
    }
    else if (drift_model == DRIFT_MAP_DIPOLE_QUAD_EXPANDED) {
        split.drift.k0 = k0;
        split.drift.k1 = k1;
        split.drift.h = h;
        split.drift.ks = 0.0;
        split.k0_kick = 0.0;
        split.k1_kick = 0.0;
        split.h_kick = h;
        split.k0_h_correction = 0.;
        split.k1_h_correction = k1;
        split.kick_rot_frame = 0;
    }
    else if (drift_model == DRIFT_MAP_BEND) {
        split.drift.k0 = k0;
        split.drift.k1 = 0.0;
        split.drift.h = h;
        split.drift.ks = 0.0;
        split.k0_kick = 0.0;
        split.k1_kick = k1;
        split.h_kick = h;
        split.k0_h_correction = 0.;
        split.k1_h_correction = k1;
        split.kick_rot_frame = 0;
    }
    else if (drift_model == DRIFT_MAP_BEND_STRAIGHT) {
        split.drift.k0 = k0;
        split.drift.k1 = 0.0;
        split.drift.h = 0.0;
        split.drift.ks = 0.0;
        split.k0_kick = 0.0;
        split.k1_kick = k1;
        split.h_kick = 0.0;
        split.k0_h_correction = 0.;
        split.k1_h_correction = 0.;
        split.kick_rot_frame = 0;
    }
    else if (drift_model == DRIFT_MAP_SOLENOID) {
        split.drift.k0 = 0.0;
        split.drift.k1 = 0.0;
        split.drift.h = 0.0;
        split.drift.ks = ks;
        split.k0_kick = k0;
        split.k1_kick = k1;
        split.h_kick = h;
        split.k0_h_correction = k0;
        split.k1_h_correction = k1;
        split.kick_rot_frame = 1;
    }
    else if (drift_model == DRIFT_MAP_BEND_YOSHIDA4) {
        split.drift.k0 = k0;
        split.drift.k1 = 0.0;
        split.drift.h = h;
        split.drift.ks = 0.0;
        split.k0_kick = 0.0;
        split.k1_kick = k1;
        split.h_kick = h;
        split.k0_h_correction = k0;
        split.k1_h_correction = k1;
        split.kick_rot_frame = 0;
    }
    else if (drift_model == DRIFT_MAP_BEND_YOSHIDA6) {
        split.drift.k0 = k0;
        split.drift.k1 = 0.0;
        split.drift.h = h;
        split.drift.ks = 0.0;
        split.k0_kick = 0.0;
        split.k1_kick = k1;
        split.h_kick = h;
        split.k0_h_correction = k0;
        split.k1_h_correction = k1;
        split.kick_rot_frame = 0;
    }

    split.drift.model = drift_model;
    split.drift.x0_solenoid = 0.;
    split.drift.y0_solenoid = 0.;
    return split;
}

#undef H_TOLERANCE

#endif // XTRACK_TRACK_MAGNET_CONFIGURE_H
