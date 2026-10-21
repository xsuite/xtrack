// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MAGNET_H
#define XTRACK_TRACK_MAGNET_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_magnet_kick.h"
#include "xtrack/beam_elements/elements_src/track_magnet_drift.h"
#include "xtrack/beam_elements/elements_src/track_magnet_edge.h"
#include "xtrack/beam_elements/elements_src/edge_flags.h"
#include "xtrack/beam_elements/elements_src/track_magnet_configure.h"
#include "xtrack/beam_elements/elements_src/integrator.h"

#ifndef XTRACK_MULTIPOLE_NO_SYNRAD
#include "xtrack/beam_elements/elements_src/track_magnet_radiation.h"
#endif

#define H_TOLERANCE (1e-8)
// Highest multipole order of a magnet (as in MAD-X), checked in Python
#define MAGNET_MAX_MULTIPOLE_ORDER 20


// Order of knl + rel_ref_strength * knl_rel: the order of knl, or the highest
// nonzero relative component above it
GPUFUN
int64_t magnet_multipole_order(
    const int64_t order,
    const int64_t order_rel,
    GPUGLMEM const double* knl_rel,
    GPUGLMEM const double* ksl_rel
) {
    if (knl_rel == NULL) return order;
    for (int64_t ii = order_rel; ii > order; ii--) {
        if (knl_rel[ii] != 0 || ksl_rel[ii] != 0) return ii;
    }
    return order;
}

// knl + rel_ref_strength * knl_rel (same for ksl), orders 0 ... order_tot,
// into knl_out, ksl_out
GPUFUN
void magnet_total_multipoles(
    const int64_t order,
    GPUGLMEM const double* knl,
    GPUGLMEM const double* ksl,
    const int64_t order_rel,
    GPUGLMEM const double* knl_rel,
    GPUGLMEM const double* ksl_rel,
    xt_float_or_tpsa_arg rel_ref_strength,
    const int64_t order_tot,
    xt_float_or_tpsa* knl_out,
    xt_float_or_tpsa* ksl_out
) {
    for (int64_t ii = 0; ii <= order_tot; ii++) {
        knl_out[ii] = ii <= order ? knl[ii] : 0.;
        ksl_out[ii] = ii <= order ? ksl[ii] : 0.;
    }
    if (knl_rel == NULL) return;
    for (int64_t ii = 0; ii <= order_rel && ii <= order_tot; ii++) {
        if (knl_rel[ii] != 0) knl_out[ii] += rel_ref_strength * knl_rel[ii];
        if (ksl_rel[ii] != 0) ksl_out[ii] += rel_ref_strength * ksl_rel[ii];
    }
}

#ifndef VSWAP
    #define VSWAP(a, b)     \
        {                   \
            double tmp = a; \
            a = b;          \
            b = tmp;        \
        }
#endif

// Body of a magnet. The settings are given by the element;
// MagnetBody_prepare derives, from them and the field, how the body is
// integrated (the split of the field between drifts and kicks, see
// configure_tracking_model, and the number of kicks), once per element call,
// so that the per-particle map only reads it.
typedef struct MagnetBody_s {
    // Settings
    int8_t model, default_model;
    int8_t integrator, default_integrator;
    int64_t num_multipole_kicks;
    double weight;               // fraction of the element (slices)
    double length;               // signed for the direction of travel
    double inv_factorial_order;  // of the order of the field
    double factor_knl_ksl;       // multiplies knl, ksl
    double h, hxl;
    double dks_ds;
    int64_t radiation_flag;
    int64_t spin_flag;
    SynchrotronRadiationRecordData radiation_record;

    // Derived by MagnetBody_prepare
    const MagnetField* field;  // strengths, read through it (no copies)
    int8_t kick_integrator;    // integrator and number of kicks resolved
    int64_t num_kicks;         // (defaults, kick only model, auto mode)
    int8_t drift_model;
    xt_float_or_tpsa k0_drift, k1_drift, ks_drift;
    double h_drift;
    int8_t kick_rot_frame;
    xt_float_or_tpsa k0_kick, k1_kick;
    double h_kick;
    uint8_t do_drift_correction;
    xt_float_or_tpsa k0_h_correction, k1_h_correction;
    uint8_t kick_is_empty;  // all multipole and curvature terms vanish
#ifdef XTRACK_TPSA_TRACK
    // The TPSA type has no default constructor
    MagnetBody_s()
        : k0_drift(0.), k1_drift(0.), ks_drift(0.), k0_kick(0.), k1_kick(0.),
          k0_h_correction(0.), k1_h_correction(0.) {}
#endif
} MagnetBody;


GPUFUN
void track_magnet_body_single_particle(
    LocalParticle* part,
    const MagnetBody* body,
    double* dp_record_exit,
    double* dpx_record_exit,
    double* dpy_record_exit
) {
    const double length = body->length;
    const MagnetField* field = body->field;
    const int64_t order = field->kl_order;
    const double inv_factorial_order = body->inv_factorial_order;
    const xt_float_or_tpsa* knl = field->knl;
    const xt_float_or_tpsa* ksl = field->ksl;
    const double factor_knl_ksl = body->factor_knl_ksl;
    const int64_t num_multipole_kicks = body->num_kicks;
    const int8_t kick_rot_frame = body->kick_rot_frame;
    const int8_t drift_model = body->drift_model;
    const uint8_t do_drift_correction = body->do_drift_correction;
    const uint8_t kick_is_empty = body->kick_is_empty;
    const int8_t integrator = body->kick_integrator;
    xt_float_or_tpsa_arg k0_drift = body->k0_drift;
    xt_float_or_tpsa_arg k1_drift = body->k1_drift;
    xt_float_or_tpsa_arg ks_drift = body->ks_drift;
    const double h_drift = body->h_drift;
    xt_float_or_tpsa_arg k0_kick = body->k0_kick;
    xt_float_or_tpsa_arg k1_kick = body->k1_kick;
    const double h_kick = body->h_kick;
    const double hxl = body->hxl;
    xt_float_or_tpsa_arg k0_h_correction = body->k0_h_correction;
    xt_float_or_tpsa_arg k1_h_correction = body->k1_h_correction;
    xt_float_or_tpsa_arg k2 = *field->knorm[2];
    xt_float_or_tpsa_arg k3 = *field->knorm[3];
    xt_float_or_tpsa_arg k0s = *field->kskew[0];
    xt_float_or_tpsa_arg k1s = *field->kskew[1];
    xt_float_or_tpsa_arg k2s = *field->kskew[2];
    xt_float_or_tpsa_arg k3s = *field->kskew[3];
    const double dks_ds = body->dks_ds;
    const double x0_solenoid = field->x0_solenoid;
    const double y0_solenoid = field->y0_solenoid;
    const int64_t radiation_flag = body->radiation_flag;
    const int64_t spin_flag = body->spin_flag;
    SynchrotronRadiationRecordData radiation_record = body->radiation_record;
    // Unused in the kernels without radiation, spin or solenoid field
    (void)dks_ds;
    (void)radiation_flag;
    (void)spin_flag;
    (void)radiation_record;


#define MAGNET_KICK(part, weight)      \
    track_magnet_kick_single_particle( \
        part,                          \
        length,                        \
        order,                         \
        inv_factorial_order,           \
        knl,                           \
        ksl,                           \
        factor_knl_ksl,                \
        (weight),                      \
        k0_kick,                       \
        k1_kick,                       \
        k2,                            \
        k3,                            \
        k0s,                           \
        k1s,                           \
        k2s,                           \
        k3s,                           \
        h_kick,                        \
        hxl,                           \
        k0_h_correction,               \
        k1_h_correction,               \
        kick_rot_frame,                \
        do_drift_correction,           \
        kick_is_empty                  \
    )

#define MAGNET_DRIFT(part, dlength)     \
    track_magnet_drift_single_particle( \
        part,                           \
        (dlength),                      \
        k0_drift,                       \
        k1_drift,                       \
        ks_drift,                       \
        h_drift,                        \
        x0_solenoid,                    \
        y0_solenoid,                    \
        drift_model                     \
    )

#if defined(XTRACK_TPSA_TRACK) || defined(XTRACK_MULTIPOLE_NO_SYNRAD)
    #define WITH_RADIATION(ll, code) \
        {                            \
            code;                    \
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

    if (num_multipole_kicks == 0 && xt_float_or_tpsa_const_part(k0_kick) == 0
            && xt_float_or_tpsa_const_part(k1_kick) == 0 && h_kick == 0) { //only drift
        WITH_RADIATION(length,
            MAGNET_DRIFT(part, length);
        )
    }
    else{

        RUN_INTEGRATOR(
            integrator, length, num_multipole_kicks, part,
            MAGNET_DRIFT, MAGNET_KICK, WITH_RADIATION
        );

    }

    #undef MAGNET_KICK
    #undef MAGNET_DRIFT
    #undef WITH_RADIATION
}


// Straight-body RBend: the body is a straight line and the curvature is in
// the patches of the faces. At the entry face the reference orbit makes the
// angle theta_entry = (angle - angle_diff) / 2 with the body and is offset by
// dx_entry from it, at the exit face theta_exit = (angle + angle_diff) / 2 and
// dx_exit. The body is shifted by -rbend_shift, and by half the sagitta if
// compensate_sagitta. The linear fringe is the expansion around the angle of
// the orbit at the face, hence the feed-down angles.
GPUFUN
void rbend_straight_body_patches(
    const double angle,
    const double h,
    const double length_straight,
    const double angle_diff,
    const double rbend_shift,
    const int64_t compensate_sagitta,
    MagnetFace* entry_face,
    MagnetFace* exit_face
) {
    const double theta_entry = (angle - angle_diff) / 2.0;
    const double theta_exit = (angle + angle_diff) / 2.0;
    double cos_theta_entry = 1., sin_theta_exit = 0., cos_theta_exit = 1.;
    if (fabs(theta_entry) > 1e-10) {
        cos_theta_entry = cos(theta_entry);
    }
    if (fabs(theta_exit) > 1e-10) {
        sin_theta_exit = sin(theta_exit);
        cos_theta_exit = cos(theta_exit);
    }

    double dx_mid = 0. - rbend_shift;
    if (compensate_sagitta && fabs(angle) > 1e-10) {
        // Half the sagitta, via 1 - cos(u) = 2*sin(u/2)^2 to avoid the
        // cancellation that the large 1/h would amplify.
        double const sin_rbqa = sin(angle / 4.);
        dx_mid += sin_rbqa * sin_rbqa / h;
    }

    double dx_entry = dx_mid;
    double dx_exit = dx_mid;
    if (fabs(angle) > 1e-10) {
        double const px0_entry = sin(theta_entry);
        double const px0_mid = px0_entry - h * length_straight / 2;
        double const sqrt_mid = sqrt(1 - px0_mid * px0_mid);
        // Rationalised forms of (1/h)*(sqrt_mid - cos_theta_entry) and
        // (1/h)*(cos_theta_exit - sqrt_mid), so that h cancels exactly.
        dx_entry -=
            0.5 * length_straight * (px0_entry + px0_mid) / (sqrt_mid + cos_theta_entry);
        double const term_exit_over_h =
            2 * cos(angle / 2.) * sin(angle_diff / 2.) / h + length_straight / 2.;
        dx_exit -=
            term_exit_over_h * (px0_mid + sin_theta_exit) / (cos_theta_exit + sqrt_mid);
    }

    entry_face->angle_fdown += theta_entry;
    exit_face->angle_fdown += theta_exit;
    entry_face->has_patch = 1;
    entry_face->patch_angle = theta_entry;
    entry_face->patch_dx = dx_entry;
    exit_face->has_patch = 1;
    exit_face->patch_angle = theta_exit;
    exit_face->patch_dx = dx_exit;
}


// Derive the integration of the body from its settings and the field of the
// element (the strengths of the faces, see track_magnet_edge.h; they need not
// be prepared).
GPUFUN
void MagnetBody_prepare(MagnetBody* body, const MagnetField* field) {
    int8_t model = body->model;
    int8_t integrator = body->integrator;
    int64_t num_multipole_kicks = body->num_multipole_kicks;
    const double weight = body->weight;
    const double body_length = body->length;
    const double h = body->h;
    const double hxl = body->hxl;
    const int64_t order = field->kl_order;
    const xt_float_or_tpsa* knl = field->knl;
    const xt_float_or_tpsa* ksl = field->ksl;
    xt_float_or_tpsa_arg k0 = *field->knorm[0];
    xt_float_or_tpsa_arg k1 = *field->knorm[1];
    xt_float_or_tpsa_arg k2 = *field->knorm[2];
    xt_float_or_tpsa_arg k3 = *field->knorm[3];
    xt_float_or_tpsa_arg k0s = *field->kskew[0];
    xt_float_or_tpsa_arg k1s = *field->kskew[1];
    xt_float_or_tpsa_arg k2s = *field->kskew[2];
    xt_float_or_tpsa_arg k3s = *field->kskew[3];
    xt_float_or_tpsa_arg ks = *field->ks;

    body->field = field;

    if (integrator == 0) {
        integrator = body->default_integrator;
    }
    if (model == 0) {
        model = body->default_model;
    }
    if (model == -1) {   // kick only
        integrator = 3;  // uniform
        num_multipole_kicks = 1;
    }

    // Adjust the number of multipole kicks based on the weight
    if (weight != 1.0 && num_multipole_kicks > 0) {
        num_multipole_kicks = (int64_t)ceil(num_multipole_kicks * weight);
    }

    configure_tracking_model(
        model,
        k0,
        k1,
        h,
        ks,
        &body->k0_drift,
        &body->k1_drift,
        &body->h_drift,
        &body->ks_drift,
        &body->k0_kick,
        &body->k1_kick,
        &body->h_kick,
        &body->k0_h_correction,
        &body->k1_h_correction,
        &body->kick_rot_frame,
        &body->drift_model,
        &body->do_drift_correction
    );

    // Compute the number of kicks for auto mode
    if (num_multipole_kicks == 0) {  // num_multipole_kicks = 0 means auto mode
        uint8_t const kick_inactive =
            kick_is_inactive(order, knl, ksl, k0, k1, k2, k3, k0s, k1s, k2s, k3s, h);

        // The exact matrix model needs a path-correction kick even at zero field.
        // Otherwise, guess the number of active kicks or perform only the drift.
        if (body->do_drift_correction || !kick_inactive) {
            if (fabs(h) < 1e-8) {
                num_multipole_kicks =
                    1;  // straight magnet, one multipole kick in the middle
            }
            else {
                double b_circum = 2 * 3.14159 / fabs(h);
                num_multipole_kicks = fabs(body_length) / b_circum
                                      / 0.5e-3;  // 0.5 mrad per kick (on average)
                if (num_multipole_kicks < 1) {
                    num_multipole_kicks = 1;
                }
            }
        }
    }

    body->kick_is_empty = (hxl == 0.0)
                          && kick_is_inactive(
                              order,
                              knl,
                              ksl,
                              body->k0_kick,
                              body->k1_kick,
                              k2,
                              k3,
                              k0s,
                              k1s,
                              k2s,
                              k3s,
                              body->h_kick
                          );

    body->kick_integrator = integrator;
    body->num_kicks = num_multipole_kicks;
}


// Settings of a magnet (or of a slice of it) for track_magnet_particles,
// filled by the element; the faces are given separately (MagnetFace).
typedef struct MagnetSettings_s {
    double weight;
    double length;
    int64_t order;
    double inv_factorial_order;
    GPUGLMEM const double* knl;
    GPUGLMEM const double* ksl;
    int64_t order_rel;
    GPUGLMEM const double* knl_rel;
    GPUGLMEM const double* ksl_rel;
    xt_float_or_tpsa rel_ref_strength;  // knl_rel, ksl_rel: relative to it
    int64_t num_multipole_kicks;
    int8_t model;
    int8_t default_model;
    int8_t integrator;
    int8_t default_integrator;
    int64_t radiation_flag;
    int64_t radiation_flag_parent;
    SynchrotronRadiationRecordData radiation_record;
    double delta_taper;
    double h;
    double hxl;
    xt_float_or_tpsa k0;
    xt_float_or_tpsa k1;
    xt_float_or_tpsa k2;
    xt_float_or_tpsa k3;
    xt_float_or_tpsa k0s;
    xt_float_or_tpsa k1s;
    xt_float_or_tpsa k2s;
    xt_float_or_tpsa k3s;
    xt_float_or_tpsa ks;
    double dks_ds;
    double x0_solenoid;
    double y0_solenoid;
    int64_t rbend_model;  // -1: not used, 0: auto, 1: curved body, 2: straight body
    int64_t rbend_compensate_sagitta;
    double rbend_shift;
    double rbend_angle_diff;
    double length_straight;
    int64_t body_active;
#ifdef XTRACK_TPSA_TRACK
    // The TPSA type has no default constructor
    MagnetSettings_s()
        : rel_ref_strength(0.), k0(0.), k1(0.), k2(0.), k3(0.), k0s(0.), k1s(0.), k2s(0.),
          k3s(0.), ks(0.) {}
#endif
} MagnetSettings;


// Settings of a magnet that has nothing beyond a straight body: no field,
// no multipoles, default model and integrator, no radiation, not an RBend.
// Fills the given struct in place (no copy: copying a TPSA allocates; the
// TPSA members are only assigned scalars here). Each element then sets the
// fields it has and lists the others under "Kept at default".
GPUFUN
void MagnetSettings_init_default(MagnetSettings* s) {
    s->weight = 1.;
    s->length = 0.;
    s->order = -1;
    s->inv_factorial_order = 1.;
    s->knl = NULL;
    s->ksl = NULL;
    s->order_rel = -1;
    s->knl_rel = NULL;
    s->ksl_rel = NULL;
    s->rel_ref_strength = 0.;
    s->num_multipole_kicks = 0;
    s->model = 0;
    s->default_model = 0;
    s->integrator = 0;
    s->default_integrator = 0;
    s->radiation_flag = 0;
    s->radiation_flag_parent = 0;
    s->radiation_record = NULL;
    s->delta_taper = 0.;
    s->h = 0.;
    s->hxl = 0.;
    s->k0 = 0.;
    s->k1 = 0.;
    s->k2 = 0.;
    s->k3 = 0.;
    s->k0s = 0.;
    s->k1s = 0.;
    s->k2s = 0.;
    s->k3s = 0.;
    s->ks = 0.;
    s->dks_ds = 0.;
    s->x0_solenoid = 0.;
    s->y0_solenoid = 0.;
    s->rbend_model = -1;  // not an RBend
    s->rbend_compensate_sagitta = 0;
    s->rbend_shift = 0.;
    s->rbend_angle_diff = 0.;
    s->length_straight = 0.;
    s->body_active = 1;
}


// Track a magnet, or a slice of it (weight: its fraction of the length),
// through its first face, its body and its second face, in the order of the
// direction of travel. The faces are given in the element frame.
GPUFUN
void track_magnet_particles(
    LocalParticle* part0,
    MagnetSettings* settings,  // the strengths are scaled in place (tapering)
    const MagnetFace* entry_face_in,
    const MagnetFace* exit_face_in
) {
    // Values changed below (RBend, backtracking, tapering) are local copies;
    // the rest is read from settings
    const double weight = settings->weight;
    double length = settings->length;
    double h = settings->h;
    double hxl = settings->hxl;
    int64_t rbend_model = settings->rbend_model;
    int64_t radiation_flag = settings->radiation_flag;
    double delta_taper = settings->delta_taper;
    (void)delta_taper;  // unused in the kernels without radiation

    const uint8_t backtrack = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);

    // Faces, in the element frame; the RBend changes their angles or adds
    // patches
    MagnetFace entry_face = *entry_face_in;
    MagnetFace exit_face = *exit_face_in;

    // RBend: the faces of a curved body are at the angle of the reference
    // orbit; a straight body follows the chord, and its length and the
    // length of the curved reference path differ
    double length_curved = 0.;
    if (rbend_model == 0){
        // auto mode, curved body
        rbend_model = 1;
    }
    double const angle = h * length;
    if (rbend_model == 1){
        entry_face.angle += (angle - settings->rbend_angle_diff) / 2.0;
        exit_face.angle += (angle + settings->rbend_angle_diff) / 2.0;
    }
    else if (rbend_model == 2){
        rbend_straight_body_patches(
            angle,
            h,
            settings->length_straight,
            settings->rbend_angle_diff,
            settings->rbend_shift,
            settings->rbend_compensate_sagitta,
            &entry_face,
            &exit_face
        );
        length_curved = length;
        length = settings->length_straight;
        h = 0; // treat magnet as straight
    }

    // Direction of travel: backtracking runs the body backwards; the faces
    // invert their own maps
    double knl_scale = 1.0;
    double body_length, body_length_curved, knl_scale_body;
    if (backtrack) {
        body_length = -length * weight;
        body_length_curved = -length_curved * weight;
        knl_scale_body = -knl_scale * weight;
        hxl = -hxl;
    }
    else {
        body_length = length * weight;
        body_length_curved = length_curved * weight;
        knl_scale_body = knl_scale * weight;
    }

    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        if (radiation_flag == 10){ // from parent
            radiation_flag = settings->radiation_flag_parent;
        }
    #endif

        // Tapering: all strengths scale with 1 + delta_taper
#ifndef XTRACK_TPSA_TRACK
    // delta_taper is a double param, get_delta is xt_float_or_tpsa. Tapering is a radiation-
    // adjacent double feature, so it is disabled for the TPSA flavor.
    if (LocalParticle_check_track_flag(part0, XS_FLAG_SR_TAPER)){
        part0->ipart = 0;
        delta_taper = LocalParticle_get_delta(part0); // I can use part0 because
                                                      // there is only one particle
                                                      // when doing the tapering
    }
#endif
    #ifndef XTRACK_MULTIPOLE_NO_SYNRAD
        if (radiation_flag){
            // knl and ksl are scaled by the called functions below using knl_scale
            knl_scale_body *= (1. + delta_taper);
            knl_scale *= (1. + delta_taper);
            settings->k0 *= (1 + delta_taper);
            settings->k1 *= (1 + delta_taper);
            settings->k2 *= (1 + delta_taper);
            settings->k3 *= (1 + delta_taper);
            settings->k0s *= (1 + delta_taper);
            settings->k1s *= (1 + delta_taper);
            settings->k2s *= (1 + delta_taper);
            settings->k3s *= (1 + delta_taper);
        }
    #endif

        // Multipoles of the body and the faces, formed once here (a local copy:
        // a few entries per element call)
        const int64_t order_tot = magnet_multipole_order(
            settings->order, settings->order_rel, settings->knl_rel, settings->ksl_rel
        );
        double inv_factorial_order_tot = settings->inv_factorial_order;
        for (int64_t ii = settings->order + 1; ii <= order_tot; ii++) {
            inv_factorial_order_tot /= ii;
        }
#ifdef XTRACK_TPSA_TRACK
        // The TPSA type has no default constructor: lift zero-initialised storage
        const double zeros[MAGNET_MAX_MULTIPOLE_ORDER + 1] = {0};
        xt_tpsa_lifted_array knl_lifted(zeros, order_tot + 1);
        xt_tpsa_lifted_array ksl_lifted(zeros, order_tot + 1);
        xt_float_or_tpsa* knl_tot = knl_lifted.mutable_ptr();
        xt_float_or_tpsa* ksl_tot = ksl_lifted.mutable_ptr();
#else
    double knl_tot[MAGNET_MAX_MULTIPOLE_ORDER + 1];
    double ksl_tot[MAGNET_MAX_MULTIPOLE_ORDER + 1];
#endif
        magnet_total_multipoles(
            settings->order,
            settings->knl,
            settings->ksl,
            settings->order_rel,
            settings->knl_rel,
            settings->ksl_rel,
            settings->rel_ref_strength,
            order_tot,
            knl_tot,
            ksl_tot
        );

        // Faces, in the order they are crossed: the particle enters the field at
        // the first and leaves it at the second, in both directions of travel
        MagnetField field;
        field.knorm[0] = &settings->k0;
        field.knorm[1] = &settings->k1;
        field.knorm[2] = &settings->k2;
        field.knorm[3] = &settings->k3;
        field.kskew[0] = &settings->k0s;
        field.kskew[1] = &settings->k1s;
        field.kskew[2] = &settings->k2s;
        field.kskew[3] = &settings->k3s;
        field.k_order = MAGNET_K_ORDER;
        field.knl = knl_tot;
        field.ksl = ksl_tot;
        field.kl_order = order_tot;
        field.knl_scale = knl_scale;
        field.length = length;
        field.ks = &settings->ks;
        field.x0_solenoid = settings->x0_solenoid;
        field.y0_solenoid = settings->y0_solenoid;

        MagnetFace* first_face = backtrack ? &exit_face : &entry_face;
        MagnetFace* second_face = backtrack ? &entry_face : &exit_face;
        if ((entry_face.flags | exit_face.flags) & EDGE_ACTIVE) {
            MagnetField_prepare(&field);
        }
        MagnetFace_prepare(first_face, &field, backtrack);
        MagnetFace_prepare(second_face, &field, backtrack);

        // Faces only (edge slices)
        if (!settings->body_active) {
            START_PER_PARTICLE_BLOCK(part0, part)
                ;
                MagnetFace_enter_single_particle(part, first_face);
                MagnetFace_leave_single_particle(part, second_face);
            END_PER_PARTICLE_BLOCK;
            return;
        }

        // Body
        MagnetBody body;
        body.model = settings->model;
        body.default_model = settings->default_model;
        body.integrator = settings->integrator;
        body.default_integrator = settings->default_integrator;
        body.num_multipole_kicks = settings->num_multipole_kicks;
        body.weight = weight;
        body.length = body_length;
        body.inv_factorial_order = inv_factorial_order_tot;
        body.factor_knl_ksl = knl_scale_body;
        body.h = h;
        body.hxl = hxl;
        body.dks_ds = settings->dks_ds;
        body.radiation_flag = radiation_flag;
        body.spin_flag = 1;
        body.radiation_record = settings->radiation_record;
        MagnetBody_prepare(&body, &field);

        // Straight body: the path length is measured on the curved reference
        // (not for the kick only model, model -1)
        const uint8_t correct_s = (rbend_model == 2 && settings->model >= 0);
        const double ds_curved = body_length_curved - body_length;

        double dp_record_exit, dpx_record_exit, dpy_record_exit;

        START_PER_PARTICLE_BLOCK(part0, part)
            ;
            MagnetFace_enter_single_particle(part, first_face);
            track_magnet_body_single_particle(
                part, &body, &dp_record_exit, &dpx_record_exit, &dpy_record_exit
            );
            if (correct_s) {
                LocalParticle_add_to_s(part, ds_curved);
                LocalParticle_add_to_zeta(part, ds_curved);
            }
            MagnetFace_leave_single_particle(part, second_face);
        END_PER_PARTICLE_BLOCK;
}

#endif
