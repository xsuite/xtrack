// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MAGNET_H
#define XTRACK_TRACK_MAGNET_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_magnet_structs.h"
#include "xtrack/beam_elements/elements_src/track_magnet_kick.h"
#include "xtrack/beam_elements/elements_src/track_magnet_drift.h"
#include "xtrack/beam_elements/elements_src/track_magnet_edge.h"
#include "xtrack/beam_elements/elements_src/edge_flags.h"
#include "xtrack/beam_elements/elements_src/track_magnet_configure.h"
#include "xtrack/beam_elements/elements_src/integrator.h"
#include "xtrack/beam_elements/elements_src/track_misalignments.h"

#ifndef XTRACK_MULTIPOLE_NO_SYNRAD
#include "xtrack/beam_elements/elements_src/track_magnet_radiation.h"
#endif

#define H_TOLERANCE (1e-8)


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

// Whether knl_rel or ksl_rel has a nonzero component (if not, order_tot is
// the order of knl)
GPUFUN
uint8_t magnet_has_rel(
    const int64_t order_rel,
    GPUGLMEM const double* knl_rel,
    GPUGLMEM const double* ksl_rel
) {
    if (knl_rel == NULL) return 0;
    for (int64_t ii = 0; ii <= order_rel; ii++) {
        if (knl_rel[ii] != 0 || ksl_rel[ii] != 0) return 1;
    }
    return 0;
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


#if !defined(XTRACK_TPSA_TRACK) && !defined(XTRACK_MULTIPOLE_NO_SYNRAD)
// Spin and radiation over a step of length ll of the body, from the
// coordinates before it (old_*) and after it (the particle). The body's
// fields come as values: a pointer to the body would keep it out of
// registers in every magnet class (GPU).
GPUFUN XT_NOINLINE
void magnet_body_spin_radiation(
    LocalParticle* part,
    const MagnetField* field,
    double const length,
    double const inv_factorial_order,
    double const factor_knl_ksl,
    double const ks,
    double const dks_ds,
    double const x0_solenoid,
    double const y0_solenoid,
    double const h_radiation,
    int64_t const radiation_flag,
    MagnetRadiationRecord radiation_record,
    double const ll,
    double const old_x,
    double const old_y,
    double const old_zeta,
    double const old_kin_px,
    double const old_kin_py,
    double* dp_record_exit,
    double* dpx_record_exit,
    double* dpy_record_exit
) {
    double Bx_T, By_T, Bz_T;
    double const new_kin_px = LocalParticle_get_px(part) - LocalParticle_get_ax(part);
    double const new_kin_py = LocalParticle_get_py(part) - LocalParticle_get_ay(part);
    double const mean_kin_px = 0.5 * (old_kin_px + new_kin_px);
    double const mean_kin_py = 0.5 * (old_kin_py + new_kin_py);
    evaluate_field_from_strengths(
        LocalParticle_get_p0c(part),
        LocalParticle_get_q0(part),
        0.5 * (old_x + LocalParticle_get_x(part)),
        0.5 * (old_y + LocalParticle_get_y(part)),
        length,
        field->kl_order,
        inv_factorial_order,
        field->knl,
        field->ksl,
        factor_knl_ksl,
        field->knorm[0],
        field->knorm[1],
        field->knorm[2],
        field->knorm[3],
        field->kskew[0],
        field->kskew[1],
        field->kskew[2],
        field->kskew[3],
        ks,
        dks_ds,
        x0_solenoid,
        y0_solenoid,
        &Bx_T,
        &By_T,
        &Bz_T
    );
    double const dzeta = LocalParticle_get_zeta(part) - old_zeta;
    double const l_path = LocalParticle_get_rvv(part) * (ll - dzeta);
    magnet_spin(part, Bx_T, By_T, Bz_T, h_radiation, ll, l_path);
    if (radiation_flag) {
        double const B_perp_T = compute_b_perp_mod(
            mean_kin_px, mean_kin_py, LocalParticle_get_delta(part), Bx_T, By_T, Bz_T
        );
        magnet_radiation(
            part, B_perp_T, ll, l_path, radiation_flag, radiation_record,
            dp_record_exit, dpx_record_exit, dpy_record_exit
        );
    }
}
#endif


GPUFUN XT_INLINE
void track_magnet_body_single_particle(
    LocalParticle* part,
    const MagnetBody* body,
    double* dp_record_exit,
    double* dpx_record_exit,
    double* dpy_record_exit
) {
    const double length = body->length;

// The maps for RUN_INTEGRATOR
#define MAGNET_KICK(part, weight)                              \
    do {                                                       \
        if (!body->kick_is_noop)                               \
            MagnetKick_apply(part, &body->kick, (weight));     \
    } while (0)
#define MAGNET_DRIFT(part, dlength) MagnetDrift_apply(part, &body->drift, (dlength))

#if defined(XTRACK_TPSA_TRACK) || defined(XTRACK_MULTIPOLE_NO_SYNRAD)
    #define WITH_RADIATION(ll, code) \
        {                            \
            code;                    \
        }
    #else
        // Spin and radiation of a step, after it: one shared copy (not
        // inlined), not one per integrator step and magnet class
        #define WITH_RADIATION(ll, code) \
        { \
            double const old_x = LocalParticle_get_x(part); \
            double const old_y = LocalParticle_get_y(part); \
            double const old_zeta = LocalParticle_get_zeta(part); \
            double const old_kin_px = LocalParticle_get_px(part) - LocalParticle_get_ax(part); \
            double const old_kin_py = LocalParticle_get_py(part) - LocalParticle_get_ay(part); \
            code; \
            if (length > 0) { /* spin always, radiation if radiation_flag */ \
                magnet_body_spin_radiation( \
                    part, body->field, length, body->inv_factorial_order, \
                    body->factor_knl_ksl, body->drift.ks, body->dks_ds, \
                    body->drift.x0_solenoid, body->drift.y0_solenoid, \
                    body->h_radiation, body->radiation_flag, body->radiation_record, \
                    (ll), old_x, old_y, old_zeta, old_kin_px, old_kin_py, \
                    dp_record_exit, dpx_record_exit, dpy_record_exit \
                ); \
            } \
        }
    #endif

    if (body->drift_only) {
        WITH_RADIATION(length,
            MAGNET_DRIFT(part, length);
        )
    }
    else{

        RUN_INTEGRATOR(
            body->integrator, length, body->num_kicks, part,
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


// Prepare the kick of a body (the per-element part of
// track_magnet_kick_single_particle) from the field and the split of the
// tracking model
GPUFUN
void MagnetKick_prepare(
    MagnetKick* kick,
    const MagnetField* field,
    const MagnetSplit* split,
    double length,
    double inv_factorial_order,
    double factor_knl_ksl,
    double hxl,
    uint8_t is_empty
) {
    const int64_t order = field->kl_order;
    const xt_float_or_tpsa* knl = field->knl;

    kick->length = length;
    kick->order = order;
    kick->inv_factorial_order = inv_factorial_order;
    kick->knl = knl;
    kick->ksl = field->ksl;
    kick->factor_knl_ksl = factor_knl_ksl;
    // The kick part of k0, k1 (configure_tracking_model) and the other
    // strengths, times the length
    kick->knl_main[0] = split->k0_kick * length;
    kick->knl_main[1] = split->k1_kick * length;
    kick->knl_main[2] = field->knorm[2] * length;
    kick->knl_main[3] = field->knorm[3] * length;
    for (int ii = 0; ii <= 3; ii++) {
        kick->ksl_main[ii] = field->kskew[ii] * length;
    }
    kick->h = split->h_kick;
    kick->hxl = hxl;
    kick->htot = split->h_kick;
    if (length != 0) {
        kick->htot += hxl / length;
    }
    // k0h correction can be computed from this term in the hamiltonian
    // H = 1/2 h k0 x^2
    // (see MAD 8 physics manual, eq. 5.15, and apply Hamilton's eq. dp/ds = -dH/dx)
    xt_float_or_tpsa k0l_mult = 0.0;
    if (order >= 0) {
        k0l_mult = knl[0] * factor_knl_ksl;
    }
    kick->k0_h_coeff = split->k0_h_correction * length + k0l_mult;
    // k1h correction can be computed from this term in the hamiltonian
    // H = 1/3 hk1 x^3 - 1/2 hk1 xy^2
    // (see MAD 8 physics manual, eq. 5.15, and apply Hamilton's eq. dp/ds = -dH/dx)
    xt_float_or_tpsa k1l_mult = 0.0;
    if (order >= 1) {
        k1l_mult = knl[1] * factor_knl_ksl;
    }
    kick->k1_h_coeff = split->k1_h_correction * length + k1l_mult;
    kick->rot_frame = split->kick_rot_frame;
    kick->do_drift_correction = split->do_drift_correction;
    kick->is_empty = is_empty;
}


// Derive the integration of the body from its settings and the field of the
// element (the strengths of the faces, see track_magnet_edge.h; they need not
// be prepared).
GPUFUN XT_INLINE
void MagnetBody_prepare(
    MagnetBody* body,
    const MagnetField* field,
    int8_t model,
    int8_t integrator,
    int64_t num_multipole_kicks,
    double weight,  // fraction of the element (slices)
    double h,
    double hxl
) {
    const double body_length = body->length;
    const int64_t order = field->kl_order;
    const xt_float_or_tpsa* knl = field->knl;
    const xt_float_or_tpsa* ksl = field->ksl;
    xt_float_or_tpsa_arg k0 = field->knorm[0];
    xt_float_or_tpsa_arg k1 = field->knorm[1];
    xt_float_or_tpsa_arg k2 = field->knorm[2];
    xt_float_or_tpsa_arg k3 = field->knorm[3];
    xt_float_or_tpsa_arg k0s = field->kskew[0];
    xt_float_or_tpsa_arg k1s = field->kskew[1];
    xt_float_or_tpsa_arg k2s = field->kskew[2];
    xt_float_or_tpsa_arg k3s = field->kskew[3];
    xt_float_or_tpsa_arg ks = field->ks;

    body->field = field;

    if (model == MODEL_KICK_ONLY) {
        integrator = INTEGRATOR_UNIFORM;
        num_multipole_kicks = 1;
    }

    // Adjust the number of multipole kicks based on the weight
    if (weight != 1.0 && num_multipole_kicks > 0) {
        num_multipole_kicks = (int64_t)ceil(num_multipole_kicks * weight);
    }

    const MagnetSplit split = configure_tracking_model(model, k0, k1, h, ks);
    body->drift = split.drift;
    body->drift.x0_solenoid = field->x0_solenoid;
    body->drift.y0_solenoid = field->y0_solenoid;

    // Compute the number of kicks for auto mode
    if (num_multipole_kicks == 0) {  // num_multipole_kicks = 0 means auto mode
        uint8_t const kick_inactive =
            kick_is_inactive(order, knl, ksl, k0, k1, k2, k3, k0s, k1s, k2s, k3s, h);

        // The exact matrix model needs a path-correction kick even at zero field.
        // Otherwise, guess the number of active kicks or perform only the drift.
        if (split.do_drift_correction || !kick_inactive) {
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

    const uint8_t kick_is_empty = (hxl == 0.0)
                          && kick_is_inactive(
                              order,
                              knl,
                              ksl,
                              split.k0_kick,
                              split.k1_kick,
                              k2,
                              k3,
                              k0s,
                              k1s,
                              k2s,
                              k3s,
                              split.h_kick
                          );

    body->integrator = integrator;
    body->num_kicks = num_multipole_kicks;

    // Curvature for the radiation (only for length > 0)
    if (fabs(split.drift.h) > 0) {
        body->h_radiation = split.drift.h;
    }
    else {
        body->h_radiation = body_length > 0 ? split.h_kick + hxl / body_length : 0.;
    }

    body->drift_only = num_multipole_kicks == 0
                       && xt_float_or_tpsa_const_part(split.k0_kick) == 0
                       && xt_float_or_tpsa_const_part(split.k1_kick) == 0
                       && split.h_kick == 0;
    // An empty kick still applies the path correction of mat-kick-mat-exact
    body->kick_is_noop = kick_is_empty && !split.do_drift_correction;
    if (body->drift_only || body->kick_is_noop) return;

    MagnetKick_prepare(
        &body->kick,
        field,
        &split,
        body_length,
        body->inv_factorial_order,
        body->factor_knl_ksl,
        hxl,
        kick_is_empty
    );
}


// The model or integrator of an element: 0 selects the default of the
// element type (default_magnet_config.h)
GPUFUN XT_INLINE
int8_t magnet_or_default(int8_t value, int8_t default_value) {
    return value == 0 ? default_value : value;
}


// The radiation flag of a slice: 10 takes that of the parent element
GPUFUN XT_INLINE
int64_t magnet_radiation_flag(int64_t flag, int64_t parent_flag) {
    return flag == 10 ? parent_flag : flag;
}



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
    s->model = MODEL_ADAPTIVE;
    s->integrator = INTEGRATOR_ADAPTIVE;
    s->radiation_flag = 0;
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
    s->rbend_model = RBEND_MODEL_NONE;
    s->rbend_compensate_sagitta = 0;
    s->rbend_shift = 0.;
    s->rbend_angle_diff = 0.;
    s->length_straight = 0.;
    s->body_active = 1;
}


// Prepare a magnet, or a slice of it (weight: its fraction of the length),
// for track_magnet_particles: its first face, its body and its second face, in
// the order of the direction of travel. The faces are given in the element
// frame. The misalignment is set by the caller (X_prepare_plan). Reads only
// the track flags of part0 and, when tapering, the momentum of its particle.
// Runs once per element (prepare_magnet_plans), not per particle: not
// inlined, so the kernel holds one copy, not one per magnet class.
GPUFUN XT_NOINLINE
void MagnetPlan_prepare(
    MagnetPlan* plan,
    const MagnetSettings* settings,
    const MagnetFace* entry_face_in,
    const MagnetFace* exit_face_in,
    LocalParticle* part0
) {
    // Values changed below (RBend, backtracking, tapering) are local copies;
    // the rest is read from settings
    const double weight = settings->weight;
    double length = settings->length;
    double h = settings->h;
    double hxl = settings->hxl;
    const int64_t rbend_model = settings->rbend_model;
    const int64_t radiation_flag = settings->radiation_flag;
    double delta_taper = settings->delta_taper;
    (void)delta_taper;  // unused in the kernels without radiation

    const uint8_t backtrack = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);

    plan->transform.active = 0;
    plan->is_thick = 0;

    // Field of the body and the faces (the multipoles are added below)
    MagnetField* field = &plan->field;
    field->knorm[0] = settings->k0;
    field->knorm[1] = settings->k1;
    field->knorm[2] = settings->k2;
    field->knorm[3] = settings->k3;
    field->kskew[0] = settings->k0s;
    field->kskew[1] = settings->k1s;
    field->kskew[2] = settings->k2s;
    field->kskew[3] = settings->k3s;
    field->k_order = MAGNET_K_ORDER;
    field->ks = settings->ks;
    field->x0_solenoid = settings->x0_solenoid;
    field->y0_solenoid = settings->y0_solenoid;

    // Faces, in the element frame; the RBend changes their angles or adds
    // patches
    MagnetFace entry_face = *entry_face_in;
    MagnetFace exit_face = *exit_face_in;

    // RBend: the faces of a curved body are at the angle of the reference
    // orbit; a straight body follows the chord, and its length and the
    // length of the curved reference path differ
    double length_curved = 0.;
    double const angle = h * length;
    if (rbend_model == RBEND_MODEL_CURVED_BODY){
        entry_face.angle += (angle - settings->rbend_angle_diff) / 2.0;
        exit_face.angle += (angle + settings->rbend_angle_diff) / 2.0;
    }
    else if (rbend_model == RBEND_MODEL_STRAIGHT_BODY){
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
    double body_length, body_length_curved, knl_scale_body;
    if (backtrack) {
        body_length = -length * weight;
        body_length_curved = -length_curved * weight;
        knl_scale_body = -weight;
        hxl = -hxl;
    }
    else {
        body_length = length * weight;
        body_length_curved = length_curved * weight;
        knl_scale_body = weight;
    }

    // Tapering: all strengths scale with 1 + delta_taper
    double taper = 1.0;
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
            // knl and ksl are scaled where they are used (knl_scale_body, taper)
            knl_scale_body *= (1. + delta_taper);
            taper = 1. + delta_taper;
            for (int ii = 0; ii <= MAGNET_K_ORDER; ii++) {
                field->knorm[ii] *= (1 + delta_taper);
                field->kskew[ii] *= (1 + delta_taper);
            }
        }
    #endif

    // Multipoles of the body and the faces
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
    plan->knl_tot.lift(zeros, order_tot + 1);
    plan->ksl_tot.lift(zeros, order_tot + 1);
    magnet_total_multipoles(
        settings->order,
        settings->knl,
        settings->ksl,
        settings->order_rel,
        settings->knl_rel,
        settings->ksl_rel,
        settings->rel_ref_strength,
        order_tot,
        plan->knl_tot.mutable_ptr(),
        plan->ksl_tot.mutable_ptr()
    );
    field->knl = plan->knl_tot.ptr();
    field->ksl = plan->ksl_tot.ptr();
#else
    // Without a relative component the element's own arrays are used
    field->knl = settings->knl;
    field->ksl = settings->ksl;
    if (magnet_has_rel(settings->order_rel, settings->knl_rel, settings->ksl_rel)) {
        magnet_total_multipoles(
            settings->order,
            settings->knl,
            settings->ksl,
            settings->order_rel,
            settings->knl_rel,
            settings->ksl_rel,
            settings->rel_ref_strength,
            order_tot,
            plan->knl_tot,
            plan->ksl_tot
        );
        field->knl = plan->knl_tot;
        field->ksl = plan->ksl_tot;
    }
#endif
    field->kl_order = order_tot;
    field->knl_scale = taper;
    field->length = length;

    // Faces, in the order they are crossed: the particle enters the field at
    // the first and leaves it at the second, in both directions of travel
    plan->first_face = backtrack ? exit_face : entry_face;
    plan->second_face = backtrack ? entry_face : exit_face;
    if (MagnetFace_uses_field(&entry_face) || MagnetFace_uses_field(&exit_face)) {
        MagnetField_prepare(field);
    }
    MagnetFace_prepare(&plan->first_face, field, backtrack);
    MagnetFace_prepare(&plan->second_face, field, backtrack);
    // A face that does nothing is skipped as a whole, not sub-map by sub-map
    plan->first_active = plan->first_face.has_work;
    plan->second_active = plan->second_face.has_work;

    // Faces only (edge slices)
    plan->body_active = settings->body_active != 0;
    if (!plan->body_active) return;

    // Body
    MagnetBody* body = &plan->body;
    body->length = body_length;
    body->inv_factorial_order = inv_factorial_order_tot;
    body->factor_knl_ksl = knl_scale_body;
    body->dks_ds = settings->dks_ds;
    body->radiation_flag = radiation_flag;
    body->radiation_record = settings->radiation_record;
    MagnetBody_prepare(
        body,
        field,
        settings->model,
        settings->integrator,
        settings->num_multipole_kicks,
        weight,
        h,
        hxl
    );

    // Straight body: the path length is measured on the curved reference
    // (not for the kick only model)
    plan->correct_s = rbend_model == RBEND_MODEL_STRAIGHT_BODY
                      && settings->model != MODEL_KICK_ONLY;
    plan->ds_curved = body_length_curved - body_length;
}


// Misalignment on one side of a magnet: before it (at_exit = 0) or after it
// (at_exit = 1), in the direction of travel. Returns 0 if the particles were
// lost (and the magnet must not be tracked).
GPUFUN
int8_t magnet_transform(LocalParticle* part0, const MagnetTransform* tr, int8_t at_exit) {
    if (!at_exit && tr->invalid) {
        START_PER_PARTICLE_BLOCK(part0, part);
            LocalParticle_set_state(part, XT_INVALID_THIN_SLICE_TRANSFORM);
        END_PER_PARTICLE_BLOCK;
        return 0;
    }
    int8_t const backtrack = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);
    // Backtracking crosses the exit side first
    uint8_t const entry_side = at_exit == backtrack;
    if (tr->curved) {
        if (entry_side) {
            track_misalignment_entry_curved(
                part0, tr->shift_x, tr->shift_y, tr->shift_s, tr->rot_y_rad, tr->rot_x_rad,
                tr->rot_s_rad_no_frame, tr->anchor, tr->length, tr->angle, tr->h,
                tr->rot_s_rad, backtrack
            );
        }
        else {
            track_misalignment_exit_curved(
                part0, tr->shift_x, tr->shift_y, tr->shift_s, tr->rot_y_rad, tr->rot_x_rad,
                tr->rot_s_rad_no_frame, tr->anchor, tr->length, tr->angle, tr->h,
                tr->rot_s_rad, backtrack
            );
        }
    }
    else {
        if (entry_side) {
            track_misalignment_entry_straight(
                part0, tr->shift_x, tr->shift_y, tr->shift_s, tr->rot_y_rad, tr->rot_x_rad,
                tr->rot_s_rad_no_frame, tr->anchor, tr->length, tr->rot_s_rad, backtrack
            );
        }
        else {
            track_misalignment_exit_straight(
                part0, tr->shift_x, tr->shift_y, tr->shift_s, tr->rot_y_rad, tr->rot_x_rad,
                tr->rot_s_rad_no_frame, tr->anchor, tr->length, tr->rot_s_rad, backtrack
            );
        }
    }
    return 1;
}


// Track a magnet, or a slice of it, from its plan: the misalignment, the
// first face, the body, the second face. Every magnet class is tracked by
// this one function, called from one place in track_line, where it is
// inlined (faster than a call on GPU, with NVRTC and clang).
GPUFUN XT_INLINE
void track_magnet_particles(LocalParticle* part0, const MagnetPlan* plan) {
    const MagnetTransform* transform = &plan->transform;
    if (transform->active && !magnet_transform(part0, transform, 0)) {
        return;
    }

    const uint8_t first_active = plan->first_active;
    const uint8_t second_active = plan->second_active;
    if (plan->body_active) {
        const uint8_t correct_s = plan->correct_s;
        const double ds_curved = plan->ds_curved;
        double dp_record_exit, dpx_record_exit, dpy_record_exit;
        START_PER_PARTICLE_BLOCK(part0, part)
            ;
            if (first_active) MagnetFace_enter_single_particle(part, &plan->first_face);
            track_magnet_body_single_particle(
                part, &plan->body, &dp_record_exit, &dpx_record_exit, &dpy_record_exit
            );
            if (correct_s) {
                LocalParticle_add_to_s(part, ds_curved);
                LocalParticle_add_to_zeta(part, ds_curved);
            }
            if (second_active) MagnetFace_leave_single_particle(part, &plan->second_face);
        END_PER_PARTICLE_BLOCK;
    }
    else {
        // Faces only (edge slices)
        START_PER_PARTICLE_BLOCK(part0, part)
            ;
            if (first_active) MagnetFace_enter_single_particle(part, &plan->first_face);
            if (second_active) MagnetFace_leave_single_particle(part, &plan->second_face);
        END_PER_PARTICLE_BLOCK;
    }

    if (transform->active) {
        magnet_transform(part0, transform, 1);
    }
}

#endif
