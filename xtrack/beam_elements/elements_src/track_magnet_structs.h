// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //

#ifndef XTRACK_TRACK_MAGNET_STRUCTS_H
#define XTRACK_TRACK_MAGNET_STRUCTS_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/edge_flags.h"

// The radiation record (SynchrotronRadiationRecordData, a pointer to this
// struct) is typedef'd by xobjects only in kernels with radiation; the
// struct tag works in all, including the drift-only users of this file.
typedef GPUGLMEM struct SynchrotronRadiationRecordData_s* MagnetRadiationRecord;

// Highest multipole order of a magnet (as in MAD-X), checked in Python
#define MAGNET_MAX_MULTIPOLE_ORDER 20

// The data of the magnet kernel, in the order it is derived:
// MagnetSettings (filled by the element) -> MagnetDrift and MagnetSplit (the
// tracking model) -> MagnetField, MagnetFace (the faces) -> MagnetKick,
// MagnetBody (the body, prepared once per element call).


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
    int8_t integrator;
    int64_t radiation_flag;
    MagnetRadiationRecord radiation_record;
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
    int64_t rbend_model;  // RBEND_MODEL_*
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


// The drift of a magnet body: the part of the field that the drift map
// integrates, and the map (see configure_tracking_model)
typedef struct MagnetDrift_s {
    int8_t model;
    xt_float_or_tpsa k0, k1, ks;
    double h;
    double x0_solenoid, y0_solenoid;
#ifdef XTRACK_TPSA_TRACK
    // The TPSA type has no default constructor
    MagnetDrift_s() : k0(0.), k1(0.), ks(0.) {}
#endif
} MagnetDrift;


// How a magnet body is integrated: the split of the field between the drifts
// and the kicks of the integrator, for a tracking model
typedef struct MagnetSplit_s {
    MagnetDrift drift;  // solenoid offsets at 0: set by the caller
    xt_float_or_tpsa k0_kick, k1_kick;
    double h_kick;
    xt_float_or_tpsa k0_h_correction, k1_h_correction;  // curvature terms of the kick
    int8_t kick_rot_frame;
    uint8_t do_drift_correction;  // exact path correction in the kick (model 9)
#ifdef XTRACK_TPSA_TRACK
    // The TPSA type has no default constructor
    MagnetSplit_s()
        : k0_kick(0.), k1_kick(0.), k0_h_correction(0.), k1_h_correction(0.) {}
#endif
} MagnetSplit;


// Field of a magnet, shared by its body and its faces: the strengths
// k = knorm / kskew (orders 0 ... k_order), plus knl_scale * knl / length
// (knl includes the relative multipoles); the solenoid strength for the
// vector potential.
// The strengths are held by value: on GPU, pointers into the element's
// settings would force those settings into local memory.
typedef struct MagnetField_s {
    xt_float_or_tpsa knorm[MAGNET_K_ORDER + 1];
    xt_float_or_tpsa kskew[MAGNET_K_ORDER + 1];
    int64_t k_order;
    const xt_float_or_tpsa* knl;
    const xt_float_or_tpsa* ksl;
    int64_t kl_order;
    double knl_scale;
    double length;
    xt_float_or_tpsa ks;
    double x0_solenoid, y0_solenoid;

    // Derived by MagnetField_prepare and MagnetFace_prepare, only when a
    // face is active
    xt_float_or_tpsa k0, k1, k1s;  // total (skew) dipole and quadrupole
    // Linear edge, [is_exit_face]. Kept here and not in MagnetFace,
    // which holds only doubles: a TPSA is built only for an active face.
    xt_float_or_tpsa r21[2], r43[2];
    uint8_t has_mult;  // a multipole of order >= 1 (the multipole fringe)
#ifdef XTRACK_TPSA_TRACK
    // The TPSA type has no default constructor
    MagnetField_s() : knorm{0., 0., 0., 0.}, kskew{0., 0., 0., 0.}, ks(0.), k0(0.), k1(0.), k1s(0.),
                      r21{0., 0.}, r43{0., 0.} {}
#endif
} MagnetField;


// One face of a magnet. The settings are given in the element frame by the
// element; MagnetFace_prepare derives what the face does for one direction of
// travel, so that the per-particle code only branches on the do_* switches.
typedef struct MagnetFace_s {
    // Settings
    int64_t flags;         // fringe mask, see edge_flags.h
    uint8_t is_exit_face;  // 0: entry face, 1: exit face of the element
    double angle;          // face angle
    double angle_fdown;    // feed-down angle of the linear model
    double fint;           // fringe field integral
    double hgap;           // half gap
    double f1, f2;         // soft quadrupole fringe (Lee-Whiting)
    // Solenoid strength at the face, when it is not the one of the field
    // (VariableSolenoid: ks varies along the body). The vector potential of
    // this ks is set both when entering and when leaving: the field goes on
    // past the face
    uint8_t has_ks;
    double ks;
    // Patch between the drift and the frame of the magnet (straight-body
    // RBend): a y-rotation by patch_angle and a shift by patch_dx
    uint8_t has_patch;
    double patch_angle, patch_dx;

    // Derived by MagnetFace_prepare; angles and kicks signed for the
    // direction of travel
    const MagnetField* field;
    uint8_t backtrack;
    uint8_t has_work;  // set by MagnetFace_prepare: the face applies something
    uint8_t do_patch, do_vector_potential, do_linear;
    uint8_t do_rotate, do_bend, do_mult, do_quad_lw, do_quad_wedge, do_wedge;
    double patch_sin, patch_cos, patch_tan;
    double rot_sin, rot_cos, rot_tan;  // y-rotation by the face angle
    double wedge_angle;
    double quad_lw_g1, quad_lw_g2;
} MagnetFace;


// The kick of a magnet body, prepared once per element call
// (MagnetKick_prepare): only the per-particle part is left to the kick.
typedef struct MagnetKick_s {
    double length;
    int64_t order;
    double inv_factorial_order;
    const xt_float_or_tpsa* knl;  // knl + rel_ref_strength * knl_rel
    const xt_float_or_tpsa* ksl;
    double factor_knl_ksl;
    xt_float_or_tpsa knl_main[4];  // k0 ... k3 times the length
    xt_float_or_tpsa ksl_main[4];
    double h;
    double hxl;
    double htot;                    // h + hxl / length
    xt_float_or_tpsa k0_h_coeff;    // curvature terms, see the kick
    xt_float_or_tpsa k1_h_coeff;
    uint8_t rot_frame;
    uint8_t do_drift_correction;
    uint8_t is_empty;
#ifdef XTRACK_TPSA_TRACK
    // The TPSA type has no default constructor
    MagnetKick_s()
        : knl_main{0., 0., 0., 0.}, ksl_main{0., 0., 0., 0.},
          k0_h_coeff(0.), k1_h_coeff(0.) {}
#endif
} MagnetKick;


// Body of a magnet: an integrator (RUN_INTEGRATOR) alternating a drift and a
// kick, which share the field as the tracking model says
// (configure_tracking_model). Prepared once per element call by
// MagnetBody_prepare, so that the per-particle map only applies it.
typedef struct MagnetBody_s {
    double length;             // signed for the direction of travel
    const MagnetField* field;  // strengths, read through it (no copies)

    // Integration
    int8_t integrator;   // integrator and number of kicks resolved
    int64_t num_kicks;   // (kick only model, auto mode)
    uint8_t drift_only;  // no kicks: the body is one drift
    uint8_t kick_is_noop;  // the kick changes nothing: neither prepared nor called
    MagnetDrift drift;
    MagnetKick kick;     // prepared only if !drift_only and !kick_is_noop

    // Field along the body, for the kick and the radiation and spin
    double inv_factorial_order;  // of the order of the field
    double factor_knl_ksl;       // multiplies knl, ksl
    double dks_ds;
    double h_radiation;  // curvature
    int64_t radiation_flag;
    MagnetRadiationRecord radiation_record;
} MagnetBody;


// Misalignment of a magnet, read from the element by X_prepare_plan
// (track_local_particle_with_transformations.h); length and angle are those
// of the slice
typedef struct MagnetTransform_s {
    uint8_t active;   // a shift or a rotation is nonzero
    uint8_t invalid;  // rotated thin slice of a curved element: particles are lost
    uint8_t curved;
    double shift_x, shift_y, shift_s;
    double rot_x_rad, rot_y_rad, rot_s_rad, rot_s_rad_no_frame;
    double anchor;
    double length, angle, h;
} MagnetTransform;


// A magnet, or a slice of it, prepared once per element (MagnetPlan_prepare)
// for one direction of travel: track_magnet_particles only applies it. The
// pointers in it (field, multipoles) point into the plan or the element.
typedef struct MagnetPlan_s {
    MagnetField field;
    MagnetFace first_face, second_face;  // in the order they are crossed
    MagnetBody body;
    uint8_t first_active, second_active;
    uint8_t body_active;
    uint8_t correct_s;  // straight-body RBend: path length on the curved reference
    double ds_curved;
    uint8_t is_thick;   // the global aperture is checked after it
    MagnetTransform transform;
    // knl + rel_ref_strength * knl_rel (same for ksl), formed only with
    // relative components; otherwise the field uses the element's arrays
#ifdef XTRACK_TPSA_TRACK
    xt_tpsa_lifted_array knl_tot, ksl_tot;
#else
    double knl_tot[MAGNET_MAX_MULTIPOLE_ORDER + 1];
    double ksl_tot[MAGNET_MAX_MULTIPOLE_ORDER + 1];
#endif
} MagnetPlan;

#endif // XTRACK_TRACK_MAGNET_STRUCTS_H
