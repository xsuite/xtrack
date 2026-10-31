// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_DEFAULT_MAGNET_CONFIG_H
#define XTRACK_DEFAULT_MAGNET_CONFIG_H


// Tracking models of a magnet body: the index of `model` in Python
// (_INDEX_TO_MODEL_* in beam_elements/_common.py), plus two internal ones
#define MODEL_ADAPTIVE                  (0)   // the default of the element type
#define MODEL_FULL                      (1)   // = rot-kick-rot (backward compatibility)
#define MODEL_BEND_KICK_BEND            (2)
#define MODEL_ROT_KICK_ROT              (3)
#define MODEL_MAT_KICK_MAT              (4)   // previously called `expanded`
#define MODEL_DRIFT_KICK_DRIFT_EXACT    (5)
#define MODEL_DRIFT_KICK_DRIFT_EXPANDED (6)
#define MODEL_ROT_KICK_ROT_LOW_ORDER    (7)
#define MODEL_ROT_KICK_ROT_HIGH_ORDER   (8)
#define MODEL_MAT_KICK_MAT_EXACT        (9)   // straight elements only
#define MODEL_KICK_ONLY                 (-1)  // internal: thin slices, thin multipoles
#define MODEL_SOL_KICK_SOL              (-2)  // internal: solenoids

// Maps of the drift of a magnet body (track_magnet_drift_single_particle),
// chosen by configure_tracking_model
#define DRIFT_MAP_NONE                (-1)  // kick only
#define DRIFT_MAP_EXPANDED            (0)
#define DRIFT_MAP_EXACT               (1)
#define DRIFT_MAP_POLAR               (2)   // h, no field
#define DRIFT_MAP_DIPOLE_QUAD_EXPANDED (3)  // k0, k1, h, expanded
#define DRIFT_MAP_BEND                (4)   // k0 and h, exact
#define DRIFT_MAP_BEND_STRAIGHT       (5)   // k0 without h, exact
#define DRIFT_MAP_SOLENOID            (6)
#define DRIFT_MAP_BEND_YOSHIDA4       (7)   // k0 and h: rot-kick-rot, Yoshida 4
#define DRIFT_MAP_BEND_YOSHIDA6       (8)   // k0 and h: rot-kick-rot, Yoshida 6

// Integrators of a magnet body (RUN_INTEGRATOR): the index of `integrator`
// in Python (_INDEX_TO_INTEGRATOR)
#define INTEGRATOR_ADAPTIVE (0)  // the default of the element type
#define INTEGRATOR_TEAPOT   (1)
#define INTEGRATOR_YOSHIDA6 (2)  // historical index
#define INTEGRATOR_UNIFORM  (3)
#define INTEGRATOR_YOSHIDA4 (4)
#define INTEGRATOR_YOSHIDA8 (5)

// Body of an RBend (_INDEX_TO_RBEND_MODEL), and of the other magnets
#define RBEND_MODEL_NONE          (-1)  // not an RBend
#define RBEND_MODEL_ADAPTIVE      (0)   // curved body
#define RBEND_MODEL_CURVED_BODY   (1)
#define RBEND_MODEL_STRAIGHT_BODY (2)

#define BEND_DEFAULT_MODEL       MODEL_ROT_KICK_ROT
#define RBEND_DEFAULT_MODEL      MODEL_ROT_KICK_ROT
#define QUADRUPOLE_DEFAULT_MODEL MODEL_MAT_KICK_MAT
#define SEXTUPOLE_DEFAULT_MODEL  MODEL_DRIFT_KICK_DRIFT_EXPANDED
#define OCTUPOLE_DEFAULT_MODEL   MODEL_DRIFT_KICK_DRIFT_EXPANDED
#define MULTIPOLE_DEFAULT_MODEL  MODEL_DRIFT_KICK_DRIFT_EXPANDED

#define BEND_DEFAULT_INTEGRATOR       INTEGRATOR_YOSHIDA6
#define RBEND_DEFAULT_INTEGRATOR      INTEGRATOR_YOSHIDA6
#define QUADRUPOLE_DEFAULT_INTEGRATOR INTEGRATOR_UNIFORM
#define SEXTUPOLE_DEFAULT_INTEGRATOR  INTEGRATOR_UNIFORM
#define OCTUPOLE_DEFAULT_INTEGRATOR   INTEGRATOR_UNIFORM
#define SOLENOID_DEFAULT_INTEGRATOR   INTEGRATOR_UNIFORM
#define MULTIPOLE_DEFAULT_INTEGRATOR  INTEGRATOR_UNIFORM

#endif
