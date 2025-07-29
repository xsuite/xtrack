// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2023.                 //
// ######################################### //

#ifndef XTRACK_DEFAULT_MAGNET_CONFIG_H
#define XTRACK_DEFAULT_MAGNET_CONFIG_H

#define BEND_DEFAULT_MODEL       (3) // rot-kick-rot
#define RBEND_DEFAULT_MODEL      (3) // rot-kick-rot
#define QUADRUPOLE_DEFAULT_MODEL (4) // mat-kick-mat
#define SEXTUPOLE_DEFAULT_MODEL  (6) // drift-kick-drift-expanded
#define OCTUPOLE_DEFAULT_MODEL   (6) // drift-kick-drift-expanded

#define BEND_DEFAULT_INTEGRATOR       (2) // Yoshida-4
#define RBEND_DEFAULT_INTEGRATOR      (2) // Yoshida-4
#define QUADRUPOLE_DEFAULT_INTEGRATOR (3) // uniform
#define SEXTUPOLE_DEFAULT_INTEGRATOR  (3) // uniform
#define OCTUPOLE_DEFAULT_INTEGRATOR   (3) // uniform
#define SOLENOID_DEFAULT_INTEGRATOR   (3) // uniform

#endif
