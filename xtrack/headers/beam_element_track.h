/*
 * Template for scalar BeamElement.track kernels. Include this file with
 * ELEMENT_NAME and KERNEL_NAME defined; it deliberately has no include guard.
 */

#ifndef ELEMENT_NAME
    #error "beam_element_track.h requires ELEMENT_NAME"
#endif

#ifndef KERNEL_NAME
    #error "beam_element_track.h requires KERNEL_NAME"
#endif

#define LOCAL_PARTICLE_FUNCTION \
    CONCAT(ELEMENT_NAME, _track_local_particle_with_transformations)
#define KERNEL_EXTRA_ARGUMENTS
#define KERNEL_EXTRA_ARGUMENT_VALUES

#include "xtrack/headers/per_particle_kernel.h"

#undef KERNEL_EXTRA_ARGUMENT_VALUES
#undef KERNEL_EXTRA_ARGUMENTS
#undef LOCAL_PARTICLE_FUNCTION
