/* This file serves as a template for generating per-element tracking functions
   that handle misalignments and transformations. It expects the following macros
   to be defined (as appropriate) before inclusion:

   - ELEMENT_NAME: Name of the beam element class
   - ALLOW_ROT_AND_SHIFT: Whether to include transformations handling
   - CURVED: Whether the element has the `angle` and `h` parameters
   - IS_SLICE: Whether to get parameters from the parent element (if slice)
   - IS_THICK: Whether the element has a length
   - IS_THICK_DYNAMIC: Whether the element has a isthick parameter
   - THIN_SLICE_OF_CURVED_ELEMENT: Special handling for thin slices of curved
     elements where we disallow some transformations.
*/

#ifndef ELEMENT_NAME
    #error "This file must be included with ELEMENT_NAME macro defined."
#endif

// Common concatenation macros, unfortunately not part of standard C headers
#ifndef _CONCAT
#define _CONCAT(a, b) a ## b
#define CONCAT(a, b) _CONCAT(a, b)
#define CONCAT3(a, b, c) CONCAT(CONCAT(a, b), c)
#endif /* !defined(_CONCAT) */

#include <headers/track.h>
#include <headers/particle_states.h>
#include <beam_elements/elements_src/track_misalignments.h>

// Shorthand for `ElementClassNameData`
#define ELEMENT_DATA CONCAT(ELEMENT_NAME, Data)

// Retrieve parameters attached to the element object
#define ELEMENT_GET(el, param_name) CONCAT3(ELEMENT_DATA, _get_, param_name)(el)

#ifdef IS_SLICE
    // Retrieve parameters from the parent element
    #define PARENT_GET(el, param_name) CONCAT3(ELEMENT_DATA, _get__parent_, param_name)(el)
    // In case of a slice the parameters need to be retrieved from the parent...
    #define GET_PARAM(el, param_name) PARENT_GET(el, param_name)
    // ...except the weight, which lives in the slice itself
    #define GET_WEIGHT(el) CONCAT(ELEMENT_DATA, _get_weight)(el)
#else
    // Standard retrieval of parameters from the element
    #define GET_PARAM(el, param_name) ELEMENT_GET(el, param_name)
    // Since it's not a slice, weight is always 1.0
    #define GET_WEIGHT(_) 1.0
#endif

// The misalignment functions for curved and straight elements have slightly
// different signatures, so we abstract the differences here, as we will call
// them in multiple places.
#ifdef CURVED
    #define MISALIGN_FUNCTION_ARGS \
        part0, \
        shift_x, \
        shift_y, \
        shift_s, \
        rot_y_rad, \
        rot_x_rad, \
        rot_s_rad_no_frame, \
        anchor, \
        length * weight, \
        angle * weight, \
        h, \
        rot_s_rad, \
        backtrack
    #define TRACK_MISALIGN_ENTRY track_misalignment_entry_curved(MISALIGN_FUNCTION_ARGS)
    #define TRACK_MISALIGN_EXIT track_misalignment_exit_curved(MISALIGN_FUNCTION_ARGS)
#else
    #define MISALIGN_FUNCTION_ARGS \
        part0, \
        shift_x, \
        shift_y, \
        shift_s, \
        rot_y_rad, \
        rot_x_rad, \
        rot_s_rad_no_frame, \
        anchor, \
        length * weight, \
        rot_s_rad, \
        backtrack
    #define TRACK_MISALIGN_ENTRY track_misalignment_entry_straight(MISALIGN_FUNCTION_ARGS)
    #define TRACK_MISALIGN_EXIT track_misalignment_exit_straight(MISALIGN_FUNCTION_ARGS)
#endif


/* If transformations are enabled for an element, we define a function that
   handles the misalignments and then calls the local tracking function.
   This functions already assumes that the transformations are
   (1) possible for the element and (2) non-zero.
*/
#ifdef ALLOW_ROT_AND_SHIFT

GPUFUN
void CONCAT(ELEMENT_NAME, _track_local_particle_with_nonzero_transformations)(
    ELEMENT_DATA el,
    LocalParticle* part0
) {
    // Retrieve the weight
    const double weight = GET_WEIGHT(el);

    // Determine the length
    #if defined(IS_THICK_DYNAMIC)
        const int8_t is_thick = GET_PARAM(el, isthick);
        const double length = is_thick ? GET_PARAM(el, length) : 0.0;
    #elif defined(IS_THICK)
        const double length = GET_PARAM(el, length);
    #else
        const double length = 0.0;
    #endif

    // Retrieve the `angle` and `h` if available
    #ifdef CURVED
        double angle = GET_PARAM(el, angle);
        double h = GET_PARAM(el, h);
    #endif

    // Retrieve misalignment parameters
    double const shift_x = GET_PARAM(el, shift_x);
    double const shift_y = GET_PARAM(el, shift_y);
    double const shift_s = GET_PARAM(el, shift_s);
    double const rot_x_rad = GET_PARAM(el, rot_x_rad);
    double const rot_y_rad = GET_PARAM(el, rot_y_rad);
    double const rot_s_rad = GET_PARAM(el, rot_s_rad);
    double const rot_s_rad_no_frame = GET_PARAM(el, rot_s_rad_no_frame);
    double anchor = GET_PARAM(el, rot_shift_anchor);

    #ifdef THIN_SLICE_OF_CURVED_ELEMENT
        if (rot_x_rad != 0.0 || rot_y_rad != 0.0 || rot_s_rad_no_frame != 0.0) {
            START_PER_PARTICLE_BLOCK(part0, part);
                LocalParticle_set_state(part, XT_INVALID_THIN_SLICE_TRANSFORM);
            END_PER_PARTICLE_BLOCK;
            return;
        }
    #endif

    int8_t const backtrack = LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK);

    #ifdef IS_SLICE
        double const slice_offset = ELEMENT_GET(el, slice_offset);
        anchor = anchor - slice_offset;
    #endif

    if (!backtrack) {
        TRACK_MISALIGN_ENTRY;
    } else {
        TRACK_MISALIGN_EXIT;
    }

    // Track the particle in the local frame
    CONCAT(ELEMENT_NAME, _track_local_particle)(el, part0);

    if (!backtrack) {
        TRACK_MISALIGN_EXIT;
    } else {
        TRACK_MISALIGN_ENTRY;
    }
}

#endif /* ALLOW_ROT_AND_SHIFT */

/* This is the main function defined in this file. It checks whether
   transformations are enabled for the element, and calls either the
   standard local tracking function or the one that handles transformations.
*/
GPUFUN
void CONCAT(ELEMENT_NAME, _track_local_particle_with_transformations)(
    ELEMENT_DATA el,
    LocalParticle* part0
) {
    #ifndef ALLOW_ROT_AND_SHIFT
        CONCAT(ELEMENT_NAME, _track_local_particle)(el, part0);
    #else
        double const shift_x = GET_PARAM(el, shift_x);
        double const shift_y = GET_PARAM(el, shift_y);
        double const shift_s = GET_PARAM(el, shift_s);
        double const rot_x_rad = GET_PARAM(el, rot_x_rad);
        double const rot_y_rad = GET_PARAM(el, rot_y_rad);
        double const rot_s_rad = GET_PARAM(el, rot_s_rad);
        double const rot_s_rad_no_frame = GET_PARAM(el, rot_s_rad_no_frame);

        const int64_t rot_shift_active = \
            NONZERO(shift_x) ||
            NONZERO(shift_y) ||
            NONZERO(shift_s) ||
            NONZERO(rot_x_rad) ||
            NONZERO(rot_y_rad) ||
            NONZERO(rot_s_rad) ||
            NONZERO(rot_s_rad_no_frame);

        if (rot_shift_active) {
            CONCAT(ELEMENT_NAME, _track_local_particle_with_nonzero_transformations)(el, part0);
        } else {
            CONCAT(ELEMENT_NAME, _track_local_particle)(el, part0);
        }
    #endif
}


#undef ELEMENT_DATA
#undef ELEMENT_GET
#undef PARENT_GET
#undef GET_PARAM
#undef GET_WEIGHT
#undef MISALIGN_FUNCTION_ARGS
#undef TRACK_MISALIGN_ENTRY
#undef TRACK_MISALIGN_EXIT
