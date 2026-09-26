#ifndef XTRACK_THICK_SLICE_BFIELDEXPANSION_H
#define XTRACK_THICK_SLICE_BFIELDEXPANSION_H

#include "xtrack/beam_elements/elements_src/bfieldexpansion.h"

/* The standard misalignment wrapper requests a physical slice_offset. Store
 * a fraction so that it follows parent length updates without rewriting slices.
 */
GPUFUN
double ThickSliceBFieldExpansionData_get_slice_offset(
        ThickSliceBFieldExpansionData el) {
    return ThickSliceBFieldExpansionData_get__slice_offset_fraction(el)
        * ThickSliceBFieldExpansionData_get__parent_length(el);
}

GPUFUN
void ThickSliceBFieldExpansion_track_local_particle(
        ThickSliceBFieldExpansionData el, LocalParticle* part0) {
    BFieldExpansionData parent = ThickSliceBFieldExpansionData_getp__parent(el);
    const double weight = ThickSliceBFieldExpansionData_get_weight(el);
    const double parent_length = BFieldExpansionData_get_length(parent);
    const double s_start = BFieldExpansionData_get_s_start(parent)
        + parent_length * ThickSliceBFieldExpansionData_get__slice_offset_fraction(el);
    int64_t nstep = (int64_t)ceil(BFieldExpansionData_get_num_integration_steps(parent) * weight);
    if (nstep < 1) nstep = 1;

    BFieldExpansion_track_interval(parent, part0, parent_length * weight, s_start, nstep);
}

#endif
