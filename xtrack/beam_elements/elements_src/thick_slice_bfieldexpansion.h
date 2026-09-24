#ifndef XTRACK_THICK_SLICE_BFIELDEXPANSION_H
#define XTRACK_THICK_SLICE_BFIELDEXPANSION_H

#include "xtrack/beam_elements/elements_src/track_bfieldexpansion.h"

GPUFUN
void ThickSliceBFieldExpansion_track_local_particle(
        ThickSliceBFieldExpansionData el, LocalParticle* part0) {
    BFieldExpansionData parent = ThickSliceBFieldExpansionData_getp__parent(el);
    const double weight = ThickSliceBFieldExpansionData_get_weight(el);
    const double parent_length = BFieldExpansionData_get_length(parent);
    const double sstart = BFieldExpansionData_get_sstart(parent)
        + parent_length * ThickSliceBFieldExpansionData_get__slice_offset_fraction(el);
    int64_t nstep = (int64_t)ceil(BFieldExpansionData_get_nstep(parent) * weight);
    if (nstep < 1) nstep = 1;

    BFieldExpansion_track_interval(parent, part0, parent_length * weight, sstart, nstep);
}

#endif
