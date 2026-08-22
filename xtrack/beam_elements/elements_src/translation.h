// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2021.                 //
// ######################################### //
#ifndef XTRACK_TRANSLATION_H
#define XTRACK_TRANSLATION_H

#include "xtrack/headers/track.h"
#include "xtrack/beam_elements/elements_src/track_xyshift.h"


GPUFUN
void Translation_track_local_particle(TranslationData el, LocalParticle* part0){

    double shift_x = TranslationData_get_shift_x(el);
    double shift_y = TranslationData_get_shift_y(el);

    if (LocalParticle_check_track_flag(part0, XS_FLAG_BACKTRACK)) {
        shift_x = -shift_x;
        shift_y = -shift_y;
    }

    START_PER_PARTICLE_BLOCK(part0, part);
        XYShift_single_particle(part, shift_x, shift_y);
    END_PER_PARTICLE_BLOCK;
}

#endif /* XTRACK_TRANSLATION_H */
