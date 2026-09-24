#ifndef XTRACK_CREATE_BFIELDEXPANSION_H
#define XTRACK_CREATE_BFIELDEXPANSION_H

#include "create_bfieldexpansion_straight.h"
#include "create_bfieldexpansion_bent.h"

void build_bfield_expansion(BFieldExpansionData el) {
    if (BFieldExpansionData_get_straight(el)) {
        build_expansion_straight(el);
    }
    else {
        build_expansion_bent(el);
    }
}

#endif
