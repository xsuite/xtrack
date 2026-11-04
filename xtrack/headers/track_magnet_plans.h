// copyright ############################### //
// This file is part of the Xtrack Package.  //
// Copyright (c) CERN, 2025.                 //
// ######################################### //
#ifndef XTRACK_TRACK_MAGNET_PLANS_H
#define XTRACK_TRACK_MAGNET_PLANS_H

/* Magnet plans: the C side of xtrack/magnet_plans.py.

   A magnet is tracked in two steps. Once per track call, the kernel
   prepare_magnet_plans reads the attributes of each magnet and stores them,
   ready to use, in its MagnetPlan. The track kernel then tracks every
   particle through the plan with track_magnet_particles.

   This header is inlined in the track kernel source right after the switch
   prepare_magnet_plan, which Python generates because it depends on the type
   ids of the build. A build with no magnets has neither.
   In TPSA tracking, only the switch and track_magnet_tapered are used: the
   plans are made at the magnets (magnet_plans.py).
*/

/* Track through a magnet with tapering. The plan depends on the energy of the
   particle at the element, so it is made here, at the element, instead of
   once per track call.
*/
GPUFUN
XT_NOINLINE
void track_magnet_tapered(
    LocalParticle* part,
    /*gpuglmem*/ void* el,
    int64_t elem_type
) {
    MagnetPlan plan;
    prepare_magnet_plan(el, elem_type, &plan, part);
    track_magnet_particles(part, &plan);
}

#ifndef XTRACK_TPSA_TRACK

    // The plans are MAGNET_PLAN_BYTES apart (set in magnet_plans.py)
    #if defined(__cplusplus)
static_assert(sizeof(MagnetPlan) <= MAGNET_PLAN_BYTES, "MAGNET_PLAN_BYTES too small");
    #elif !defined(XO_CONTEXT_CL)  // OpenCL C has no static assertion
_Static_assert(sizeof(MagnetPlan) <= MAGNET_PLAN_BYTES, "MAGNET_PLAN_BYTES too small");
    #endif


/* Fill the plans in the slots [first_plan, first_plan + num_plans), one
   thread per plan. magnet_plan_elements[slot] is the index of the magnet of
   the slot in the line.
*/
/*gpukern*/
void prepare_magnet_plans(
    /*gpuglmem*/ int8_t* buffer,
    ElementRefData elem_ref_data,
    /*gpuglmem*/ int8_t* magnet_plans,
    /*gpuglmem*/ int64_t* magnet_plan_elements,
    int64_t first_plan,
    int64_t num_plans,
    /*gpuglmem*/ int8_t* io_buffer,
    uint64_t track_flags
) {
    // A plan reads only the track flags of the particle (the direction of
    // travel). Tapering is cleared: its plans are made at the elements
    LocalParticle lpart;
    lpart.io_buffer = io_buffer;
    lpart.track_flags = track_flags & ~(((uint64_t)1) << XS_FLAG_SR_TAPER);

    // clang-format off: VECTORIZE_OVER is expanded by xobjects, as written
    VECTORIZE_OVER(ii, num_plans);
        const int64_t slot = first_plan + ii;
        const int64_t elem_idx = magnet_plan_elements[slot];
        /*gpuglmem*/ void* el = ElementRefData_member_elements(elem_ref_data, elem_idx);
        const int64_t elem_type = ElementRefData_typeid_elements(elem_ref_data, elem_idx);
        MagnetPlan* plan = (MagnetPlan*)(magnet_plans + MAGNET_PLAN_BYTES * slot);
        prepare_magnet_plan(el, elem_type, plan, &lpart);
    END_VECTORIZE;
    // clang-format on
}

#endif

#endif /* XTRACK_TRACK_MAGNET_PLANS_H */
