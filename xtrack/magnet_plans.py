# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

"""Magnet plans: how the tracker tracks magnets.

What a plan is
--------------
A magnet (Bend, RBend, Quadrupole, ..., and their slices) is tracked in two
steps, both in C:

1. ``<Class>_prepare_plan`` reads the attributes of the element and stores
   them, ready to use, in a ``MagnetPlan`` (track_magnet_structs.h): the
   field, the two faces in the order the particles cross them, the body and
   the transforms.
2. ``track_magnet_particles`` (track_magnet.h) tracks the particles through
   the plan. It does not read the element.

Step 1 runs once per magnet and track call, step 2 once per particle. A class
is a magnet if it sets ``_magnet_plan = True``.

When things happen
------------------
- When the track kernel is built (``Tracker._compile_kernel_from_classes``):
  ``plan_source`` adds the C of the plans to the kernel source, and
  ``track_line_cases`` adds the magnet cases to the element switch of
  ``track_line``. ``attach_prepare_kernel`` keeps the kernel that fills the
  plans with the track kernel of the same build.
- At the first track call of a tracker (``MagnetPlans``): one slot of
  ``MAGNET_PLAN_BYTES`` is allocated per magnet of the line. Only the
  positions of the magnets in the line are used.
- At every track call (``MagnetPlans.prepare``): the kernel
  ``prepare_magnet_plans`` fills the plans of the magnets the call will cross,
  from the current values of their attributes. A change of an attribute
  between two calls is therefore always used: nothing is cached in Python.
- During tracking (``track_line``): each magnet is tracked from its slot.

Exceptions
----------
- Tapering: the plan depends on the energy of the particle at the magnet, so
  it is made at the magnet, by ``track_magnet_tapered``.
- TPSA tracking does not use plans.
"""

import numpy as np
import xobjects as xo

from .general import _pkg_root

_PLANS_HEADER = _pkg_root.joinpath("headers/track_magnet_plans.h")

# The size of a slot, at least sizeof(MagnetPlan) (checked in C)
MAGNET_PLAN_BYTES = 1536


def magnet_classes(kernel_element_classes):
    """Return the magnets among the element classes of a kernel.

    Returns a list of ``(type_id, class)``, where ``type_id`` is the index of
    the class in ``kernel_element_classes``, i.e. its type id in the kernel.
    """
    return [(ii, cc) for ii, cc in enumerate(kernel_element_classes)
            if vars(cc._DressingClass).get('_magnet_plan', False)]


def plan_source(magnet_classes):
    """Return the C source of the plans, for the track kernel.

    It is made of two parts:

    - the function ``prepare_magnet_plan``, a switch from the type id of an
      element to the ``<Class>_prepare_plan`` of its class. It is generated
      here, because the type ids depend on the classes of the build;
    - the header headers/track_magnet_plans.h, with the kernel
      ``prepare_magnet_plans``. The header is inlined rather than included, so
      that xobjects specializes its kernel for the context like the rest of
      the source.

    Without magnets, only an empty ``prepare_magnet_plans`` is defined.
    """
    header = _PLANS_HEADER.read_text()
    if not magnet_classes:
        return "\n#define XTRACK_NO_MAGNET_PLANS\n" + header

    cases = "\n".join(
        f"""    case {ii}:
        {cc.__name__.replace("Data", "")}_prepare_plan(({cc.__name__})el, plan, part);
        break;"""
        for ii, cc in magnet_classes)
    return f"""
#ifndef XTRACK_TPSA_TRACK
    #define MAGNET_PLAN_BYTES ({MAGNET_PLAN_BYTES})

// Fill the plan of the element el, of type id elem_type
GPUFUN
void prepare_magnet_plan(
    /*gpuglmem*/ void* el,
    int64_t elem_type,
    MagnetPlan* plan,
    LocalParticle* part
) {{
    switch (elem_type) {{
{cases}
    }}
}}
#endif
""" + header


def track_line_cases(magnet_ids):
    """Return the cases of the element switch of ``track_line`` for the
    magnets, given their type ids.

    A magnet is tracked from the plan in its slot, or, with tapering, from a
    plan made at the magnet. The indentation matches the generated
    ``track_line``.
    """
    if not magnet_ids:
        return ""
    case_labels = "\n".join(
        f"                        case {ii}:" for ii in magnet_ids)
    return f"""
#ifndef XTRACK_TPSA_TRACK
{case_labels}
                        {{
                            /*gpuglmem*/ const MagnetPlan* plan = (/*gpuglmem*/ const MagnetPlan*) (
                                magnet_plans + MAGNET_PLAN_BYTES * magnet_plan_slots[elem_idx]);
                            if (LocalParticle_check_track_flag(&lpart, XS_FLAG_SR_TAPER)) {{
                                track_magnet_tapered(&lpart, el, elem_type);
                            }}
                            else {{
                                track_magnet_particles(&lpart, plan);
                            }}
                            #ifdef XTRACK_GLOBAL_XY_LIMIT
                            if (plan->is_thick) global_aperture_check(&lpart);
                            #endif
                            break;
                        }}
#endif
"""


def track_line_args():
    """Return the arguments that the plans add to ``track_line``: the plans,
    and the slot of each element."""
    return [
        xo.Arg(xo.Int8, pointer=True, name="magnet_plans"),
        xo.Arg(xo.Int64, pointer=True, name="magnet_plan_slots"),
    ]


def prepare_kernel_description(tracker_data_type):
    """Return the description of the kernel ``prepare_magnet_plans``
    (headers/track_magnet_plans.h), which fills the plans."""
    return xo.Kernel(
        c_name='prepare_magnet_plans',
        args=[
            xo.Arg(xo.Int8, pointer=True, name="buffer"),
            xo.Arg(tracker_data_type, name="tracker_data"),
            xo.Arg(xo.Int8, pointer=True, name="magnet_plans"),
            xo.Arg(xo.Int64, pointer=True, name="magnet_plan_elements"),
            xo.Arg(xo.Int64, name="first_plan"),
            xo.Arg(xo.Int64, name="num_plans"),
            xo.Arg(xo.Int8, pointer=True, name="io_buffer"),
            xo.Arg(xo.UInt64, name="track_flags"),
        ],
        n_threads='num_plans',
    )


def attach_prepare_kernel(kernels):
    """Return the track kernel of a build, with the prepare kernel of the
    same build attached as ``prepare_magnet_plans``.

    The two must stay together: the switch in the prepare kernel uses the type
    ids of its build. Track kernels are cached and can be shared between
    trackers (``Tracker.track_kernel``), so the prepare kernel goes with them.
    """
    track_line = kernels['track_line']
    track_line.prepare_magnet_plans = kernels.get('prepare_magnet_plans')
    return track_line


class MagnetPlans:
    """The plans of the magnets of a line, one slot per magnet.

    Made at the first track call of a tracker. It only depends on which
    elements of the line are magnets, which is fixed once the tracker is
    built.

    Attributes:
        plans: the slots, ``MAGNET_PLAN_BYTES`` each, on the context.
        element_of_slot: for each slot, the index of its magnet in the line
            (numpy).
        slot_of_element: for each element of the line, its slot, or -1 if it
            is not a magnet (on the context, read by ``track_line``).
    """

    def __init__(self, tracker_data, context):
        is_magnet = [getattr(el, '_magnet_plan', False)
                     for el in tracker_data.elements]

        self.element_of_slot = np.flatnonzero(is_magnet).astype(np.int64)

        slot_of_element = np.full(len(is_magnet), -1, dtype=np.int64)
        slot_of_element[self.element_of_slot] = np.arange(
            len(self.element_of_slot))
        self.slot_of_element = context.nparray_to_context_array(
            slot_of_element)

        num_slots = max(len(self.element_of_slot), 1)  # no empty arrays
        self.plans = context.zeros(num_slots * MAGNET_PLAN_BYTES, dtype=np.int8)

        # Read by the prepare kernel; padded, so that it is never empty
        self._element_of_slot_ctx = context.nparray_to_context_array(
            np.concatenate([self.element_of_slot, [0]]))

    def prepare(self, track_kernel, tracker_data, io_buffer, track_flags,
                ele_start, ele_stop):
        """Fill the plans of the magnets in ``[ele_start, ele_stop)`` from the
        current values of their attributes.

        Runs the prepare kernel attached to ``track_kernel``
        (``attach_prepare_kernel``). Does nothing if there is no magnet in the
        range.
        """
        first, stop = np.searchsorted(self.element_of_slot,
                                      [ele_start, ele_stop])
        if stop <= first:
            return
        track_kernel.prepare_magnet_plans(
            buffer=tracker_data._buffer.buffer,
            tracker_data=tracker_data._element_ref_data,
            magnet_plans=self.plans,
            magnet_plan_elements=self._element_of_slot_ctx,
            first_plan=int(first),
            num_plans=int(stop - first),
            io_buffer=io_buffer,
            track_flags=track_flags,
        )

    def track_args(self):
        """Return the arguments of ``track_line`` for the plans."""
        return dict(magnet_plans=self.plans,
                    magnet_plan_slots=self.slot_of_element)
