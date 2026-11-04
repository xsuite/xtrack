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

Inspecting a plan
-----------------
``Tracker.get_magnet_plan(name)`` prepares the plan of one magnet and returns
it as a nested dict. The layout of ``MagnetPlan`` is described in Python by
``PLAN_DTYPE``. A kernel built with the config flag
``XTRACK_CHECK_MAGNET_PLAN_LAYOUT`` checks at compile time that it matches
the C struct (``layout_checks``); the tests do this.

Exceptions
----------
- Tapering: the plan depends on the energy of the particle at the magnet, so
  it is made at the magnet, by ``track_magnet_tapered``.
- TPSA tracking: the plan holds TPSA values and cannot be stored, so it is
  made at the magnet, in the same branch of ``track_line``. Only where the
  plan comes from differs.
"""

import numpy as np
import xobjects as xo

from .general import _pkg_root

_PLANS_HEADER = _pkg_root.joinpath("headers/track_magnet_plans.h")

# The size of a slot, at least sizeof(MagnetPlan) (checked in C)
MAGNET_PLAN_BYTES = 1536


def _struct(*fields):
    """A numpy dtype laid out like the C struct with these fields."""
    return np.dtype(list(fields), align=True)


_PTR = 'u8'  # a pointer: shown as an address, for information only
_F8_K = ('f8', (4,))  # MAGNET_K_ORDER + 1
_FLAG = 'u1'

# The structs of track_magnet_structs.h, outside TPSA tracking, field by field
_FIELD_DTYPE = _struct(
    ('knorm', _F8_K), ('kskew', _F8_K), ('k_order', 'i8'),
    ('knl', _PTR), ('ksl', _PTR), ('kl_order', 'i8'),
    ('knl_scale', 'f8'), ('length', 'f8'), ('ks', 'f8'),
    ('x0_solenoid', 'f8'), ('y0_solenoid', 'f8'),
    ('k0', 'f8'), ('k1', 'f8'), ('k1s', 'f8'),
    ('r21', ('f8', (2,))), ('r43', ('f8', (2,))), ('has_mult', _FLAG),
)
_FACE_DTYPE = _struct(
    ('flags', 'i8'), ('is_exit_face', _FLAG),
    ('angle', 'f8'), ('angle_fdown', 'f8'), ('fint', 'f8'), ('hgap', 'f8'),
    ('f1', 'f8'), ('f2', 'f8'),
    ('has_ks', _FLAG), ('ks', 'f8'),
    ('has_patch', _FLAG), ('patch_angle', 'f8'), ('patch_dx', 'f8'),
    ('field', _PTR), ('backtrack', _FLAG), ('has_work', _FLAG),
    ('do_patch', _FLAG), ('do_vector_potential', _FLAG), ('do_linear', _FLAG),
    ('do_rotate', _FLAG), ('do_bend', _FLAG), ('do_mult', _FLAG),
    ('do_quad_lw', _FLAG), ('do_quad_wedge', _FLAG), ('do_wedge', _FLAG),
    ('patch_sin', 'f8'), ('patch_cos', 'f8'), ('patch_tan', 'f8'),
    ('rot_sin', 'f8'), ('rot_cos', 'f8'), ('rot_tan', 'f8'),
    ('wedge_angle', 'f8'), ('quad_lw_g1', 'f8'), ('quad_lw_g2', 'f8'),
)
_DRIFT_DTYPE = _struct(
    ('model', 'i1'), ('k0', 'f8'), ('k1', 'f8'), ('ks', 'f8'), ('h', 'f8'),
    ('x0_solenoid', 'f8'), ('y0_solenoid', 'f8'),
)
_KICK_DTYPE = _struct(
    ('length', 'f8'), ('order', 'i8'), ('inv_factorial_order', 'f8'),
    ('knl', _PTR), ('ksl', _PTR), ('factor_knl_ksl', 'f8'),
    ('knl_main', _F8_K), ('ksl_main', _F8_K),
    ('h', 'f8'), ('hxl', 'f8'), ('htot', 'f8'),
    ('k0_h_coeff', 'f8'), ('k1_h_coeff', 'f8'),
    ('rot_frame', _FLAG), ('do_drift_correction', _FLAG), ('is_empty', _FLAG),
)
_BODY_DTYPE = _struct(
    ('length', 'f8'), ('field', _PTR),
    ('integrator', 'i1'), ('num_kicks', 'i8'),
    ('drift_only', _FLAG), ('kick_is_noop', _FLAG),
    ('drift', _DRIFT_DTYPE), ('kick', _KICK_DTYPE),
    ('inv_factorial_order', 'f8'), ('factor_knl_ksl', 'f8'), ('dks_ds', 'f8'),
    ('h_radiation', 'f8'), ('radiation_flag', 'i8'), ('radiation_record', _PTR),
)
_TRANSFORM_DTYPE = _struct(
    ('active', _FLAG), ('invalid', _FLAG), ('curved', _FLAG),
    ('shift_x', 'f8'), ('shift_y', 'f8'), ('shift_s', 'f8'),
    ('rot_x_rad', 'f8'), ('rot_y_rad', 'f8'), ('rot_s_rad', 'f8'),
    ('rot_s_rad_no_frame', 'f8'), ('anchor', 'f8'),
    ('length', 'f8'), ('angle', 'f8'), ('h', 'f8'),
)

# The layout of MagnetPlan; checked against C by layout_checks
PLAN_DTYPE = _struct(
    ('field', _FIELD_DTYPE),
    ('first_face', _FACE_DTYPE), ('second_face', _FACE_DTYPE),
    ('body', _BODY_DTYPE),
    ('first_active', _FLAG), ('second_active', _FLAG),
    ('body_active', _FLAG), ('correct_s', _FLAG),
    ('ds_curved', 'f8'), ('is_thick', _FLAG),
    ('transform', _TRANSFORM_DTYPE),
    ('knl_tot', ('f8', (21,))),  # MAGNET_MAX_MULTIPOLE_ORDER + 1
    ('ksl_tot', ('f8', (21,))),
)
assert PLAN_DTYPE.itemsize <= 1536


def _leaf_offsets(dtype, prefix=''):
    """Yield (C member designator, offset) for every field of a dtype."""
    for name in dtype.names:
        sub, offset = dtype.fields[name][:2]
        if sub.names:
            for path, sub_offset in _leaf_offsets(sub, f'{prefix}{name}.'):
                yield path, offset + sub_offset
        else:
            yield prefix + name, offset


def layout_checks():
    """Return C that checks at compile time that ``PLAN_DTYPE`` has the
    layout of ``MagnetPlan``: its size and the offset of every field."""
    checks = [f'XT_PLAN_LAYOUT(sizeof(MagnetPlan) == {PLAN_DTYPE.itemsize});']
    checks += [
        f'XT_PLAN_LAYOUT(__builtin_offsetof(MagnetPlan, {path}) == {offset});'
        for path, offset in _leaf_offsets(PLAN_DTYPE)]
    return """
#ifdef __cplusplus
    #define XT_PLAN_LAYOUT(cond) static_assert(cond, "PLAN_DTYPE differs from MagnetPlan")
#else
    #define XT_PLAN_LAYOUT(cond) _Static_assert(cond, "PLAN_DTYPE differs from MagnetPlan")
#endif
""" + "\n".join(checks) + "\n"


def plan_to_dict(plan):
    """Return a plan (a record of ``PLAN_DTYPE``) as a nested dict."""
    out = {}
    for name in plan.dtype.names:
        value = plan[name]
        if plan.dtype[name].names:
            out[name] = plan_to_dict(value)
        elif np.ndim(value):
            out[name] = np.array(value)
        else:
            out[name] = value.item()
    return out


def magnet_classes(kernel_element_classes):
    """Return the magnets among the element classes of a kernel.

    Returns a list of ``(type_id, class)``, where ``type_id`` is the index of
    the class in ``kernel_element_classes``, i.e. its type id in the kernel.
    """
    return [(ii, cc) for ii, cc in enumerate(kernel_element_classes)
            if vars(cc._DressingClass).get('_magnet_plan', False)]


def plan_source(magnet_classes, check_layout=False):
    """Return the C source of the plans, for the track kernel.

    It is made of two parts:

    - the function ``prepare_magnet_plan``, a switch from the type id of an
      element to the ``<Class>_prepare_plan`` of its class. It is generated
      here, because the type ids depend on the classes of the build;
    - the header headers/track_magnet_plans.h, with the kernel
      ``prepare_magnet_plans``. The header is inlined rather than included, so
      that xobjects specializes its kernel for the context like the rest of
      the source.

    Without magnets, there is no plan source and no prepare kernel. With
    ``check_layout``, the source also checks ``PLAN_DTYPE`` (``layout_checks``).
    """
    if not magnet_classes:
        return ""
    header = _PLANS_HEADER.read_text()

    cases = "\n".join(
        f"""    case {ii}:
        {cc.__name__.replace("Data", "")}_prepare_plan(({cc.__name__})el, plan, part);
        break;"""
        for ii, cc in magnet_classes)
    checks = ""
    if check_layout:
        checks = "\n#ifndef XTRACK_TPSA_TRACK\n" + layout_checks() + "#endif\n"
    return f"""
#ifndef XTRACK_TPSA_TRACK
    #define MAGNET_PLAN_BYTES ({MAGNET_PLAN_BYTES})
#endif

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
""" + checks + header


def track_line_cases(magnet_ids):
    """Return the cases of the element switch of ``track_line`` for the
    magnets, given their type ids.

    All magnets take the same branch: get the plan, then track the particles
    through it. Only where the plan comes from differs: its slot, or, in TPSA
    tracking, a plan made at the magnet. With tapering (never in TPSA
    tracking, which has no radiation) the plan is made at the magnet by
    ``track_magnet_tapered``. The indentation matches
    the generated ``track_line``.
    """
    if not magnet_ids:
        return ""
    case_labels = "\n".join(
        f"                        case {ii}:" for ii in magnet_ids)
    return f"""
{case_labels}
                        {{
#ifdef XTRACK_TPSA_TRACK
                            // The plan holds TPSA values: it is made here
                            MagnetPlan local_plan;
                            prepare_magnet_plan(el, elem_type, &local_plan, &lpart);
                            const MagnetPlan* plan = &local_plan;
#else
                            // The plan prepared for this track call
                            /*gpuglmem*/ const MagnetPlan* plan = (/*gpuglmem*/ const MagnetPlan*) (
                                magnet_plans + MAGNET_PLAN_BYTES * magnet_plan_slots[elem_idx]);
#endif
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

    def read(self, context, element_index):
        """Return the plan of the element ``element_index`` of the line, as it
        is in its slot (a record of ``PLAN_DTYPE``)."""
        slot = np.searchsorted(self.element_of_slot, element_index)
        if (slot == len(self.element_of_slot)
                or self.element_of_slot[slot] != element_index):
            raise ValueError(f'element {element_index} is not a magnet')
        start = slot * MAGNET_PLAN_BYTES
        raw = context.nparray_from_context_array(
            self.plans[start:start + PLAN_DTYPE.itemsize])
        return np.frombuffer(np.ascontiguousarray(raw).tobytes(),
                             dtype=PLAN_DTYPE)[0]

    def track_args(self):
        """Return the arguments of ``track_line`` for the plans."""
        return dict(magnet_plans=self.plans,
                    magnet_plan_slots=self.slot_of_element)
