"""GtpsaBackend: routes ``ParticlesTpsa`` maps through the compiled bridge modules.

Registered for ``ParticlesTpsa`` via ``register_tracking_backend`` at package import.
``BeamElement.track`` / ``Line.track`` dispatch here for non-native particle objects.
The uniform ABI crosses one ``XtBridgeParticle`` struct: coords are the map's TPSA handles,
refs are floats from the internal ``xt.Particles``; the element is its xobject buffer pointer
(read in C via the generated ``<El>Data`` functions).
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import cffi
import numpy as np

import xgtpsa

from ._bridge_build import bridge_entry
from .particles import ParticlesTpsa

if TYPE_CHECKING:
    # xtrack is imported lazily below (inside the methods) to keep this module's
    # import cheap; these names exist only for editors/type checkers.
    import xobjects as xo
    import xtrack as xt

    from ._bridge_particle import XtBridgeParticle
    from ._tpsa_monitor import TpsaMonitor


def _xobject_ptr(xobj: xo.Struct | Any, ffi: cffi.FFI | None = None) -> Any:
    """cffi void* into an xobject's buffer (offset applied), cast with ``ffi``.

    Works for both a compound element (``element._xobject``) and the bridge particle
    struct (which is an ``xo.Struct``).
    ``ffi`` must be the cffi that owns the callable the pointer is passed to (the one
    returned by ``bridge_entry``). Defaults to the shared dlopen ffi used
    for the "mad_*" functions.
    """
    ffi = ffi or xgtpsa.ffi()
    buf = np.frombuffer(xobj._buffer.buffer, dtype="int8")
    return ffi.cast("void*", buf.ctypes.data + xobj._offset)


def _element_ptr(element: xt.BeamElement, ffi: cffi.FFI | None = None) -> Any:
    """cffi void* into the element's xobject buffer (offset applied)."""
    return _xobject_ptr(element._xobject, ffi)


def _flavor(particles: ParticlesTpsa) -> str:
    """The bridge flavor a map needs: ``tpsa_param`` when it carries knobs, else ``tpsa``."""
    if particles.knobs is None:
        return "tpsa"
    if getattr(particles.knobs, "mode", "table") == "slots":
        return "tpsa_slot"
    return "tpsa_param"


def registry_classes() -> list[type[xt.BeamElement]]:
    """The registry's element classes, ordered so that ``index == typeid``.

    ``TYPE_IDS`` numbers the sorted registry 0..n-1, so its key order *is* the order the
    ElementRefData UnionRef must be built in.
    """
    import xtrack as xt
    from .registry import TYPE_IDS

    return [getattr(xt, name) for name in TYPE_IDS]


def type_id_for(cls_name: str, context: str = "") -> int:
    """The bridge typeid of an element class, or ``NotImplementedError`` naming the registry."""
    from .registry import TYPE_IDS

    try:
        return TYPE_IDS[cls_name]
    except KeyError:
        raise NotImplementedError(
            f"{context}{cls_name} is not in the TPSA bridge registry yet "
            f"(supported: {', '.join(TYPE_IDS)})"
        ) from None


def num_bridge(
    coord_ptrs: Sequence[Any],
    refs: Mapping[str, float],
    line_length: float = 0.0,
) -> XtBridgeParticle:
    """A ``_num``-flavor ``XtBridgeParticle`` xobject (coord fields hold ``double*`` addresses).

    Currently only used for validation. The TPSA path fills the struct from a ``ParticlesTpsa``,
    the ``_num`` twin has no associated object, so tests build the struct from cffi ``double*`` buffers.
    Keep the buffers alive across the call.
    """
    from ._bridge_particle import XtBridgeParticle, _COORDS, _REF_VARS

    ffi = xgtpsa.ffi()
    bp = XtBridgeParticle()
    for c, b in zip(_COORDS, coord_ptrs):
        setattr(bp, c, int(ffi.cast("uintptr_t", b)))
    for r in _REF_VARS:
        setattr(bp, r, float(refs[r]))
    bp.line_length = float(line_length)
    bp.state = 1
    bp.at_element = 0
    bp.track_flags = 0
    return bp


def _fill_struct(particles: ParticlesTpsa) -> XtBridgeParticle:
    """Reset the map's ABI xobject for a fresh track and return it.

    Coordinate handles + reference doubles were set once at ``ParticlesTpsa``
    construction (they don't change during tracking); here we only reset the per-track
    fields. ``line_length`` defaults to 0, except for ``track_line`` (over a real line),
    where it's set because RF cavities read the ring circumference from it.
    """
    bp = particles._bridge
    bp.state = 1
    bp.at_element = 0
    bp.track_flags = 0
    bp.line_length = 0.0
    return bp


class GtpsaBackend:
    def __init__(self) -> None:
        # line -> (element_names tuple, ElementRefData keepalive, void* ref_ptr).
        # Cached per line; rebuilt only when the element_names tuple changes
        # In-place element *parameter* edits are picked up
        # automatically since the ElementRefData shares the tracker's buffer.
        # Weak keys: the entry (and the buffer the ElementRefData pins) dies with the line.
        self._refdata_cache: weakref.WeakKeyDictionary[
            xt.Line, tuple[tuple[str, ...], Any, int]
        ] = weakref.WeakKeyDictionary()

    def track_element(
        self, element: xt.BeamElement, particles: ParticlesTpsa
    ) -> ParticlesTpsa:
        """Track ``particles`` (a ``ParticlesTpsa`` map) through one ``element`` in place."""
        type_id = type_id_for(type(element).__name__)
        p = _fill_struct(particles)
        flavor = _flavor(particles)
        if flavor == "tpsa_param":
            self._set_knob_table(particles, flavor)
        elif flavor == "tpsa_slot":
            self._check_knobs_supported(particles.knobs)
            particles.knobs.prepare_slots()
        fn, ffi = bridge_entry(flavor, f"xt_bridge_track_element_{flavor}")
        try:
            fn(type_id, _element_ptr(element, ffi), _xobject_ptr(p, ffi))
        finally:
            if flavor == "tpsa_slot":
                particles.knobs.restore_slots()
        return particles

    def _check_knobs_supported(self, knobs) -> None:
        for elem, _ in knobs._targets:
            el = knobs.line.element_dict[elem]
            if getattr(el, "edge_entry_active", 0) or getattr(
                el, "edge_exit_active", 0
            ):
                raise NotImplementedError(
                    f"knobbed element '{elem}' has active edges. Edge knob sensitivities "
                    f"are not supported yet (set edge_entry_active/edge_exit_active to 0)"
                )
            if getattr(el, "radiation_flag", 0):
                raise NotImplementedError(
                    f"knobbed element '{elem}' has radiation_flag set. Strength tapering "
                    f"is a double-only feature (set radiation_flag to 0)"
                )

    def _set_knob_table(self, particles: ParticlesTpsa, flavor: str) -> None:
        """Push the knob table (addresses -> parametric strength TPSAs) into the C flavor.

        Rebuilt before every track: element buffers may realloc, so the field addresses
        are recomputed by ``Knobs.table()`` each call. Knobbed elements with active edges
        or with radiation on are rejected: the edge path takes the strength's const part
        and tapering scales the strengths in place. Both are not currently supported by
        TPSA.
        """
        knobs = particles.knobs
        self._check_knobs_supported(knobs)
        addrs, ptrs = knobs.table()
        fn, ffi = bridge_entry(flavor, f"xt_bridge_set_knob_table_{flavor}")
        mad_ffi = xgtpsa.ffi()
        n = len(addrs)
        if n:
            a_arr = ffi.new("void*[]", [ffi.cast("void*", int(a)) for a in addrs])
            t_arr = ffi.new(
                "void*[]",
                [ffi.cast("void*", int(mad_ffi.cast("uintptr_t", p))) for p in ptrs],
            )
        else:
            a_arr = t_arr = ffi.NULL
        proto = ffi.cast(
            "void*", int(mad_ffi.cast("uintptr_t", particles.coords[0].ptr))
        )
        fn(a_arr, t_arr, proto, n)

    def track_line(
        self,
        line: xt.Line,
        particles: ParticlesTpsa,
        ele_start: int | str = 0,
        ele_stop: int | str | None = None,
        num_elements: int | None = None,
        num_turns: int | None = None,
        turn_by_turn_monitor: bool
        | str
        | xt.ParticlesMonitor
        | TpsaMonitor
        | None = None,
        multi_element_monitor_at: list | None = None,
    ) -> ParticlesTpsa:
        """Track a ``ParticlesTpsa`` map through a contiguous element range in one C call.

        The element loop runs in C: one ``XtBridgeParticle`` struct crosses the ABI,
        the ElementRefData supplies element pointers + typeids. Only a single forward pass
        over one contiguous range is supported.

        ``turn_by_turn_monitor`` follows ``Line.track``: ``'ONE_TURN_EBE'`` records the
        FULL map before every element plus once at the end, into a ``TpsaMonitor`` left in
        ``line.record_last_track``.

        ``multi_element_monitor_at`` is a list of positions (names, indices, or ``'begin'``/
        ``'end'``): the full map is recorded at these positions in the same single C
        pass, into a ``TpsaMonitor`` left in ``line.record_multi_element_last_track``.
        Cheaper than EBE over a whole ring.
        """
        ele_start, num = self._resolve_range(
            line, ele_start, ele_stop, num_elements, num_turns
        )
        flavor = _flavor(particles)
        fn, ffi = bridge_entry(flavor, f"xt_bridge_track_line_{flavor}")
        if multi_element_monitor_at is not None:
            mon, flag, observe = self._resolve_observe(
                line, particles, multi_element_monitor_at, ele_start, num, ffi
            )
            record_attr = "record_multi_element_last_track"
        else:
            mon, flag = self._resolve_monitor(
                line, particles, turn_by_turn_monitor, num
            )
            observe = ffi.NULL
            record_attr = "record_last_track"
        # _refdata_ptr builds the tracker (relocating element buffers into it), so the
        # knob-address table MUST be computed after it, against the same buffers the C
        # loop reads. (Computing it earlier keys the table to pre-relocation addresses.)
        ref_ptr = self._refdata_ptr(line, ffi)
        if flavor == "tpsa_param":
            if particles.knobs.line is not line:
                raise ValueError(
                    "ParticlesTpsa knobs were built for a different line than the "
                    "one being tracked"
                )
            self._set_knob_table(particles, flavor)
        elif flavor == "tpsa_slot":
            if particles.knobs.line is not line:
                raise ValueError(
                    "ParticlesTpsa knobs were built for a different line than the "
                    "one being tracked"
                )
            self._check_knobs_supported(particles.knobs)
            particles.knobs.prepare_slots()
        p = _fill_struct(particles)
        # RF cavities read the revolution time off the ring circumference.
        p.line_length = float(line.tracker._tracker_data_base.line_length)
        mon_ptr = ffi.NULL if mon is None else _xobject_ptr(mon._xobject, ffi)
        try:
            fn(ref_ptr, ele_start, num, _xobject_ptr(p, ffi), mon_ptr, flag, observe)
        finally:
            if flavor == "tpsa_slot":
                particles.knobs.restore_slots()
        setattr(
            line.tracker, record_attr, mon
        )  # Line.record_*_last_track proxies the tracker
        if p.state <= 0:
            at = p.at_element
            name = line.element_names[at] if at < len(line.element_names) else "?"
            raise RuntimeError(
                f"TPSA map lost at element index {at} ('{name}'); a map past its "
                f"loss point is meaningless"
            )
        return particles

    def _resolve_monitor(
        self,
        line: xt.Line,
        particles: ParticlesTpsa,
        turn_by_turn_monitor: bool | str | xt.ParticlesMonitor | TpsaMonitor | None,
        num: int,
    ) -> tuple[xt.ParticlesMonitor | TpsaMonitor | None, int]:
        """Normalize ``turn_by_turn_monitor`` to ``(monitor | None, flag_monitor)``.

        Mirrors ``Tracker._get_monitor`` without the multi-turn cases: a map is one
        particle on one turn, so ``True`` records a single slot and ``'ONE_TURN_EBE'``
        records ``num + 1`` slots (before each element, plus the end of the range).

        ``'ONE_TURN_EBE'`` records FULL maps into a ``TpsaMonitor`` (flag 3), the TPSA
        analogue of the ``ParticlesMonitor``. Pass an explicit
        ``xt.ParticlesMonitor`` (in ``ebe_mode``, or placed in the line) to record only the
        orbit into a doubles buffer instead.
        """
        import xtrack as xt
        from ._tpsa_monitor import TpsaMonitor

        if turn_by_turn_monitor is None or turn_by_turn_monitor is False:
            return None, 0
        ctx = line.tracker._context
        if isinstance(turn_by_turn_monitor, (xt.ParticlesMonitor, TpsaMonitor)):
            if isinstance(turn_by_turn_monitor, TpsaMonitor):
                if len(turn_by_turn_monitor) < num + 1:
                    raise ValueError(
                        f"TpsaMonitor has {len(turn_by_turn_monitor)} slots, needs "
                        f"{num + 1} (one before each of the {num} elements + the end)"
                    )
                return turn_by_turn_monitor, 3
            return turn_by_turn_monitor, (
                2 if turn_by_turn_monitor.ebe_mode == 1 else 1
            )
        if turn_by_turn_monitor is True:
            mon = xt.ParticlesMonitor(
                _context=ctx, start_at_turn=0, stop_at_turn=1, particle_id_range=(0, 1)
            )
            return mon, 1
        if turn_by_turn_monitor == "ONE_TURN_EBE":
            mon = TpsaMonitor(
                num + 1, particles.descriptor, ref_particle=particles._ref_particle
            )
            return mon, 3
        raise ValueError(f"invalid turn_by_turn_monitor {turn_by_turn_monitor!r}")

    def _resolve_pos(self, line, pos, ele_start, num):
        """A position (name / index / 'begin' / 'end') -> absolute element index."""
        n = len(line.element_names)
        if isinstance(pos, str):
            if pos == "begin":
                return ele_start
            if pos == "end":
                return ele_start + num
            return line.element_names.index(pos)
        idx = int(pos)
        return idx if idx >= 0 else n + idx

    def _resolve_observe(self, line, particles, positions, ele_start, num, ffi):
        """Build the (TpsaMonitor, flag=3, observe C-array) for ``multi_element_monitor_at``.

        Records the full map at the given positions only, in one forward pass. Slots fill in
        ascending position order; ``mon.obs_names`` maps slot -> position for read-back.
        """
        from ._tpsa_monitor import TpsaMonitor

        idxs = [self._resolve_pos(line, a, ele_start, num) for a in positions]
        for a, gi in zip(positions, idxs):
            if not (ele_start <= gi <= ele_start + num):
                raise ValueError(
                    f"observe position {a!r} (index {gi}) is outside the tracked range "
                    f"[{ele_start}, {ele_start + num}]"
                )
        order = sorted(range(len(positions)), key=idxs.__getitem__)
        ks = [idxs[i] - ele_start for i in order]
        if len(set(ks)) != len(ks):
            raise ValueError(
                f"multi_element_monitor_at has duplicate positions: {positions}"
            )
        observe = ffi.new("int64_t[]", num + 1)
        for k in ks:
            observe[k] = 1
        mon = TpsaMonitor(
            len(positions), particles.descriptor, ref_particle=particles._ref_particle
        )
        mon.obs_names = [positions[i] for i in order]  # slot i -> position
        return mon, 3, observe

    def _resolve_range(
        self,
        line: xt.Line,
        ele_start: int | str,
        ele_stop: int | str | None,
        num_elements: int | None,
        num_turns: int | None,
    ) -> tuple[int, int]:
        """Normalize (ele_start, ele_stop|num_elements) to (start_index, n_elements).

        Mirrors ``Tracker._prepare_common_track`` semantics (names->indices,
        ``ele_stop==0`` means 'to the end').  A single forward pass only.
        """
        n = len(line.element_names)
        if num_turns not in (None, 1):
            raise NotImplementedError(
                "multi-turn TPSA tracking lands in Phase 5 (num_turns must be 1)"
            )
        if isinstance(ele_start, str):
            ele_start = line.element_names.index(ele_start)
        ele_start = ele_start or 0
        if not (0 <= ele_start < n):
            raise ValueError(f"ele_start {ele_start} out of range [0, {n})")

        if num_elements is not None:
            if ele_stop is not None:
                raise ValueError("Cannot use both num_elements and ele_stop")
            num = num_elements
        else:
            if isinstance(ele_stop, str):
                ele_stop = line.element_names.index(ele_stop)
            if ele_stop is None or ele_stop == 0:
                ele_stop = n  # 'to the end' (ele_stop==0 wraps to ring start)
            if ele_stop <= ele_start:
                raise NotImplementedError(
                    f"wrap-around range (ele_stop={ele_stop} <= ele_start={ele_start}) "
                    f"needs multi-turn support (Phase 5)"
                )
            num = ele_stop - ele_start

        if num < 0 or ele_start + num > n:
            raise ValueError(
                f"range [{ele_start}, {ele_start + num}) out of bounds [0, {n}]"
            )
        return ele_start, num

    def _refdata_ptr(self, line: xt.Line, ffi: cffi.FFI | None = None) -> Any:
        """cffi void* to an ElementRefData built over the full registry class set.

        The C loop reads each element's typeid from the union's typeid array,
        so the union must be built from the full sorted registry (same as
        gen_bridge). Building it from only the line's classes gives a different
        UnionRef order and mis-dispatches (SIGSEGV). Cached per line as a plain address,
        cast with the caller's ``ffi`` on each call (defaults to the shared ``mad_*`` ffi).
        """
        ffi = ffi or xgtpsa.ffi()
        from xtrack.tracker import _element_ref_data_class_from_element_classes

        names = tuple(line.element_names)
        cached = self._refdata_cache.get(line)
        if cached is not None and cached[0] == names:
            return ffi.cast("void*", cached[2])

        # Resolve through element_dict and guard every class.
        for nn in names:
            type_id_for(
                type(line.element_dict[nn]).__name__, context=f"element '{nn}': "
            )

        if not line._has_valid_tracker():
            line.build_tracker()
        RefCls = _element_ref_data_class_from_element_classes(registry_classes())
        buf = line.tracker._tracker_data_base._buffer
        erd = RefCls(elements=len(names), names=list(names), _buffer=buf)
        erd.elements = [line.element_dict[nn]._xobject for nn in names]
        raw = np.frombuffer(erd._buffer.buffer, dtype="int8")
        addr = raw.ctypes.data + erd._offset
        self._refdata_cache[line] = (names, erd, addr)  # erd kept alive
        return ffi.cast("void*", addr)
