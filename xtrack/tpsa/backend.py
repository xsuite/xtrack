"""GtpsaBackend: routes ``ParticlesTpsa`` maps through libgtpsa.so.

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

from . import _gtpsa
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
    returned by ``_gtpsa.bridge_entry``). Defaults to the shared dlopen ffi used
    for the "mad_*" functions.
    """
    ffi = ffi or _gtpsa.ffi()
    buf = np.frombuffer(xobj._buffer.buffer, dtype="int8")
    return ffi.cast("void*", buf.ctypes.data + xobj._offset)


def _element_ptr(element: xt.BeamElement, ffi: cffi.FFI | None = None) -> Any:
    """cffi void* into the element's xobject buffer (offset applied)."""
    return _xobject_ptr(element._xobject, ffi)


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
            f"{context}{cls_name} has no libgtpsa.so TPSA wrapper yet "
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
    ffi = _gtpsa.ffi()
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
        fn, ffi = _gtpsa.bridge_entry("tpsa", "xt_bridge_track_element_tpsa")
        fn(type_id, _element_ptr(element, ffi), _xobject_ptr(p, ffi))
        return particles

    def track_line(
        self,
        line: xt.Line,
        particles: ParticlesTpsa,
        ele_start: int | str = 0,
        ele_stop: int | str | None = None,
        num_elements: int | None = None,
        num_turns: int | None = None,
        turn_by_turn_monitor: bool | str | xt.ParticlesMonitor | TpsaMonitor | None = None,
    ) -> ParticlesTpsa:
        """Track a ``ParticlesTpsa`` map through a contiguous element range in one C call.

        The element loop runs in C: one ``XtBridgeParticle`` struct crosses the ABI,
        the ElementRefData supplies element pointers + typeids. Only a single forward pass
        over one contiguous range is supported.

        ``turn_by_turn_monitor`` follows ``Line.track``: ``'ONE_TURN_EBE'`` records the
        FULL map before every element plus once at the end, into a ``TpsaMonitor`` left in
        ``line.record_last_track``.
        """
        ele_start, num = self._resolve_range(
            line, ele_start, ele_stop, num_elements, num_turns
        )
        mon, flag = self._resolve_monitor(line, particles, turn_by_turn_monitor, num)
        fn, ffi = _gtpsa.bridge_entry("tpsa", "xt_bridge_track_line_tpsa")
        ref_ptr = self._refdata_ptr(line, ffi)
        p = _fill_struct(particles)
        # RF cavities read the revolution time off the ring circumference.
        p.line_length = float(line.tracker._tracker_data_base.line_length)
        mon_ptr = ffi.NULL if mon is None else _xobject_ptr(mon._xobject, ffi)
        # Route A: build the per-position dispatch array + push the knob table so the
        # ~20-30 knobbed instances run the tpsa-strength kernel (all else stays double).
        # _keep pins the arrays for the duration of the C call.
        knob_ptr, _keep = self._knob_dispatch(line, particles, ele_start, num, ffi)
        fn(ref_ptr, ele_start, num, _xobject_ptr(p, ffi), mon_ptr, flag, knob_ptr)
        line.tracker.record_last_track = (
            mon  # Line.record_last_track proxies the tracker
        )
        if p.state <= 0:
            at = p.at_element
            name = line.element_names[at] if at < len(line.element_names) else "?"
            raise RuntimeError(
                f"TPSA map lost at element index {at} ('{name}'); a map past its "
                f"loss point is meaningless"
            )
        return particles

    def _knob_dispatch(
        self,
        line: xt.Line,
        particles: ParticlesTpsa,
        ele_start: int,
        num: int,
        ffi: cffi.FFI,
    ) -> tuple[Any, list[Any]]:
        """Route A knob setup. Returns ``(knob_dispatch_ptr, keepalive)``.

        ``knob_dispatch_ptr`` is ``NULL`` when the map carries no knobs (pure-double
        path). Otherwise it is an ``int64_t[num]`` array holding ``real_typeid + 1``
        at every knobbed instance's loop position and 0 elsewhere. The
        knob table (field address -> parametric TPSA) is pushed via ``xt_knob_set_table``
        before tracking. Both the address table and the dispatch array are rebuilt every
        track (xobjects buffers may realloc). ``keepalive`` must outlive the C call.
        """
        knobs = particles.knobs
        if knobs is None:
            return ffi.NULL, []

        knobbed = {e for e, _ in knobs._targets}
        for name in knobbed:  # edge sensitivities are const in this milestone -> refuse
            el = line.element_dict[name]
            if getattr(el, "edge_entry_active", 0) or getattr(
                el, "edge_exit_active", 0
            ):
                raise NotImplementedError(
                    f"knobbed element '{name}' has active edges; turn edges off "
                    f"(edge sensitivities are not parametric in this milestone)"
                )

        # Per-position dispatch array (real_typeid + 1 marks a knobbed instance).
        kd = np.zeros(num, dtype=np.int64)
        for j in range(num):
            name = line.element_names[ele_start + j]
            if name in knobbed:
                kd[j] = type_id_for(type(line.element_dict[name]).__name__) + 1

        # Push the knob table (field addr -> parametric TPSA). Cast everything to the
        # bridge ffi via raw integer addresses (cross-ffi cdata is not interchangeable).
        addrs, ptrs = knobs.table()
        shared = _gtpsa.ffi()
        addr_arr = ffi.new("void*[]", [ffi.cast("void*", int(a)) for a in addrs])
        tpsa_arr = ffi.new(
            "void*[]",
            [ffi.cast("void*", int(shared.cast("uintptr_t", p))) for p in ptrs],
        )
        proto = ffi.cast("void*", int(shared.cast("uintptr_t", particles.coords[0]._p)))
        set_fn, _ = _gtpsa.bridge_entry("tpsa", "xt_knob_set_table")
        set_fn(addr_arr, tpsa_arr, proto, len(addrs))

        kd_ptr = ffi.cast("int64_t*", kd.ctypes.data)
        return kd_ptr, [kd, addr_arr, tpsa_arr]

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
        ffi = ffi or _gtpsa.ffi()
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
