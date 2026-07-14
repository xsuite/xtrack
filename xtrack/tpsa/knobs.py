"""Knobs: expand line variables into GTPSA parameters.

A ``Knobs(line, names)`` turns each named line variable into a GTPSA parameter and
expands every element strength that depends on those variables into a
``Tpsa`` in the parameters (exact for linear knob expressions, truncated to ``po``
for nonlinear ones). ``ParticlesTpsa(knobs=...)`` binds these seeds so the tracked
map carries ``d coord / d knob`` sensitivities.

This implementation supports scalar element attributes (e.g. ``k1``) only; array targets (``knl[2]``)
raise ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Any

from . import _gtpsa
from ._gtpsa import Tpsa


class _Dummy:
    """Attribute bag standing in for an element ref during expression expansion."""


class Knobs:
    def __init__(self, line: Any, names: list[str], order: int = 1) -> None:
        self.line = line
        self.names = list(names)
        self.order = int(order)
        for n in self.names:
            if n not in line.vars:
                raise KeyError(f"knob {n!r} is not a line variable")
        self._targets = self._enumerate_targets()  # sorted [(elem, attr)]
        self._desc: _gtpsa.Descriptor | None = None
        self._attr_tpsas: dict[tuple[str, str], Tpsa] = {}  # target -> expansion (owned)
        self._seed_vals: list[float] = []

    def __len__(self) -> int:
        return len(self.names)

    # --- target enumeration ---------------------------------------------- #

    def _enumerate_targets(self) -> list[tuple[str, str]]:
        """Union over knobs of the element scalar attrs they drive."""
        mgr = self.line._xdeps_manager
        elems = self.line.element_dict
        targets: set[tuple[str, str]] = set()
        for n in self.names:
            for dd in mgr.find_deps([self.line.vars[n]]):
                cls = dd.__class__.__name__
                if cls == "AttrRef" and getattr(dd._owner, "_key", None) in elems:
                    targets.add((dd._owner._key, dd._key))
                elif cls == "ItemRef":
                    owner = dd._owner
                    if owner.__class__.__name__ == "AttrRef" and \
                            getattr(owner._owner, "_key", None) in elems:
                        raise NotImplementedError(
                            f"array target {owner._owner._key}.{owner._key}"
                            f"[{dd._key}] not supported in v1 (scalar attrs only)"
                        )
        return sorted(targets)

    # --- binding / expansion --------------------------------------------- #

    def _bind(self, desc: _gtpsa.Descriptor) -> None:
        """Build param seeds and attribute TPSAs on ``desc`` (idempotent per desc).

        Called by ``ParticlesTpsa`` with the map's descriptor; standalone callers get
        a self-owned one via ``_ensure_bound`` and never touch this.
        """
        if self._desc == desc and self._seed_vals == [self.line[n] for n in self.names]:
            return
        self._expand(desc)

    def _ensure_bound(self) -> None:
        """Bind to a self-owned descriptor for standalone use (no map needed)."""
        if self._desc is None:
            self._expand(_gtpsa.Descriptor.new(
                6, 1, num_parameters=len(self), param_order=self.order))

    def _expand(self, desc: _gtpsa.Descriptor) -> None:
        mgr = self.line._xdeps_manager
        self._seed_vals = [self.line[n] for n in self.names]
        seeds = [Tpsa.param(desc, i + 1, v) for i, v in enumerate(self._seed_vals)]
        argmap = {f"a{i}": self.line.vars[n] for i, n in enumerate(self.names)}
        fdef = mgr.mk_fun("knob_fun", **argmap)
        target_elems = {e for e, _ in self._targets}
        gbl = {
            "vars": dict(mgr.containers["vars"]._owner),  # copy: never touch the real dict
            "element_refs": {e: _Dummy() for e in target_elems},
            "f": mgr.containers["f"]._owner,
        }
        exec(fdef, gbl, (lcl := {}))
        lcl["knob_fun"](*seeds)
        attr_tpsas: dict[tuple[str, str], Tpsa] = {}
        for e, a in self._targets:
            t = getattr(gbl["element_refs"][e], a, None)
            if not isinstance(t, Tpsa):  # constant strength -> lift to a constant Tpsa
                t = 0.0 * seeds[0] + float(getattr(self.line.element_dict[e], a))
            attr_tpsas[(e, a)] = t
        self._attr_tpsas = attr_tpsas
        self._desc = desc

    # --- C table --------------------------------------------------------- #

    def table(self) -> tuple[list[int], list[Any]]:
        """``(addrs, tpsa_ptrs)`` for the C knob table, rebuilt now.

        Field addresses are recomputed every call (xobjects buffers may realloc). If a
        knob's value changed since binding the expansions are rebuilt (np is small).
        Each address is sanity-checked against the live scalar; a mismatch raises.
        """
        self._ensure_bound()
        if self._seed_vals != [self.line[n] for n in self.names]:
            self._expand(self._desc)
        ffi = _gtpsa.ffi()
        addrs, ptrs = [], []
        for (e, a) in self._targets:
            addr = self._field_addr(e, a)
            read = ffi.cast("double*", addr)[0]
            live = float(getattr(self.line.element_dict[e], a))
            if abs(read - live) > 1e-12 * (1 + abs(live)):
                raise RuntimeError(
                    f"field address for {e}.{a} reads {read!r}, expected {live!r}"
                )
            addrs.append(addr)
            ptrs.append(self._attr_tpsas[(e, a)]._p)
        return addrs, ptrs

    def _field_addr(self, elem: str, attr: str) -> int:
        """Absolute address of a scalar element field in its xobjects buffer."""
        ffi = _gtpsa.ffi()
        xo = self.line.element_dict[elem]._xobject
        base = int(ffi.cast("uintptr_t", ffi.from_buffer(xo._buffer.buffer)))
        offs = {f.name: f.offset for f in type(xo)._fields}
        if attr not in offs:
            raise KeyError(f"{elem}.{attr}: no scalar field offset")
        return base + xo._offset + offs[attr]

    def strength_jacobian(self) -> dict[tuple[str, str], list[float]]:
        """``d strength / d knob`` per driven element field (auto-binds a self-owned descriptor).

        Distinct from ``ParticlesTpsa.param_jacobian`` (``d coord / d knob``): this is the
        upstream element-strength sensitivity that seeds the map.
        """
        self._ensure_bound()
        return {t: self._attr_tpsas[t].param_grad() for t in self._targets}

    def __repr__(self) -> str:
        return f"Knobs(names={self.names}, order={self.order}, targets={self._targets})"
