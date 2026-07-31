"""Knobs: expand line variables into GTPSA parameters.

A ``Knobs(line, names)`` turns each named line variable into a GTPSA parameter and
expands every element strength that depends on those variables into a
``Tpsa`` in the parameters (exact for linear knob expressions, truncated to ``po``
for nonlinear ones). ``ParticlesTpsa(knobs=...)`` binds these seeds so the tracked
map carries ``d coord / d knob`` sensitivities.

This implementation supports scalar element attributes (e.g. ``k1``) only.
Array targets (``knl[2]``) raise ``NotImplementedError``.
"""

from __future__ import annotations

from typing import Any

import xgtpsa

from xgtpsa import Tpsa


class _Dummy:
    """Attribute bag standing in for an element ref during expression expansion."""


class Knobs:
    def __init__(
        self,
        line: Any,
        names: list[str],
        order: int = 1,
        descriptor: xgtpsa.Descriptor | None = None,
    ) -> None:
        self.line = line
        self.names = list(names)
        self.order = int(order)
        for n in self.names:
            if n not in line.vars:
                raise KeyError(f"knob {n!r} is not a line variable")
        # ActionTpsaTrack owns a descriptor and passes it here. Left None, the first ParticlesTpsa
        # built on these knobs supplies one, and a standalone caller falls back to a
        # minimal order-1 descriptor, enough for strength gradients.
        self.descriptor = descriptor
        self._targets = self._enumerate_targets()  # sorted [(elem, attr)]
        self._attr_tpsas: dict[tuple[str, str], Tpsa] = {}  # target -> expansion (owned)
        self._seed_vals: list[float] = []
        # Cached xdeps-generated expansion function + the globals it runs against.
        # The dependency structure lives as long as the line's expression graph, so it
        # is compiled once. Only the seed values change per solver step.
        self._expansion_function = None
        self._expansion_globals: dict[str, Any] = {}
        # Are the knob expressions linear in the knobs? (None = not yet probed.)
        # Linear means a knob change only moves constant terms, so a per-iteration
        # refresh is one set_const_part per target instead of a full re-expansion.
        self._expressions_are_linear: bool | None = None

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

    # --- expansion -------------------------------------------------------- #

    def _compile_expansion_function(self) -> None:
        """Build the xdeps expansion function once.

        ``mk_fun`` toposorts the tasks reachable from the knob refs and returns Python
        source assigning each one in order. Running it with Tpsa seeds in ``vars`` makes
        the expression graph evaluate in TPSA arithmetic. The dependency structure never
        changes between solver steps, only the seed values, so it is compiled once and
        re-run against new globals.
        """
        if self._expansion_function is not None:
            return
        mgr = self.line._xdeps_manager
        argmap = {f"a{i}": self.line.vars[n] for i, n in enumerate(self.names)}
        fdef = mgr.mk_fun("knob_fun", **argmap)
        self._expansion_globals = {
            "vars": {},  # refilled per run, never the manager's real dict
            "element_refs": {},
            "f": mgr.containers["f"]._owner,
        }
        exec(fdef, self._expansion_globals, (compiled := {}))
        self._expansion_function = compiled["knob_fun"]

    def _run_expansion(self, seeds: list[Tpsa]) -> dict[tuple[str, str], Any]:
        """Run the cached function on ``seeds``, returning the raw per-target results.

        A target whose expression turned out knob-independent comes back as a plain
        float or missing, so callers must lift it to a constant Tpsa themselves.
        """
        self._compile_expansion_function()
        mgr = self.line._xdeps_manager
        # Copy so the expansion never writes into the manager's live containers.
        self._expansion_globals["vars"] = dict(mgr.containers["vars"]._owner)
        refs = {e: _Dummy() for e, _ in self._targets}
        self._expansion_globals["element_refs"] = refs
        self._expansion_function(*seeds)
        return {(e, a): getattr(refs[e], a, None) for e, a in self._targets}

    def _expand(self) -> None:
        """Build every target's expansion from scratch on the borrowed descriptor."""
        if self.descriptor is None:
            self.descriptor = xgtpsa.Descriptor(
                6, 1, num_params=len(self.names), param_order=self.order
            )
        self._seed_vals = [self.line[n] for n in self.names]
        seeds = [
            self.descriptor.param(i + 1, v) for i, v in enumerate(self._seed_vals)
        ]
        raw = self._run_expansion(seeds)
        attr_tpsas: dict[tuple[str, str], Tpsa] = {}
        for e, a in self._targets:
            t = raw[(e, a)]
            if not isinstance(t, Tpsa):  # constant strength -> lift to a constant Tpsa
                t = 0.0 * seeds[0] + float(getattr(self.line.element_dict[e], a))
            attr_tpsas[(e, a)] = t
        self._attr_tpsas = attr_tpsas

    def _probe_linearity(self, tol: float = 1e-12) -> bool:
        """Are all knob expressions linear in the knobs?

        Expands once at parameter order 2 on a throwaway descriptor. If every
        second-order parameter coefficient vanishes the expansions are linear, so the
        parameter coefficients are knob-independent and the fast refresh is exact.
        A nonlinear expression would give a silently wrong gradient there, so it falls
        back to re-running the expansion each iteration.
        """
        # Maximum order must be at least the parameter order, so mo=2 here, not 1.
        probe_descriptor = xgtpsa.Descriptor(
            6, 2, num_params=len(self), param_order=2
        )
        seeds = [
            probe_descriptor.param(i + 1, self.line[n])
            for i, n in enumerate(self.names)
        ]
        num_vars = probe_descriptor.num_vars
        for value in self._run_expansion(seeds).values():
            if not isinstance(value, Tpsa):
                continue
            for monomial, _ in value.monomial_coeffs(tol).items():
                if sum(monomial[num_vars:]) > 1:
                    return False
        return True

    def _refresh_values(self) -> None:
        """Bring the expansions up to date with the current knob values.

        Linear case: the parameter coefficients are unchanged, so only the constant
        term moves, and xdeps has already propagated it into the element. Writing that
        live value back is O(number of targets) and keeps the same Tpsa objects, whose
        pointers the C side holds.
        """
        if not self._attr_tpsas:  # first use
            self._expand()
            return
        current = [self.line[n] for n in self.names]
        if current == self._seed_vals:
            return
        if self._expressions_are_linear is None:
            self._expressions_are_linear = self._probe_linearity()
        if not self._expressions_are_linear:
            self._expand()
            return
        for e, a in self._targets:
            self._attr_tpsas[(e, a)].set_const_part(
                float(getattr(self.line.element_dict[e], a))
            )
        self._seed_vals = current

    # --- C table --------------------------------------------------------- #

    def table(self) -> tuple[list[int], list[Any]]:
        """``(addrs, tpsa_ptrs)`` for the C knob table, rebuilt now.

        Field addresses are recomputed every call, as xobjects buffers may realloc, and
        the expansions are brought up to date with the current knob values. Each address
        is sanity-checked against the live scalar and a mismatch raises. The check stays
        even though the getters look the address up at read time, because the table
        itself is still address-keyed and a realloc between here and the track would
        leave it stale.
        """
        self._refresh_values()
        ffi = xgtpsa.ffi()
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
            ptrs.append(self._attr_tpsas[(e, a)].ptr)
        return addrs, ptrs

    def _field_addr(self, elem: str, attr: str) -> int:
        """Absolute address of a scalar element field in its xobjects buffer."""
        ffi = xgtpsa.ffi()
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
        self._refresh_values()
        return {t: self._attr_tpsas[t].param_grad() for t in self._targets}

    def __repr__(self) -> str:
        return f"Knobs(names={self.names}, order={self.order}, targets={self._targets})"
