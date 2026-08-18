"""KnobParameters: hold line variables as GTPSA parameters, and restore them.

Assigning a parameter to a line variable makes xdeps re-evaluate the dependent
expressions in the TPSA algebra, so the driven element strengths hold Taylor series in
the knob and a map tracked on the same descriptor carries d(coord)/d(knob).
"""

from __future__ import annotations

from typing import Any

from xgtpsa import Descriptor, Tpsa


class KnobParameters:
    """Line variables ``names`` held as parameters ``1..len(names)`` of ``descriptor``.

    ``apply()`` puts the parameters in, ``refresh(values)`` moves them to new knob
    values, ``teardown()`` puts plain doubles back everywhere.

    While the parameters are in, the knobs must be moved with ``refresh``: assigning a
    float to ``line.vars`` re-propagates constants and drops the knob dependence.
    """

    def __init__(self, line: Any, names: list[str], descriptor: Descriptor) -> None:
        self.line = line
        self.names = list(names)
        self.descriptor = descriptor
        for name in self.names:
            if name not in line.vars:
                raise KeyError(f"knob {name!r} is not a line variable")
        if descriptor.num_params != len(self.names):
            raise ValueError(
                f"descriptor has {descriptor.num_params} parameters, "
                f"expected {len(self.names)} (one per knob)"
            )
        self._applied = False
        self._reached = None      # element names the knob expressions reach

    def __len__(self) -> int:
        return len(self.names)

    @property
    def applied(self) -> bool:
        """Are the parameters currently in the line variables?"""
        return self._applied

    def apply(self, values: list[float] | None = None) -> None:
        """Assign parameter ``k+1`` to knob ``k``, seeded at its current value."""
        if values is None:
            values = [float(self.line[name]) for name in self.names]
        for index, (name, value) in enumerate(zip(self.names, values), start=1):
            self.line.vars[name] = self.descriptor.param(index, float(value))
        self._applied = True
        if self._reached is None:
            # The expression topology is fixed, so the reached elements are too.
            self._reached = sorted({name for name, _ in self.driven_elements()})

    def refresh(self, values: list[float]) -> None:
        """Move the knobs, re-seeding the parameters (one xdeps propagation each)."""
        self.apply(values)

    def teardown(self) -> None:
        """Plain doubles back in the variables and in every element they reached."""
        if not self._applied:
            return
        self.apply_doubles([float(self.line[name]) for name in self.names])

    def apply_doubles(self, values: list[float]) -> None:
        """Set the knobs to floats and drop TPSA storage from the elements they drive."""
        for name, value in zip(self.names, values):
            self.line.vars[name] = float(value)
        # xdeps re-propagates the new floats, but an enabled element stores them as
        # constant series, so the switch back to doubles is explicit by disabling tpsa
        # for the reached elements.
        elements = self.line.element_dict
        for name in (self._reached if self._reached is not None else elements):
            element = elements[name]
            if getattr(getattr(element, "_xobject", None), "_tpsa_enabled", 0):
                element.disable_tpsa()
        self._applied = False

    def driven_elements(self) -> list[tuple[str, str]]:
        """``(element, attribute)`` of every field currently holding a series."""
        driven = []
        for name, element in self.line.element_dict.items():
            if not getattr(getattr(element, "_xobject", None), "_tpsa_enabled", 0):
                continue
            for attr in getattr(element, "_float_or_tpsa_fields", ()):
                driven.append((name, attr))
        return sorted(driven)

    def strength_jacobian(self) -> dict[tuple[str, str], list[float]]:
        """``d strength / d knob`` per driven field.

        Distinct from ``ParticlesTpsa.param_jacobian`` (``d coord / d knob``): this is the
        upstream element-strength sensitivity that seeds the map.
        """
        jacobian = {}
        for element_name, attr in self.driven_elements():
            value = getattr(self.line.element_dict[element_name], attr)
            if isinstance(value, Tpsa):
                jacobian[(element_name, attr)] = value.param_grad()
        return jacobian

    def __repr__(self) -> str:
        return f"KnobParameters(names={self.names}, applied={self._applied})"
