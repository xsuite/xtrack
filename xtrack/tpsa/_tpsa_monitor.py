"""TpsaMonitor: an element-by-element record of full TPSA maps.

``xt.ParticlesMonitor`` stores into a ``ParticlesData``, i.e. six numbers per slot, so a
``ParticlesTpsa`` passing through it can only leave its const part (= orbit) behind.
The higher orders would be lost. For this reason, there is the ``TpsaMonitor``,
which records the full TPSA maps in each slot. Each slot owns six preallocated ``Tpsa`` series
and the C loop ``mad_tpsa_copy``s the whole polynomial into them, so every slot is a complete map.

The ABI struct (``XtBridgeTpsaMonitor``) is an xobject holding the ``tpsa_t*`` destination handles as ``UInt64``
addresses (slot-major), exactly like ``XtBridgeParticle`` holds the map's own coords.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xobjects as xo

from . import _gtpsa
from ._bridge_particle import _COORDS

if TYPE_CHECKING:
    # Imported lazily at runtime (in __getitem__) to keep the package import acyclic.
    import xtrack as xt

    from .particles import ParticlesTpsa


class XtBridgeTpsaMonitor(xo.Struct):
    n_slots = xo.Int64          # number of recorded slots
    n_coords = xo.Int64         # 6 (len(_COORDS)), so C never assumes the coord count
    coords = xo.UInt64[:]       # n_slots * n_coords destination tpsa_t* addresses, slot-major


class TpsaMonitor:
    """``n_slots`` full TPSA maps, written in place by the C element loop.

    Read a slot as a map (``mon[i]`` -> ``ParticlesTpsa``),
    a coordinate series across slots (``mon.x[i]`` -> ``Tpsa``),
    or the whole record at once
    (``mon.const_part`` -> ``(n_slots, 6)``, ``mon.jacobian()`` -> ``(n_slots, 6, 6)``).
    """

    def __init__(
        self,
        n_slots: int,
        descriptor: _gtpsa.Descriptor,
        ref_particle: xt.Particles | None = None,
    ) -> None:
        self.n_slots = int(n_slots)
        self.descriptor = descriptor
        self._ref_particle = ref_particle
        # Destinations must outlive the C call: mad_tpsa_copy writes into them in place.
        self._slots = [[_gtpsa.Tpsa(descriptor) for _ in _COORDS]
                       for _ in range(self.n_slots)]
        ffi = _gtpsa.ffi()
        addrs = [int(ffi.cast("uintptr_t", t._p)) for row in self._slots for t in row]
        self._xobject = XtBridgeTpsaMonitor(coords=addrs)   # slot-major destinations
        self._xobject.n_slots = self.n_slots
        self._xobject.n_coords = len(_COORDS)

    def __len__(self) -> int:
        return self.n_slots

    def __getitem__(self, i: int) -> ParticlesTpsa:
        """Slot ``i`` as a ``ParticlesTpsa`` map view (shares the recorded series)."""
        from .particles import ParticlesTpsa
        return ParticlesTpsa._from_coords(self._slots[i], self._ref_particle)

    def __getattr__(self, name: str) -> np.ndarray:
        # mon.x, mon.px, ... -> object array of the per-slot Tpsa for that coordinate.
        if name in _COORDS:
            j = _COORDS.index(name)
            return np.array([row[j] for row in self._slots], dtype=object)
        raise AttributeError(name)

    @property
    def const_part(self) -> np.ndarray:
        """The recorded orbit: ``(n_slots, num_variables)``."""
        return np.array([[t.const_part for t in row] for row in self._slots])

    def jacobian(self) -> np.ndarray:
        """The recorded transfer matrices: ``(n_slots, num_variables, num_variables)``."""
        return np.array([[t.grad() for t in row] for row in self._slots])

    def __repr__(self) -> str:
        return (f"TpsaMonitor(n_slots={self.n_slots}, "
                f"order={self._slots[0][0].order if self.n_slots else None})")
