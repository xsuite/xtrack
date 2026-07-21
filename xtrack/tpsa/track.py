"""VALIDATION ONLY: Tracking for the ``_num`` path.

Production TPSA tracking now flows through ``element.track(particles_tpsa)`` (see
``GtpsaBackend`` + ``xtrack.BeamElement.track``).  This helper remains for
*validation*: it runs the ``.so``'s ``_num`` twin of the element physics -- the same C
source compiled with doubles -- so a test can prove it is bit-identical to native
``line.track`` (and to the TPSA const part).  It is not used in normal tracking.
"""

from __future__ import annotations

import numpy as np
import xtrack as xt

import xgtpsa

from ._bridge_build import bridge_entry
from .particles import _COORDS, _REF_VARS
from .backend import _element_ptr, _xobject_ptr, num_bridge, type_id_for


def _refval(particles: xt.Particles, name: str, i: int) -> float:
    """A reference quantity for particle ``i`` (per-particle array or scalar var)."""
    v = np.asarray(getattr(particles, name)).reshape(-1)
    return float(v[i] if v.size > 1 else v[0])


def track_num_twin(element: xt.BeamElement, particles: xt.Particles) -> xt.Particles:
    """Track ``xt.Particles`` (doubles) through one ``element`` via the .so ``_num`` twin."""
    if not isinstance(particles, xt.Particles):
        raise TypeError(f"track_num_twin expects xt.Particles, got {type(particles)}")
    type_id = type_id_for(type(element).__name__)
    fn, call_ffi = bridge_entry("num", "xt_bridge_track_element_num")
    ffi = (
        xgtpsa.ffi()
    )  # allocates the coord double* buffers (their addresses are universal)
    el_ptr = _element_ptr(element, call_ffi)
    for i in range(len(particles.x)):
        if particles.state[i] <= 0:
            continue
        # coords are double* here; the num-flavor C casts p->x back to double*.
        coord_bufs = [ffi.new("double*", float(getattr(particles, nm)[i]))
                      for nm in _COORDS]
        refs = {r: _refval(particles, r, i) for r in _REF_VARS}
        p = num_bridge(coord_bufs, refs)
        fn(type_id, el_ptr, _xobject_ptr(p, call_ffi))
        for nm, cb in zip(_COORDS, coord_bufs):
            getattr(particles, nm)[i] = cb[0]
    return particles
