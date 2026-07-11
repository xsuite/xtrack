"""The bridge ABI particle as an xobject.

``XtBridgeParticle`` is the struct that crosses the libgtpsa.so ABI: it carries
the coordinates (``tpsa_t*`` in the TPSA flavor, ``double*`` in the ``_num``
flavor, stored as ``UInt64`` addresses, since a ``tpsa_t`` is variable-size,
GTPSA-owned memory that cannot live by value in a buffer), and more parameters.

As this is an ``xo.Struct``, xobjects generates its C definition + accessors
(``XtBridgeParticle_get_x`` ...) and its cffi cdef. It allocates that
in a buffer and marshals it to C as ``buffer + _offset`` (same as element xobjects
already cross). ``gen_bridge.py`` emits its C-API alongside the elements. The physics
reads the reference/int fields through the generated ``XtBridgeParticle_get_*`` accessors.
"""

from __future__ import annotations

import xobjects as xo

# Coordinate + reference classification. Must match
# gen_bridge.py's COORD_VARS/REF_VARS (asserted there against xtrack's Particles vars).
_COORDS: tuple[str, ...] = ("x", "px", "y", "py", "zeta", "delta")
_REF_VARS: tuple[str, ...] = ("q0", "mass0", "beta0", "gamma0", "p0c",
                              "chi", "charge_ratio", "weight",
                              "anomalous_magnetic_moment")


class XtBridgeParticle(xo.Struct):
    # Coordinate handles: tpsa_t*/double* addresses (UInt64).
    x = xo.UInt64
    px = xo.UInt64
    y = xo.UInt64
    py = xo.UInt64
    zeta = xo.UInt64
    delta = xo.UInt64
    # Reference quantities (plain doubles in both flavors).
    q0 = xo.Float64
    mass0 = xo.Float64
    beta0 = xo.Float64
    gamma0 = xo.Float64
    p0c = xo.Float64
    chi = xo.Float64
    charge_ratio = xo.Float64
    weight = xo.Float64
    anomalous_magnetic_moment = xo.Float64
    # Ring circumference (RF revolution time, track_rf.h reads it).
    line_length = xo.Float64
    # Particle loss / bookkeeping.
    state = xo.Int64
    at_element = xo.Int64
    track_flags = xo.UInt64


# Names the generator asserts the struct against (drift guard).
COORD_FIELDS: list[str] = list(_COORDS)
REF_FIELDS: list[str] = list(_REF_VARS)
