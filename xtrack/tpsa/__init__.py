"""TPSA module to enable tracking for Xtrack elements with TPSA's through the standalone libgtpsa.so.

Only supported for CPU. Set XTRACK_GTPSA_LIB to the .so path.

    import xtrack as xt
    import xtrack.tpsa as xtpsa
    el = xt.DriftExact(length=2.5)

    # TPSA map: track the element's own physics through it
    m = xtpsa.ParticlesTpsa(order=3, x=1e-4, px=1.5e-4, y=-1e-4, py=1e-4,
                            zeta=1e-3, delta=2e-3, p0c=7e12)
    el.track(m)                        # m.const_part (orbit), m.jacobian() (first-order)

    # equivalent call with float: same C source compiled with doubles
    p = xt.Particles(x=1e-4, px=1.5e-4, y=-1e-4, py=1e-4, zeta=1e-3, delta=2e-3, p0c=7e12)
    xtpsa.track_num_twin(el, p)        # same as native line.track
"""

from __future__ import annotations

from .particles import ParticlesTpsa
from .knobs import Knobs
from .track import track_num_twin
from .backend import GtpsaBackend
from ._tpsa_monitor import TpsaMonitor
from xtrack.tracking_backends import register_tracking_backend

register_tracking_backend(ParticlesTpsa, GtpsaBackend())

__all__ = ["ParticlesTpsa", "Knobs", "TpsaMonitor", "track_num_twin", "GtpsaBackend"]
