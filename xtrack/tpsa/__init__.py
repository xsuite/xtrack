"""TPSA module to enable tracking for Xtrack elements with the ``xgtpsa`` package.

The TPSA algebra itself comes from the standalone ``xgtpsa`` package (which owns the GTPSA engine).
Everything here is related to tracking with TPSAs through normal Xtrack tracker
kernel assembly. CPU only.

    import xtrack as xt
    import xtrack.tpsa as xtpsa
    el = xt.DriftExact(length=2.5)

    # TPSA map: track the element's own physics through it
    m = xtpsa.ParticlesTpsa(order=3, x=1e-4, px=1.5e-4, y=-1e-4, py=1e-4,
                            zeta=1e-3, delta=2e-3, p0c=7e12)
    el.track(m)                        # m.const_part (orbit), m.jacobian() (first-order)
"""

from __future__ import annotations

try:
    import xgtpsa  # the GTPSA engine, import it early for a clear error
except ImportError as exc:
    raise ImportError(
        "xtrack.tpsa needs the xgtpsa package (the GTPSA engine): "
        "pip install -e <gtpsa_lib> and run its build.sh"
    ) from exc

from .particles import ParticlesTpsa
from .optics import TpsaOptics

__all__ = ["ParticlesTpsa", "TpsaOptics"]
