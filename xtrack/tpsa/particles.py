"""ParticlesTpsa: a 6D TPSA map (one truncated power series per coordinate)."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any, Iterable, Sequence

import xtrack as xt
import xobjects as xo
from madng_tpsa import ffi, Descriptor, Tpsa, TpsaMap


COORDS = ("x", "px", "y", "py", "zeta", "pzeta")
_REF_VARS: tuple[str, ...] = (
    "q0",
    "mass0",
    "t_sim",
    "beta0",
    "gamma0",
    "p0c",
    "chi",
    "charge_ratio",
    "weight",
    "anomalous_magnetic_moment",
)
_DERIVED_COORDS = ("delta", "rvv", "rpp", "s")
_LOCAL_COORDS = ("ax", "ay")
_SPIN_COORDS = ("spin_x", "spin_y", "spin_z")
_INT_FIELDS = (
    "pdg_id",
    "particle_id",
    "at_element",
    "at_turn",
    "state",
    "parent_particle_id",
)
_RNG_FIELDS = ("_rng_s1", "_rng_s2", "_rng_s3", "_rng_s4")
_TPSA_NUM_FIELDS = COORDS + _DERIVED_COORDS + _LOCAL_COORDS + _SPIN_COORDS


class TpsaParticleData(xo.Struct):
    x = xo.UInt64
    px = xo.UInt64
    y = xo.UInt64
    py = xo.UInt64
    zeta = xo.UInt64
    delta = xo.UInt64
    pzeta = xo.UInt64
    rvv = xo.UInt64
    rpp = xo.UInt64
    s = xo.UInt64
    ax = xo.UInt64
    ay = xo.UInt64
    spin_x = xo.UInt64
    spin_y = xo.UInt64
    spin_z = xo.UInt64
    q0 = xo.Float64
    mass0 = xo.Float64
    t_sim = xo.Float64
    beta0 = xo.Float64
    gamma0 = xo.Float64
    p0c = xo.Float64
    chi = xo.Float64
    charge_ratio = xo.Float64
    weight = xo.Float64
    anomalous_magnetic_moment = xo.Float64
    line_length = xo.Float64
    pdg_id = xo.Int64
    particle_id = xo.Int64
    state = xo.Int64
    at_element = xo.Int64
    at_turn = xo.Int64
    parent_particle_id = xo.Int64
    _rng_s1 = xo.UInt32
    _rng_s2 = xo.UInt32
    _rng_s3 = xo.UInt32
    _rng_s4 = xo.UInt32
    track_flags = xo.UInt64

if TYPE_CHECKING:
    from .optics import TpsaOptics


class ParticlesTpsa(TpsaMap[Tpsa]):
    """6 coordinates as TPSA around a reference orbit.  Identity map in -> element map out.

    Construction mimics ``xt.Particles``: an internal single-particle ``xt.Particles``
    (``_ref_particle``) resolves all reference algebra (``p0c``/``energy0``/``gamma0``/
    ``beta0``/...) exactly as native particles do.
    ``coords`` is the list of 6 ``Tpsa`` ([x, px, y, py, zeta, pzeta]) expanded around
    that reference orbit. The dispatcher passes their handles to the shared object.
    Read the result with ``.const_part`` (orbit) and ``.jacobian()`` (transfer matrix R),
    or per-coordinate ``.x`` etc.

    For parametric tracking, pass a descriptor with GTPSA parameters and assign
    descriptor parameters directly to participating element fields or line variables.
    """

    def __init__(
        self,
        order: int = 1,
        descriptor: Descriptor | None = None,
        **kwargs: Any,
    ) -> None:
        # Single source of truth for kwargs and derived values.
        self._ref_particle = xt.Particles(**kwargs)
        if len(np.atleast_1d(self._ref_particle.x)) != 1:
            raise ValueError("ParticlesTpsa is a single map: pass scalar coordinates")
        if descriptor is not None:
            desc = descriptor
            if desc.num_vars != 6:
                raise ValueError(
                    f"ParticlesTpsa descriptor must have 6 variables, got {desc.num_vars}"
                )
            if desc.order != order:
                raise ValueError(
                    f"descriptor is order {desc.order}, map asks for {order}"
                )
        else:
            desc = Descriptor(variables=COORDS, order=order)
        coords = [
            desc.var(i + 1, self._ref(c))
            for i, c in enumerate(COORDS)
        ]
        super().__init__(coords, coord_names=COORDS)
        pzeta = self.pzeta
        beta0 = self._ref("beta0")
        ptau = beta0 * pzeta
        one_plus_delta = np.sqrt(ptau * ptau + 2 * pzeta + 1)
        self._local_series = {
            "delta": one_plus_delta - 1,
            "rpp": 1 / one_plus_delta,
            "rvv": one_plus_delta / (1 + beta0 * ptau),
            "s": desc.constant(self._ref("s")),
        }

        self._local_series.update({
            name: Tpsa(desc) for name in _LOCAL_COORDS
        })
        self._local_series.update({
            name: desc.constant(self._ref(name)) for name in _SPIN_COORDS
        })
        self._xobject = self._build_xobject()

    def _build_xobject(self) -> TpsaParticleData:
        """The ABI struct as an xobject: coordinate handles and reference variables.

        Coordinate ``tpsa_t*`` addresses are stable for the life of the ``Tpsa`` objects
        and the shared object writes the map in place through them, so they are set once here.
        The reference (double) variables never change during tracking. The kernel copies
        this data into an unrolled ``LocalParticle`` and synchronizes tracking state back.
        """
        bp = TpsaParticleData()
        for c, t in zip(COORDS, self.coords):
            setattr(bp, c, int(ffi.cast("uintptr_t", t.ptr)))
        for c in _DERIVED_COORDS + _LOCAL_COORDS + _SPIN_COORDS:
            setattr(bp, c, int(ffi.cast("uintptr_t", self._local_series[c].ptr)))
        for r in _REF_VARS:
            setattr(bp, r, self._ref(r))
        for name in _INT_FIELDS + _RNG_FIELDS:
            setattr(bp, name, int(self._ref(name)))
        bp.track_flags = 0
        bp.line_length = 0.0
        return bp

    @classmethod
    def _from_coords(
        cls,
        coords: Iterable[Tpsa],
        ref_particle: xt.Particles | None = None,
    ) -> ParticlesTpsa:
        """A map over existing ``Tpsa`` handles without using the ABI.

        For read-only views of a map produced elsewhere. The six series are shared,
        not copied. Not trackable.
        """
        obj = object.__new__(cls)
        TpsaMap.__init__(obj, list(coords), coord_names=COORDS)
        obj._ref_particle = ref_particle
        obj._xobject = None
        obj._local_series = None
        return obj

    @property
    def delta(self) -> Tpsa:
        """Momentum deviation derived from the canonical ``pzeta`` series."""
        try:
            local_series = object.__getattribute__(self, "_local_series")
        except AttributeError:
            local_series = None
        if local_series is not None:
            return local_series["delta"]
        beta0 = self._ref("beta0")
        pzeta = self.pzeta
        ptau = beta0 * pzeta
        return np.sqrt(ptau * ptau + 2 * pzeta + 1) - 1

    def _ref(self, name: str) -> float:
        """A reference scalar as ``float`` (per-particle vars are length-1 arrays)."""
        if self._ref_particle is None:
            raise AttributeError(f"{name}: this map view carries no reference particle")
        return float(np.asarray(getattr(self._ref_particle, name)).reshape(-1)[0])

    def to_particles(self) -> xt.Particles:
        """A fresh single ``xt.Particles`` at the current const part (validation use)."""
        p = self._ref_particle.copy()
        for c, v in zip(COORDS, self.const_part):
            setattr(p, c, [v])
        return p

    def __getattr__(self, name: str) -> Tpsa | float:
        if name in _REF_VARS:
            try:
                xobject = object.__getattribute__(self, '_xobject')
            except AttributeError:
                xobject = None
            if xobject is not None:
                return float(getattr(xobject, name))
            return self._ref(name)
        return super().__getattr__(name)

    def optics(self) -> TpsaOptics:
        """Uncoupled optics (betx, alfx, mux, dx, ...) + parameter gradients."""
        from .optics import TpsaOptics

        return TpsaOptics(self)
