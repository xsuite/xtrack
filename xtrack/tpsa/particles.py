"""ParticlesTpsa: a 6D TPSA map (one truncated power series per coordinate)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence

import numpy as np
import xtrack as xt

import madng_tpsa

import xobjects as xo

COORDS: tuple[str, ...] = ("x", "px", "y", "py", "zeta", "delta")
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
_DERIVED_COORDS = ("ptau", "rvv", "rpp", "s")
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
    ptau = xo.UInt64
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


class ParticlesTpsa:
    """6 coordinates as TPSA around a reference orbit.  Identity map in -> element map out.

    Construction mimics ``xt.Particles``: an internal single-particle ``xt.Particles``
    (``_ref_particle``) resolves all reference algebra (``p0c``/``energy0``/``gamma0``/
    ``beta0``/...) exactly as native particles do.
    ``coords`` is the list of 6 ``Tpsa`` ([x, px, y, py, zeta, delta]) expanded around
    that reference orbit. The dispatcher passes their handles to the shared object.
    Read the result with ``.const_part`` (orbit) and ``.jacobian()`` (transfer matrix R),
    or per-coordinate ``.x`` etc.

    For parametric tracking, pass a descriptor with GTPSA parameters and assign
    descriptor parameters directly to participating element fields or line variables.
    """

    coords: list[madng_tpsa.Tpsa] | None = None

    def __init__(
        self,
        order: int = 1,
        descriptor: madng_tpsa.Descriptor | None = None,
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
            desc = madng_tpsa.Descriptor(6, order)
        self.coords = [
            desc.var(i + 1, self._ref(c))
            for i, c in enumerate(COORDS)
        ]
        self._local_series = {
            name: desc.constant(self._ref(name)) for name in _DERIVED_COORDS
        }
        self._local_series.update({
            name: madng_tpsa.Tpsa(desc) for name in _LOCAL_COORDS
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
        ffi = madng_tpsa.ffi()
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
        coords: Iterable[madng_tpsa.Tpsa],
        ref_particle: xt.Particles | None = None,
    ) -> ParticlesTpsa:
        """A map over existing ``Tpsa`` handles without using the ABI.

        For read-only views of a map produced elsewhere. The six series are shared,
        not copied. Not trackable.
        """
        obj = object.__new__(cls)
        obj.coords = list(coords)
        obj._ref_particle = ref_particle
        obj._xobject = None
        obj._local_series = None
        return obj

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

    def __getattr__(self, name: str) -> madng_tpsa.Tpsa | float:
        if name in COORDS:
            return self.coords[COORDS.index(name)]
        if name in _REF_VARS:
            if self._xobject is not None:
                return float(getattr(self._xobject, name))
            return self._ref(name)
        raise AttributeError(name)

    @property
    def descriptor(self) -> madng_tpsa.Descriptor:
        """The GTPSA ``Descriptor`` shared by the six coordinate series (from C)."""
        return self.coords[0].descriptor

    @property
    def order(self) -> int:
        """Truncation order, read back from the coordinate series (single source of truth)."""
        return self.coords[0].order

    @property
    def num_vars(self) -> int:
        """Number of variables of the underlying descriptor (from C)."""
        return self.coords[0].descriptor.num_vars

    @property
    def num_params(self) -> int:
        """Number of parameters (``np``) of the underlying descriptor (0 if none)."""
        return self.coords[0].descriptor.num_params

    def param_jacobian(self) -> np.ndarray:
        """(6, np) first-order sensitivities d coord / d parameter."""
        return np.array([c.param_grad() for c in self.coords])

    def sensitivity(self, coord: str | int, knob: str | int) -> float:
        """First-order d coord / d parameter (0-based parameter index)."""
        if isinstance(knob, str):
            raise TypeError("ParticlesTpsa does not store parameter names")
        ip = knob
        return self._series(coord).param_grad()[ip]

    @property
    def const_part(self) -> np.ndarray:
        """Tracked orbit: the order-0 part of each coordinate (length-6 array)."""
        return np.array([c.const_part for c in self.coords])

    def jacobian(self) -> np.ndarray:
        """The 6x6 order-1 transfer matrix R."""
        return np.array([c.grad() for c in self.coords])

    def optics(self) -> TpsaOptics:
        """Uncoupled optics (betx, alfx, mux, dx, ...) + parameter gradients."""
        from .optics import TpsaOptics

        return TpsaOptics(self)

    def set_const_part(self, values: Sequence[float] | np.ndarray) -> None:
        """Set the order-0 part (orbit) of each coordinate from a length-6 array."""
        v = np.asarray(values, dtype=float).reshape(-1)
        if v.size != 6:
            raise ValueError(f"const_part must be length 6, got {v.size}")
        for c, x in zip(self.coords, v):
            c.set_const_part(x)

    def set_jacobian(self, R: np.ndarray) -> None:
        """Set the 6x6 order-1 transfer matrix R (the 6 variables only).

        Always a 6x6 over ``[x, px, y, py, zeta, delta]``, even when the descriptor has
        parameters: only the order-1 *variable* block is written; the parameter
        columns are left untouched (``set1`` is normally not used to seed parameters).
        """
        R = np.asarray(R, dtype=float)
        if R.shape != (6, 6):
            raise ValueError(f"jacobian must be 6x6, got {R.shape}")
        for i, c in enumerate(self.coords):
            for j in range(6):
                mono = [0] * 6
                mono[j] = 1
                c.set(mono, R[i, j])

    def _series(self, coord: str | int) -> madng_tpsa.Tpsa:
        """The ``Tpsa`` output series for ``coord`` (name like ``'x'`` or index 0..5)."""
        if isinstance(coord, str):
            return self.coords[COORDS.index(coord)]
        return self.coords[coord]

    def coefficient(
        self,
        coord: str | int,
        monomials: Sequence[int] | Sequence[Sequence[int]] | np.ndarray,
    ) -> float | np.ndarray:
        """Coefficient(s) of the ``coord`` output series for one or multiple monomials.

        ``coord`` selects the output polynomial (``'x'``, ``'px'``, ... or index 0..5).
        A monomial is a length ``6 + np`` tuple of per-variable orders over
        ``[x, px, y, py, zeta, delta, p1..pnp]`` (the same keys ``monomial_coeffs``
        returns. ``np`` = number of descriptor parameters, 0 without parameters).
        ``monomials`` is one monomial (-> ``float``) or an iterable of them,
        e.g. a list of tuples or an ``(N, 6+np)`` array (-> length-N array).
        Arrays are converted to tuples internally, for example
        the x^2*px^2 term of x is ``coefficient('x', (2, 2, 0, 0, 0, 0))``.

        A malformed or beyond-order monomial raises ``ValueError`` here rather than
        letting the C library ``exit(1)`` the interpreter (see ``is_valid_monomial``).
        """
        desc = self.descriptor
        arr = np.asarray(monomials)
        rows = arr.reshape(1, -1) if arr.ndim == 1 else arr
        for row in rows:
            mono = tuple(int(v) for v in row)
            if len(mono) != desc.monomial_length or not desc.is_valid_monomial(mono):
                raise ValueError(
                    f"Invalid monomial {mono}: expected length {desc.monomial_length} "
                    f"(6 vars + {desc.num_params} params) and total order within the "
                    f"descriptor's order/param-order"
                )
        return self._series(coord).coefficient(monomials)

    def set_coefficient(
        self, coord: str | int, monomial: Sequence[int] | np.ndarray, value: float
    ) -> None:
        """Set the coefficient of one ``monomial`` of the ``coord`` output series.

        ``coord`` selects the output polynomial (``'x'``, ``'px'``, ... or index 0..5).
        ``monomial`` is a length ``6 + np`` tuple of per-variable orders over
        ``[x, px, y, py, zeta, delta, p1..pnp]`` (same shape ``coefficient`` accepts).
        A malformed or beyond-order monomial raises ``ValueError`` rather than letting
        the C library ``exit(1)`` the interpreter (see ``is_valid_monomial``).
        """
        desc = self.descriptor
        mono = tuple(int(v) for v in np.asarray(monomial).reshape(-1))
        if len(mono) != desc.monomial_length or not desc.is_valid_monomial(mono):
            raise ValueError(
                f"Invalid monomial {mono}: expected length {desc.monomial_length} "
                f"(6 vars + {desc.num_params} params) and total order within the "
                f"descriptor's order/param-order"
            )
        self._series(coord).set(mono, value)

    def monomial_coeffs(
        self, coord: str | int | None = None, tol: float = 1e-14
    ) -> dict[tuple[int, ...], float] | dict[str, dict[tuple[int, ...], float]]:
        """All ``|c| > tol`` coefficients as ``{monomial_tuple: coefficient}``.

        With ``coord`` given, returns that output series' dictionary.
        With ``coord=None``, returns ``{coord_name: {monomial_tuple: coefficient}}`` for all.
        """
        if coord is not None:
            return self._series(coord).monomial_coeffs(tol)
        return {c: s.monomial_coeffs(tol) for c, s in zip(COORDS, self.coords)}
