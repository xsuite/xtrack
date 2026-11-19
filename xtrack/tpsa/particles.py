"""ParticlesTpsa: a 6D TPSA map (one truncated power series per coordinate)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Sequence, Self
if TYPE_CHECKING:
    from collections.abc import Mapping

    from madng_tpsa._typing import Scalar, TpsaSeries, TpsaSeriesT

    from .optics import TpsaOptics

import inspect
from types import MappingProxyType
from warnings import warn

import numpy as np

import xtrack as xt
import xobjects as xo
from madng_tpsa import ffi, Descriptor, Tpsa, ComplexTpsa, TpsaTensor, TpsaMap


# TPSAs
COORDS: tuple[str, ...]  = ("x", "px", "y", "py", "zeta", "pzeta")
_DERIVED_COORDS: tuple[str, ...]  = ("delta", "rvv", "rpp")
_EXTRA_COORDS: tuple[str, ...]  = ("s", "ax", "ay", "spin_x", "spin_y", "spin_z")


class TpsaParticleData(xo.Struct):
    # TPSA handles (pointers) for each coordinate and derived quantity
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

    # Regular particle attributes (non-TPSA)
    q0 = xo.Float64
    mass0 = xo.Float64
    t_sim = xo.Float64
    time_s = xo.Float64
    reference_turns = xo.Float64
    beta0 = xo.Float64
    gamma0 = xo.Float64
    p0c = xo.Float64
    chi = xo.Float64
    charge_ratio = xo.Float64
    weight = xo.Float64
    anomalous_magnetic_moment = xo.Float64
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

    # Extra attributes
    line_length = xo.Float64
    track_flags = xo.UInt64


def _check_all_fields():
    native_fields = {kk: vv._itemtype if hasattr(vv, '_itemtype') else vv
                     for kk, vv in xt.Particles._xofields.items()}
    for field in ['_capacity', '_num_active_particles', '_num_lost_particles',
                  'start_tracking_at_element']:
        native_fields.pop(field, None)
    tpsa_fields = {kk: vv._itemtype if hasattr(vv, '_itemtype') else vv
                   for kk, vv in TpsaParticleData._xofields.items()}
    missing_fields = set(native_fields) - set(tpsa_fields)
    if missing_fields:
        raise ValueError(f"Missing fields in TpsaParticleData: {missing_fields}")
    extra_fields = set(tpsa_fields) - set(native_fields)
    if extra_fields:
        raise ValueError(f"Extra fields in TpsaParticleData: {extra_fields}")


def _get_attr_from_particle(particle, field):
    try:
        attr = getattr(particle, field)
    except AttributeError as e:
        raise ValueError(f"Missing attribute {field} in Particles class") from e
    try:
        return attr[0]
    except IndexError as e:
        return attr


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

    _fields: dict[str, Any] = {ff.name: ff.ftype._c_type for ff in TpsaParticleData._fields}

    __slots__ = ('_extra_coords', '_xobject')
    _extra_coords: dict[str, Tpsa]
    _xobject: TpsaParticleData

    def __init__(
        self,
        *,
        descriptor: Descriptor | None = None,
        truncation_order: int | None = None,
        parameters: Sequence[str] | None = None,
        line_length: float = 0.0,
        track_flags: int = 0,
        **kwargs: Any,
    ) -> None:
        # Allow initialisation from TPSAs
        coords = {}
        descriptor_from_coords = None
        for coord in COORDS:
            if coord in kwargs:
                if isinstance(kwargs[coord], ComplexTpsa):
                    raise ValueError("ComplexTpsa is not supported in ParticlesTpsa")
                elif isinstance(kwargs[coord], Tpsa):
                    if descriptor_from_coords is None:
                        descriptor_from_coords = kwargs[coord].descriptor
                    elif descriptor_from_coords != kwargs[coord].descriptor:
                        raise ValueError("Inconsistent descriptors in initial coordinates")
                    coords[coord] = kwargs[coord]
                    kwargs[coord] = kwargs[coord].const_part

        # Make a temporary particle to hold the initial coordinates and reference values
        particle = xt.Particles(**kwargs)
        if len(np.atleast_1d(particle.x)) != 1:
            raise ValueError("ParticlesTpsa is a single map: pass scalar coordinates")

        # Get the truncation order. Prioritise the variable truncation_order, then the
        # truncation_order in the provided descriptor, then in the descriptor from
        # the coordinate TPSAs.
        if truncation_order is None:
            if descriptor is None:
                if descriptor_from_coords is None:
                    raise ValueError(
                        "truncation_order must be specified if descriptor is not "
                        "provided and no coordinate TPSAs are available"
                    )
                else:
                    truncation_order = descriptor_from_coords.order
            else:
                truncation_order = descriptor.order
        if any([tt.truncation_order < truncation_order for tt in coords.values()]):
            warn(
                "Provided 'truncation_order' is smaller than order from coordinates. "
                "The coordinate TPSAs are truncated."
            )
            for tt in coords.values():
                tt.truncate(truncation_order)

        # Get the descriptor. Prioritise the provided descriptor over the descriptor from
        # coordinates, then compare and unionise with the expected defaults.
        default_descriptor = Descriptor(
            variables=COORDS, order=truncation_order, parameters=parameters
        )
        if descriptor is None:
            if descriptor_from_coords is None:
                descriptor = default_descriptor
            else:
                descriptor = default_descriptor.union(
                                descriptor_from_coords, restrict_order=True
                            )
        elif descriptor_from_coords is None:
            descriptor = default_descriptor.union(descriptor, restrict_order=True)
        else:
            descriptor = default_descriptor.union(
                            descriptor, restrict_order=True
                        ).union(descriptor_from_coords, restrict_order=True)
        if descriptor.variables != COORDS:
            raise ValueError(
                f"Descriptor has unsupported variables: {descriptor.variables}"
            )

        # Initialise the canonical coordinates and a TpsaMap
        components = []
        for i, c in enumerate(COORDS):
            if c in coords:
                components.append(coords[c])
            else:
                val = descriptor.var(i + 1, _get_attr_from_particle(particle, c))
                components.append(val)
        super().__init__(components, descriptor=descriptor)

        # Initialize derived coordinates
        assert _DERIVED_COORDS == ('delta', 'rvv', 'rpp')
        pzeta = self.pzeta
        beta0 = _get_attr_from_particle(particle, 'beta0')
        ptau = beta0 * pzeta
        one_plus_delta = np.sqrt(ptau * ptau + 2 * pzeta + 1)
        self._extra_coords = {
            'delta': one_plus_delta - 1,
            'rpp': 1 / one_plus_delta,
            'rvv': one_plus_delta / (1 + beta0 * ptau),
        }

        # Initialize extra coordinates as Tpsa objects
        for name in _EXTRA_COORDS:
            val = _get_attr_from_particle(particle, name)
            self._extra_coords[name] = descriptor.constant(val)

        # Initialise C storage
        bp = TpsaParticleData()
        for nn, cc in self._fields.items():
            if nn in self.component_names:
                tt = self.named_components[nn]
                setattr(bp, nn, int(ffi.cast("uintptr_t", tt.ptr)))
            elif nn in self._extra_coords:
                tt = self._extra_coords[nn]
                setattr(bp, nn, int(ffi.cast("uintptr_t", tt.ptr)))
            elif nn == 'track_flags':
                setattr(bp, nn, track_flags)
            elif nn == 'line_length':
                setattr(bp, nn, line_length)
            else:
                val = _get_attr_from_particle(particle, nn)
                c_type = cc
                if c_type.startswith('int') or c_type.startswith('uint'):
                    val = int(val)
                setattr(bp, nn, val)
        self._xobject = bp

    def _new_like(self, values: Sequence[TpsaSeriesT]) -> Self:
        """Create a new instance like this one, but with different coordinate values."""
        kwargs = self._xobject._to_dict()
        kwargs.update(dict(zip(self.component_names, values)))
        return type(self)(**kwargs)

    @classmethod
    def _parse_dict(cls, dct: dict[str, Any]) -> dict[str, Any]:
        """Parse and validate a dict representing the map."""
        particle_data = dct.pop('particle_data', None)
        if particle_data is None:
            raise ValueError("Missing 'particle_data' in dictionary")
        descriptor, _,  tpsa_flat = super()._parse_dict(dct)
        particle_data.update(dict(zip(descriptor.variables, tpsa_flat)))
        return particle_data

    @classmethod
    def from_dict(cls, dct) -> Self:
        particle_data = cls._parse_dict(dct)
        return cls(**particle_data)

    def to_dict(self, tol: float = 0.0) -> dict[str, Any]:
        """Serialise the polynomial map and its particle/tracking state."""
        dct = super().to_dict(tol=tol)
        particle_data = self._xobject._to_dict()
        for kk in COORDS + tuple(self._extra_coords.keys()):
            particle_data.pop(kk)
        dct['particle_data'] = particle_data
        return dct

    def __getitem__(self, key: int) -> TpsaSeriesT:
        if key < 6:
            return super().__getitem__(key)
        elif key < 6 + len(self._extra_coords):
            extra_coords_keys = list(self._extra_coords.keys())
            return self._extra_coords[extra_coords_keys[key - 6]]
        else:
            raise IndexError("Index out of range")

    def __setitem__(self, key: object, value: Any) -> None:
        """Reject handle replacement on a map wired to the tracking ABI.

        Mutating a TPSA *in place* (e.g. via ``set_coefficient``) is fine; replacing
        an entire series would leave the C xobject pointing at freed/stale storage.
        """
        raise TypeError(
            "Cannot replace TPSA components in a ParticlesTpsa;"
            "modify their coefficients in place instead."
        )

    def __getattr__(self, name: str) -> Tpsa | float:
        # TpsaMap native TPSAs
        if name in COORDS:
            return super().__getattr__(name)
        # Extra coordinate TPSAs
        try:
            dct = object.__getattribute__(self, '_extra_coords')
        except AttributeError:
            dct = {}
        if name in dct:
            return dct[name]
        # TpsaParticleData fields
        if name in self._fields:
            try:
                xobject = object.__getattribute__(self, '_xobject')
            except AttributeError:
                xobject = None
            if xobject is not None:
                val = getattr(xobject, name)
                return val
                # c_type = self._fields[name]
                # if c_type.startswith('int') or c_type.startswith('uint'):
                #     return int(val)
                # else:
                #     return float(val)
        # Fallback to the superclass attribute access
        return super().__getattr__(name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Let Python handle properties and other data descriptors first.
        descriptor = inspect.getattr_static(type(self), name, None)
        if hasattr(descriptor, "__set__"):
            super().__setattr__(name, value)
            return

        try:
            components = self.named_components
        except AttributeError:
            components = {}
        try:
            components = components | self._extra_coords
        except AttributeError:
            pass
        try:
            fields = self._fields
        except AttributeError:
            fields = {}
        try:
            xobject = self._xobject
        except AttributeError:
            xobject = None

        if name in components:
            if isinstance(value, (int, float)):
                components[name].const_part = value
            else:
                raise TypeError(f"Cannot assign {type(value)} to coordinate {name}")
        elif name in fields and xobject is not None:
            setattr(xobject, name, value)
        else:
            super().__setattr__(name, value)

    def to_map(self, extend: bool = False) -> TpsaMap:
        """A TpsaMap view over existing ``Tpsa`` handles without using the ABI.

        For read-only views of a map produced elsewhere. The six series are shared,
        not copied. Not trackable.
        """
        if extend:
            return TpsaTensor(self.components + list(self._extra_coords.values()))
        else:
            return TpsaMap(self.components)

    def to_particles(self) -> xt.Particles:
        """A fresh single ``xt.Particles`` at the current const part (validation use)."""
        particles_dict = self._xobject._to_dict()
        for nn, tt in (self.named_components | self._extra_coords).items():
            particles_dict[nn] = tt.const_part
        particles_dict.pop('line_length')
        particles_dict.pop('track_flags')
        return xt.Particles.from_dict(particles_dict)

    @property
    def time_s(self) -> np.float64:
        return self._xobject.time_s

    @time_s.setter
    def time_s(self, value):
        # Match Particles: changing the clock origin invalidates its turn counter.
        self._xobject.time_s = value
        self._xobject.reference_turns = np.nan

    # Convenience methods for optics and matching
    # ===========================================

    def optics(self) -> TpsaOptics:
        """Uncoupled optics (betx, alfx, mux, dx, ...) + parameter gradients."""
        from .optics import TpsaOptics

        return TpsaOptics(self)

    def sensitivity(self, coord: str | int, knob: str | int) -> float:
        """First-order d coord / d parameter (0-based parameter index)."""
        if isinstance(knob, str):
            raise TypeError("ParticlesTpsa does not store parameter names")
        ip = knob
        return self.named_components[coord].param_grad()[ip]

    def set_jacobian(self, jacobian: Sequence[Sequence[Scalar]] | np.ndarray) -> None:
        """Replace all first-order coefficients for descriptor variables."""
        matrix = np.asarray(jacobian)
        expected_shape = (len(self), self.num_vars)
        if matrix.shape != expected_shape:
            err_mess = f'Jacobian must have shape {expected_shape}, got {matrix.shape}'
            raise ValueError(err_mess)
        if not self.is_complex and np.any(np.imag(matrix) != 0):
            err_mess = 'Cannot assign complex coefficients to a real map'
            raise TypeError(err_mess)
        for row_idx, row in enumerate(matrix):
            for variable, value in enumerate(row):
                mono = [0] * self.descriptor.monomial_length
                mono[variable] = 1
                self.set_coefficient(row_idx, mono, value)

    def set_param_jacobian(self, jacobian: Sequence[Sequence[Scalar]] | np.ndarray) -> None:
        """Replace all first-order coefficients for descriptor parameters."""
        matrix = np.asarray(jacobian)
        expected_shape = (len(self), self.num_params)
        if matrix.shape != expected_shape:
            err_mess = f'Parameter Jacobian must have shape {expected_shape}, got {matrix.shape}'
            raise ValueError(err_mess)
        if not self.is_complex and np.any(np.imag(matrix) != 0):
            err_mess = 'Cannot assign complex coefficients to a real map'
            raise TypeError(err_mess)
        for row_idx, row in enumerate(matrix):
            for parameter, value in enumerate(row):
                mono = [0] * self.descriptor.monomial_length
                mono[self.num_vars + parameter] = 1
                self.set_coefficient(row_idx, mono, value)
