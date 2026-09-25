# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import xobjects as xo

from ..base_element import BeamElement


_FIELD_VALUE_NAMES = (
    'phi',
    'Bx', 'By', 'Bs',
    'Ax', 'Ay', 'As',
    'dAx_dx', 'dAx_dy', 'dAx_ds',
    'dAs_dx', 'dAs_dy', 'dAs_ds',
)
_FIELD_VALUE_DTYPE = np.dtype([
    (name, np.float64) for name in _FIELD_VALUE_NAMES
])


def _bfield_expansion_get_field(element, x, y, s_local):
    """Run a field-expansion evaluation kernel on broadcast input arrays."""
    context = element._context

    def _to_numpy(value):
        if isinstance(value, context.nplike_array_type):
            value = context.nparray_from_context_array(value)
        return np.asarray(value, dtype=np.float64)

    x_arr, y_arr, s_arr = np.broadcast_arrays(
        _to_numpy(x), _to_numpy(y), _to_numpy(s_local) + element.s_start)
    output_shape = x_arr.shape
    n_points = x_arr.size

    if not element.straight and np.any(1.0 + element.h * x_arr == 0.0):
        raise ValueError("x contains a point on the singular curved coordinate axis")

    field_values = context.zeros(
        n_points * len(_FIELD_VALUE_NAMES), dtype=np.float64)

    if n_points:
        x_context = context.nparray_to_context_array(
            np.ascontiguousarray(x_arr).reshape(-1))
        y_context = context.nparray_to_context_array(
            np.ascontiguousarray(y_arr).reshape(-1))
        s_context = context.nparray_to_context_array(
            np.ascontiguousarray(s_arr).reshape(-1))

        n_values = int(element._ncoef) * int(element._nm)
        work_v = context.zeros(n_points * n_values, dtype=np.float64)
        work_d1 = context.zeros(n_points * n_values, dtype=np.float64)
        work_d2 = context.zeros(n_points * n_values, dtype=np.float64)
        work_q = context.zeros(n_points * int(element._nq), dtype=np.float64)

        element.compile_kernels(only_if_needed=True)
        kernel = context.kernels[element._field_evaluation_kernel_name]
        kernel(
            el=element,
            x=x_context,
            y=y_context,
            s=s_context,
            n_points=n_points,
            field_values=field_values,
            work_v=work_v,
            work_d1=work_d1,
            work_d2=work_d2,
            work_q=work_q,
        )

    field_values = context.nparray_from_context_array(field_values)
    return np.asarray(field_values).view(_FIELD_VALUE_DTYPE).reshape(output_shape)


class _BFieldExpansionArray(np.lib.mixins.NDArrayOperatorsMixin):
    """Fixed-shape coefficient view with controlled writes, including slices.

    NumPy conversions are detached copies: writable buffer views would bypass
    the expansion-cache update. All writes through this view notify the element.
    """

    def __init__(self, element, name, indices=None, readonly=False):
        self._element = element
        self._name = name
        shape = getattr(element, '_' + name).shape
        self._indices = (np.arange(np.prod(shape)).reshape(shape)
                         if indices is None else indices)
        self._readonly = readonly
        self._inplace_result = False

    @property
    def shape(self):
        return self._indices.shape

    @property
    def size(self):
        return self._indices.size

    @property
    def ndim(self):
        return self._indices.ndim

    @property
    def dtype(self):
        return np.dtype(np.float64)

    def __len__(self):
        return len(self._indices)

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __array__(self, dtype=None, copy=None):
        values = self._element._context.nparray_from_context_array(
            getattr(self._element, '_' + self._name))
        return np.array(values.ravel()[self._indices], dtype=dtype, copy=True)

    def __repr__(self):
        return repr(np.asarray(self))

    def __getitem__(self, index):
        indices = self._indices[index]
        if np.isscalar(indices):
            return np.asarray(self)[index]
        return type(self)(self._element, self._name, indices, self._readonly)

    def __setitem__(self, index, value):
        if self._readonly:
            raise ValueError(f'{self._name} is read-only')
        self._element._set_coefficients(self._name, self._indices[index], value)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        outputs = kwargs.get('out', ())
        targets = outputs if method != 'at' else inputs[:1]
        for target in targets:
            if isinstance(target, type(self)) and target._readonly:
                raise ValueError(f'{target._name} is read-only')
        arrays = tuple(np.asarray(v) if isinstance(v, type(self)) else v
                       for v in inputs)
        if outputs:
            kwargs['out'] = tuple(np.asarray(v) if isinstance(v, type(self)) else v
                                  for v in outputs)
        result = getattr(ufunc, method)(*arrays, **kwargs)
        if method == 'at' and isinstance(inputs[0], type(self)):
            inputs[0][...] = arrays[0]
        for original, updated in zip(outputs, kwargs.get('out', ())):
            if isinstance(original, type(self)):
                original[...] = updated
                original._inplace_result = True
        if outputs:
            results = result if ufunc.nout > 1 else (result,)
            results = tuple(original if original is not None else value
                            for original, value in zip(outputs, results))
            return results[0] if ufunc.nout == 1 else results
        return result

    def copy(self):
        return np.asarray(self)

    def _to_dict(self):
        return self.copy()

    def tolist(self):
        return self.copy().tolist()

    def reshape(self, *shape):
        return type(self)(self._element, self._name,
                          self._indices.reshape(*shape), self._readonly)

    def fill(self, value):
        self[...] = value


class _BFieldExpansionGeometry:
    _geometry_rename = {
        'length': '_length',
        'h': '_h',
        'straight': '_straight',
        'angle': '_angle',
        's_start': '_s_start',
        'num_phi': '_num_phi',
        'nstep': '_nstep',
        'knc': '_knc',
        'ksc': '_ksc',
        'ksol': '_ksol',
        'knl': '_knl',
        'ksl': '_ksl',
        'ksoll': '_ksoll',
    }

    @staticmethod
    def _coefficient_arrays(ksc, knc, ksol):
        ksol = np.asarray(ksol, dtype=np.float64)
        if ksol.ndim != 1 or not ksol.size:
            raise ValueError('ksol must be a nonempty one-dimensional array')
        transverse = []
        for name, values in [('ksc', ksc), ('knc', knc)]:
            values = np.asarray(values, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != ksol.size:
                raise ValueError(f'{name} must have shape (n, {ksol.size})')
            transverse.append(values)
        return transverse[0], transverse[1], ksol

    @staticmethod
    def _resolve_num_phi(num_phi, na, nb, deg, straight):
        if isinstance(num_phi, str) and num_phi == 'auto':
            # For a seed x**m*s**d in phi_p (p=0 or 1), the straight
            # recurrence -(d_x**2 + d_s**2) terminates at
            # phi_{p + 2*(m//2 + d//2)}. Keep one more y power for Ax/As.
            # Use allocated shapes, not nonzero entries, so subsequent
            # coefficient updates and deferred expressions remain covered.
            phi_even = 2 * (na // 2 + deg // 2)
            phi_odd = 1 + 2 * ((nb - 1) // 2 + deg // 2) if nb else 0
            # The integrated ksol seed has degree <= deg (trailing zero).
            num_phi = max(phi_even, phi_odd, 2 * (deg // 2)) + 1
            if not straight:
                # D_h = D_0 + h*(d_x - 2*x*d_s**2) + O(h**2).
                # One insertion of the linear-h operator permits at most
                # one extra recurrence step, i.e. two extra powers of y.
                num_phi += 2
        if not isinstance(num_phi, (int, np.integer)) or num_phi < 0:
            raise ValueError("num_phi must be 'auto' or a nonnegative integer")
        return int(num_phi)

    @property
    def num_phi(self):
        """Resolved expansion order, fixed by the allocated coefficient cache."""
        return self._num_phi

    @num_phi.setter
    def num_phi(self, value):
        value = self._resolve_num_phi(
            value, max(self.na, len(self.ksl)), max(self.nb, len(self.knl)),
            self.deg, self.straight)
        # Environment.new writes constructor arguments back to the element.
        # Accept the same order (including 'auto'), but do not invalidate
        # the allocated cache or references from existing slices.
        if value != self._num_phi:
            raise ValueError('num_phi is fixed at construction; create a new '
                             'BFieldExpansion to change the expansion order')

    @staticmethod
    def _check_spin_tracking(particles):
        if particles is not None and hasattr(particles, 'spin_x'):
            for name in ('spin_x', 'spin_y', 'spin_z'):
                spin = particles._context.nparray_from_context_array(getattr(particles, name))
                if np.any(spin != 0):
                    raise NotImplementedError('BFieldExpansion does not support spin tracking.')

    def track(self, particles=None, increment_at_element=False):
        self._check_spin_tracking(particles)
        return super().track(particles, increment_at_element=increment_at_element)

    @property
    def knc(self):
        return _BFieldExpansionArray(self, 'knc')

    @knc.setter
    def knc(self, value):
        self._check_inplace_result('knc', value)

    @property
    def ksc(self):
        return _BFieldExpansionArray(self, 'ksc')

    @ksc.setter
    def ksc(self, value):
        self._check_inplace_result('ksc', value)

    @property
    def ksol(self):
        return _BFieldExpansionArray(self, 'ksol')

    @ksol.setter
    def ksol(self, value):
        self._check_inplace_result('ksol', value)

    def _check_inplace_result(self, name, value):
        # Python assigns the result of e.g. element.knc *= 2 back to the
        # property. Accept that write-back, but never replace an array.
        if (isinstance(value, _BFieldExpansionArray)
                and value._element is self and value._name == name
                and value._inplace_result):
            value._inplace_result = False
            return
        raise AttributeError(f'{name} cannot be reassigned; update {name}[...] instead')

    @property
    def knl(self):
        # These views contain user inputs, not the integrated profile totals.
        # Controlled writes keep the combined field cache up to date.
        return _BFieldExpansionArray(self, 'knl')

    @knl.setter
    def knl(self, value):
        self._set_hard_edge_strengths('knl', value)

    @property
    def ksl(self):
        return _BFieldExpansionArray(self, 'ksl')

    @ksl.setter
    def ksl(self, value):
        self._set_hard_edge_strengths('ksl', value)

    def _set_hard_edge_strengths(self, name, value):
        if (isinstance(value, _BFieldExpansionArray)
                and value._element is self and value._name == name
                and value._inplace_result):
            self._check_inplace_result(name, value)
            return
        values = np.asarray(value, dtype=float)
        shape = getattr(self, '_' + name).shape
        if values.shape != shape:
            raise ValueError(f'{name} must retain its allocated shape {shape}')
        getattr(self, name)[:] = values

    @property
    def ksoll(self):
        return _BFieldExpansionArray(self, 'ksoll', readonly=True)

    def _set_coefficients(self, name, index, value):
        raw = getattr(self, '_' + name)
        coefficients = self._context.nparray_from_context_array(raw).copy()
        coefficients.ravel()[index] = value
        if name in ('knl', 'ksl') and self.length == 0 and np.any(coefficients):
            raise ValueError('Nonzero knl/ksl require a nonzero length')
        raw[:] = self._context.nparray_to_context_array(coefficients)
        self._update_expansion()

    def _update_expansion(self):
        # Both construction kernels accumulate seed coefficients.
        self._c[:] = 0
        self.build_bfield_expansion(el=self)
        self._update_integrated_strengths()

    def _update_integrated_strengths(self):
        integrated = self._integrate_coefficients('ksol', self.s_start, self.length)
        self._ksoll[:] = self._context.nparray_to_context_array(integrated)

    def get_total_knl_ksl(self):
        """Return normal/skew profile integrals plus the hard-edge inputs.

        The returned NumPy arrays are detached from the element, padded to
        the same length (at least four entries), and use the usual Xtrack
        integrated-multipole convention.
        """
        return self._get_total_knl_ksl(self.s_start, self.length, weight=1.)

    def _get_total_knl_ksl(self, s_start, length, weight):
        size = max(4, self.nb, self.na, len(self.knl), len(self.ksl))
        totals = []
        for source, hard_edge in [('knc', self.knl), ('ksc', self.ksl)]:
            integrated = self._integrate_coefficients(source, s_start, length)
            total = np.zeros(size)
            total[:len(integrated)] = integrated
            total[:len(hard_edge)] += weight * np.asarray(hard_edge)
            totals.append(total)
        return tuple(totals)

    def _integrate_coefficients(self, name, s_start, length):
        # Each row uses ascending powers of the polynomial coordinate.
        powers = np.arange(1, self.deg + 2)
        weights = ((s_start + length)**powers - s_start**powers) / powers
        coefficients = self._context.nparray_from_context_array(
            getattr(self, '_' + name)).reshape(-1, self.deg + 1)
        return coefficients @ weights

    @property
    def s_start(self):
        return self._s_start

    @s_start.setter
    def s_start(self, value):
        self._s_start = value
        self._update_integrated_strengths()

    @staticmethod
    def _validate_bent_curvature(h):
        if h <= 1e-4:
            raise ValueError(
                "A curved BFieldExpansion requires h > 1e-4. "
                "Create a new BFieldExpansion with h=0 for straight geometry.")

    @property
    def straight(self):
        """Read-only geometry mode, selected at construction from h."""
        return self._straight

    @property
    def angle(self):
        """Read-only bend angle in radians, equal to length * h."""
        return self._angle

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        if value == 0 and (np.any(self.knl) or np.any(self.ksl)):
            raise ValueError('Nonzero knl/ksl require a nonzero length')
        self._length = value
        self._angle = self.length * self.h
        self.ds = self.length / self.nstep
        # The hard-edge field densities are knl/length and ksl/length.
        self._update_expansion()

    @staticmethod
    def _validate_nstep(value):
        if value <= 0 or int(value) != value:
            raise ValueError('nstep must be a positive integer')

    @property
    def nstep(self):
        return self._nstep

    @nstep.setter
    def nstep(self, value):
        self._validate_nstep(value)
        self._nstep = int(value)
        self.ds = self.length / self.nstep

    @property
    def h(self):
        return self._h

    @h.setter
    def h(self, value):
        if self.straight and value != 0:
            raise ValueError('A straight BFieldExpansion requires h=0; '
                             'create a new element for curved geometry')
        if not self.straight:
            self._validate_bent_curvature(value)
        self._h = value
        self._angle = self.length * self.h
        if not self.straight:
            self._update_expansion()


class BFieldExpansion(_BFieldExpansionGeometry, BeamElement):
    """
    Magnetic-field expansion in a straight or curved reference frame.

    Parameters
    ----------
    h : float, optional
        Reference curvature in 1/m. Zero selects straight geometry; curved
        geometry requires h > 1e-4. The geometry mode is fixed at construction.
    ksc : array, shape (na, deg+1)
        Skew field coefficients normalized by the reference magnetic rigidity.
        Row i contains the polynomial for the i-th x derivative of Bx/(B rho)
        at x=y=0. Column j multiplies s**j; transverse powers include 1/i!.
    knc : array, shape (nb, deg+1)
        Normal field coefficients, with the same convention as ksc for
        By/(B rho). The coefficient knc[i, 0] has the same normalization and
        factorial convention as Xtrack's k_i (k0, k1, k2, ...):
        By(x, 0, s)/(B rho) = sum_i sum_j knc[i, j]*s**j*x**i/i!.
        Thus knc[0, 0], knc[1, 0], and knc[2, 0] are respectively the
        constant dipole, quadrupole, and sextupole strengths. They are not
        integrated strengths; get_total_knl_ksl() returns the longitudinal
        integrals including any additional hard-edge strengths.
    knl, ksl : one-dimensional arrays, optional
        Additional integrated normal and skew hard-edge strengths, in the
        usual Xtrack convention. Within the element, knl[i]/length and
        ksl[i]/length are added to the constant terms of knc[i] and ksc[i],
        respectively, without modifying those input arrays. No extra fringe
        is added at the boundaries. Defaults to zero arrays with one entry
        per corresponding coefficient row. Input arrays may include higher
        multipole orders than knc/ksc; these are included in the allocation
        and automatic expansion order. Nonzero strengths require a nonzero
        length. Changing length preserves these integrated inputs and
        recomputes their field densities.
    ksol : array, shape (deg+1,)
        On-axis Bs/(B rho) in ascending powers of s. Include a trailing zero
        coefficient so its integral fits in the scalar-potential polynomial.
    num_phi : int or 'auto', optional
        Vertical truncation order of the scalar-potential reconstruction.
        Default 'auto' uses the coefficient array shapes and polynomial
        degree to retain the complete straight-field polynomial expansion,
        including the vector potential. This also covers initially zero
        coefficients that are changed later. Curved geometry adds two orders
        to retain every term through first order in h. Terms of higher order
        in h are generally not complete: use an explicit larger integer and
        check convergence when they matter over the transverse region of
        interest. A curved expansion generally does not terminate.

        The row count alone is insufficient: longitudinal derivatives also
        generate higher powers of y. The resolved integer is stored in
        num_phi and is fixed at construction, when the cache is allocated.

        Fields are evaluated through y**num_phi, with phi_0 through
        phi_{num_phi+1} stored internally to differentiate the scalar
        potential for By.
    s_start : float, optional
        Polynomial coordinate at the element entrance, in metres. Tracking
        evaluates the profiles from s_start to s_start + length. Default is 0.

    All coefficient arrays retain their fixed input shapes. Update values with
    ``element.knc[...] = values`` (likewise for ksc, ksol, knl and ksl).
    knl/ksl also accept assignment of an array of the same shape; knc/ksc/ksol
    cannot be reassigned. Updates rebuild the cached field expansion. NumPy
    conversions return detached copies. This element does not radiate yet,
    even when radiation is enabled for the line. Spin tracking is not
    supported and raises NotImplementedError.

    ``Environment.new`` and ``Environment.set`` accept coefficient matrices
    containing numbers or deferred expressions, without special handling.

    Thick slicing shares this element's coefficients and expansion cache.
    Each slice uses its own longitudinal interval and at least one RK4 step,
    with ceil(nstep * slice_weight) steps. Use a slicing scheme with
    ``mode='thick'``; thin slicing is not supported.

    Attributes
    ----------
    ksoll : array, shape (1,)
        Read-only integral of ksol over the same interval.
    straight : int
        Read-only geometry mode: 1 for straight, 0 for curved.
    angle : float
        Read-only bend angle in radians, updated when length or h changes.
    """

    isthick = True
    behaves_like_drift = True
    has_backtrack = True
    allow_loss_refinement = False
    allow_rot_and_shift = False
    _noexpr_fields = {'num_phi'}
    _line_attr_methods = {
        'knl': ('get_total_knl_ksl', 0),
        'ksl': ('get_total_knl_ksl', 1),
    }

    _xofields = {
        "length" : xo.Float64,
        "h": xo.Float64,
        "angle": xo.Float64,
        "straight": xo.Int64,
        "num_phi": xo.Int64,
        "deg": xo.Int64,
        "nstep": xo.Int64,
        "ds": xo.Float64,

        'ksc': xo.Float64[:, :],
        'knc': xo.Float64[:, :],
        'ksol': xo.Float64[:],
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'ksoll': xo.Float64[1],

        "na": xo.Int64,
        "nb": xo.Int64,
        "deg": xo.Int64,

        "_ncoef": xo.Int64,
        "_mmax": xo.Int64,
        "_mmin": xo.Int64,
        "_moff": xo.Int64,
        "_nm": xo.Int64,

        "_qemin": xo.Int64,
        "_nq": xo.Int64,

        "_c": xo.Float64[:],
        "_V": xo.Float64[:],
        "_D1": xo.Float64[:],
        "_D2": xo.Float64[:],
        "_Q": xo.Float64[:],

        "pkin_const": xo.Int64,
        "s_start": xo.Float64,
    }

    _rename = _BFieldExpansionGeometry._geometry_rename

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/bfieldexpansion.h"',
        '#include "xtrack/beam_elements/elements_src/create_bfieldexpansion.h"',
    ]

    _field_evaluation_kernel_name = 'BFieldExpansion_get_field'

    @property
    def _thick_slice_class(self):
        from .slice_elements_thick import ThickSliceBFieldExpansion
        return ThickSliceBFieldExpansion

    _drift_slice_class = None
    _thin_slice_class = None

    _kernels = {'build_bfield_expansion': xo.Kernel(
            c_name='build_bfield_expansion',
            args=[xo.Arg(xo.ThisClass, name='el')],
            n_threads=1,
        ),
        'BFieldExpansion_get_field': xo.Kernel(
            c_name='BFieldExpansion_get_field',
            args=[
                xo.Arg(xo.ThisClass, name='el'),
                xo.Arg(xo.Float64, pointer=True, const=True, name='x'),
                xo.Arg(xo.Float64, pointer=True, const=True, name='y'),
                xo.Arg(xo.Float64, pointer=True, const=True, name='s'),
                xo.Arg(xo.Int64, name='n_points'),
                xo.Arg(xo.Float64, pointer=True, name='field_values'),
                xo.Arg(xo.Float64, pointer=True, name='work_v'),
                xo.Arg(xo.Float64, pointer=True, name='work_d1'),
                xo.Arg(xo.Float64, pointer=True, name='work_d2'),
                xo.Arg(xo.Float64, pointer=True, name='work_q'),
            ],
            n_threads='n_points',
        ),
    }

    def __init__(self, length, ksc, knc, ksol, num_phi='auto', h=0, nstep=10,
                 s_start=0, knl=None, ksl=None, **kwargs):
        ksc, knc, ksol = self._coefficient_arrays(ksc, knc, ksol)
        knl = np.zeros(knc.shape[0]) if knl is None else np.asarray(knl, dtype=float)
        ksl = np.zeros(ksc.shape[0]) if ksl is None else np.asarray(ksl, dtype=float)
        if knl.ndim != 1 or ksl.ndim != 1:
            raise ValueError('knl and ksl must be one-dimensional arrays')
        if length == 0 and (np.any(knl) or np.any(ksl)):
            raise ValueError('Nonzero knl/ksl require a nonzero length')
        self._validate_nstep(nstep)
        kwargs['length'] = length
        straight = int(h == 0)
        if not straight:
            self._validate_bent_curvature(h)
        kwargs['h'] = h
        kwargs['angle'] = length * h
        kwargs['straight'] = straight
        kwargs['nstep'] = nstep
        kwargs['ds'] = length/nstep
        kwargs['s_start'] = s_start

        kwargs['ksc'] = ksc
        kwargs['knc'] = knc
        kwargs['ksol'] = np.asarray(ksol, dtype=np.float64)

        kwargs['na'] = ksc.shape[0]
        kwargs['nb'] = knc.shape[0]
        na = max(kwargs['na'], len(ksl))
        nb = max(kwargs['nb'], len(knl))
        kwargs['num_phi'] = self._resolve_num_phi(
            num_phi, na, nb, ksc.shape[1] - 1, straight)
        kwargs['knl'] = knl
        kwargs['ksl'] = ksl
        kwargs['ksoll'] = np.zeros(1)

        kwargs['deg'] = ksc.shape[1] - 1

        if "pkin_const" in kwargs:
            kwargs["pkin_const"] = int(kwargs["pkin_const"])
        else:
            kwargs["pkin_const"] = 0  # Default is symplectic option

        # Store one extra scalar-potential order for the y derivative in By.
        kwargs['_ncoef'] = kwargs['num_phi'] + 2

        kwargs['_mmax'] = max(na, nb - 1)
        if kwargs['straight']:
            kwargs['_mmin'] = 0
            kwargs['_moff'] = 0
            kwargs['_qemin'] = 0
        else:
            kwargs['_mmin'] = -2 * ((kwargs['_ncoef'] - 1) // 2)
            kwargs['_moff'] = -kwargs['_mmin']
            kwargs['_qemin'] = kwargs['_mmin'] - 1
        kwargs['_nq'] = (kwargs['_mmax'] + 2) - kwargs['_qemin'] + 1
        kwargs['_nm'] = kwargs['_mmax'] - kwargs['_mmin'] + 1

        kwargs.setdefault("_c", np.zeros(kwargs['_ncoef'] * kwargs['_nm'] * (kwargs['deg'] + 1)))

        kwargs.setdefault("_V", np.zeros(kwargs['_ncoef'] * kwargs['_nm']))
        kwargs.setdefault("_D1", np.zeros(kwargs['_ncoef'] * kwargs['_nm']))
        kwargs.setdefault("_D2", np.zeros(kwargs['_ncoef'] * kwargs['_nm']))
        kwargs.setdefault("_Q", np.zeros(kwargs['_nq']))

        super().__init__(**kwargs)

        self._update_expansion()

    def get_field(self, x, y, s_local):
        """Evaluate ``FieldValue`` at broadcastable ``x, y, s_local``.

        ``s_local`` is measured from this element's entrance, in metres.
        The polynomial coordinate is ``s_start + s_local``, as in tracking.

        Returns a structured NumPy array with the broadcast input shape and
        fields ``phi``, ``Bx``, ``By``, ``Bs``, ``Ax``, ``Ay``, ``As``, and
        all derivatives stored by the C ``FieldValue`` structure.
        """
        return _bfield_expansion_get_field(self, x=x, y=y, s_local=s_local)
