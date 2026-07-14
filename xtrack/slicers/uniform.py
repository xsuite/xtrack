import numpy as np

import xobjects as xo
from ..base_element import BeamElement


def _configure_grid(vname, v_grid, dv, v_range, nv):
    if v_grid is not None:
        assert dv is None, (f'd{vname} cannot be given '
                            f'if {vname}_grid is provided ')
        assert nv is None, (f'n{vname} cannot be given '
                            f'if {vname}_grid is provided ')
        assert v_range is None, (f'{vname}_range cannot be given '
                                 f'if {vname}_grid is provided')
        ddd = np.diff(v_grid)
        assert np.allclose(ddd, ddd[0]), (f'{vname}_grid must be '
                                          'uniformly spaced')
    else:
        assert v_range is not None, (f'{vname}_grid or {vname}_range '
                                     f'must be provided')
        assert len(v_range) == 2, (f'{vname}_range must be in the form '
                                   f'({vname}_min, {vname}_max)')
        if dv is not None:
            assert nv is None, (f'n{vname} cannot be given '
                                f'if d{vname} is provided ')
            v_grid = np.arange(v_range[0], v_range[1] + 0.1 * dv, dv)
        else:
            assert nv is not None, (f'n{vname} must be given '
                                    f'if d{vname} is not provided ')
            v_grid = np.linspace(v_range[0], v_range[1], nv)

    return v_grid

COORDS = ['x', 'px', 'y', 'py', 'zeta', 'delta']
SECOND_MOMENTS = {}
for c1 in COORDS:
    for c2 in COORDS:
        if c1 + '_' + c2 in SECOND_MOMENTS or c2 + '_' + c1 in SECOND_MOMENTS:
            continue
        SECOND_MOMENTS[c1 + '_' + c2] = (c1, c2)

_xof = {
    'zeta_slice_centers': xo.Float64[:],
    'z_min_edge': xo.Float64,
    'num_slices': xo.Int64,
    'dzeta': xo.Float64,
    'num_bunches': xo.Int64,
    'filled_slots': xo.Int64[:],
    'bunch_selection': xo.Int64[:],
    'bunch_spacing_zeta': xo.Float64,
    'num_particles': xo.Float64[:],
}
for coordinate in COORDS:
    _xof['sum_' + coordinate] = xo.Float64[:]
for second_moment in SECOND_MOMENTS:
    _xof['sum_' + second_moment] = xo.Float64[:]

short_second_mom_names = {}
for second_moment in SECOND_MOMENTS:
    short_second_mom_names[second_moment.replace('_', '')] = second_moment
# Gives {'xx': 'x_x', 'xpx': 'x_px', ...}

_rnm = {}

for kk in _xof.keys():
    _rnm[kk] = '_' + kk


class UniformBinSlicer(BeamElement):
    """
    A slicer each with uniform bins.

    Parameters
    ----------.
    zeta_range : Tuple
        Zeta range for each bunch.
    num_slices : int
        Number of slices per bunch.
    dzeta: float
        Width of each bin in meters
    zeta_slice_edges: np.ndarray
        z position of the slice edges
    num_bunches:
        Number of bunches
    filling_scheme: np.ndarray
        List of zeros and ones representing the filling scheme. The length
        of the array is equal to the number of slots in the machine and each
        element of the array holds a one if the slot is filled or a zero
        otherwise.
    filled_slots: np.ndarray
        Explicit physical slot numbers which are filled. This is an
        alternative to `filling_scheme`.
    bunch_selection: np.ndarray
        List of the bunches indicating which slots from the filling scheme are
        used (not all the bunches are used when using multi-processing)
    bunch_spacing_zeta : float
        Bunch spacing in meters.
    moments: str or List
        Moments considered in the slicing (if 'all' is specified all moments
        are considered)
    """

    _xofields = _xof
    _rename = _rnm

    iscollective = True

    _extra_c_sources = [
        '#include "xtrack/slicers/slicers_src/uniform_bin_slicer.h"'
    ]

    _per_particle_kernels = {
            '_slice_kernel': xo.Kernel(
                c_name='UniformBinSlicer_slice',
                args=[
                    xo.Arg(xo.Int64, name='use_bunch_index_array'),
                    xo.Arg(xo.Int64, name='use_slice_index_array'),
                    xo.Arg(xo.Int64, pointer=True, name='i_slice_particles'),
                    xo.Arg(xo.Int64, pointer=True, name='i_slot_particles')
                ]),
        }

    def __init__(self, zeta_range=None, num_slices=None, dzeta=None,
                 zeta_slice_edges=None, num_bunches=None, filling_scheme=None,
                 filled_slots=None,
                 bunch_selection=None, bunch_spacing_zeta=None,
                 moments='all', **kwargs):

        if '_xobject' in kwargs:
            self.xoinitialize(_xobject=kwargs['_xobject'])
            return

        # for now we require that the first slot of the filling scheme is filled
        # needs to be tested otherwise (especially computation of _z_a, _z_b in
        # in compressed profile)
        if filling_scheme is not None:
            assert filling_scheme[0] == 1, 'First slot must be filled'

        num_edges = None
        if num_slices is not None:
            num_edges = num_slices + 1
        _zeta_slice_edges = _configure_grid('zeta', zeta_slice_edges, dzeta,
                                            zeta_range, num_edges)
        _zeta_slice_centers = _zeta_slice_edges[:-1] + (_zeta_slice_edges[1] -
                                                        _zeta_slice_edges[0])/2


        if filled_slots is not None and filling_scheme is not None:
            raise ValueError('Only one of `filled_slots` and '
                             '`filling_scheme` can be provided')

        if filled_slots is not None:
            filled_slots = np.array(filled_slots, dtype=np.int64)
        elif filling_scheme is None and num_bunches is None:
            filled_slots = np.zeros(1, dtype=np.int64)
        elif filling_scheme is None:
            filled_slots = np.arange(num_bunches, dtype=np.int64)
        else:
            filling_scheme = np.array(filling_scheme, dtype=np.int64)
            filled_slots = filling_scheme.nonzero()[0]

        if bunch_selection is None:
            bunch_selection = np.arange(len(filled_slots), dtype=np.int64)

        bunch_spacing_zeta = bunch_spacing_zeta or 0

        all_moments = COORDS + list(SECOND_MOMENTS.keys())
        if moments == 'all':
            selected_moments = all_moments
        else:
            assert isinstance(moments, (list, tuple))
            selected_moments = []
            for mm in moments:
                if mm in COORDS:
                    selected_moments.append(mm)
                elif mm in SECOND_MOMENTS:
                    selected_moments.append(mm)
                    for cc in SECOND_MOMENTS[mm]:
                        if cc not in SECOND_MOMENTS:
                            selected_moments.append(cc)
                elif mm in short_second_mom_names:
                    selected_moments.append(short_second_mom_names[mm])
                    for cc in SECOND_MOMENTS[short_second_mom_names[mm]]:
                        if cc not in SECOND_MOMENTS:
                            selected_moments.append(cc)
                else:
                    raise ValueError(f'Unknown moment {mm}')

        allocated_sizes = {}
        for mm in all_moments:
            if mm in selected_moments:
                allocated_sizes['sum_' + mm] = ((len(bunch_selection) or 1) *
                                                len(_zeta_slice_centers))
            else:
                allocated_sizes['sum_' + mm] = 0

        assert len(bunch_selection) > 0
        num_bunches = len(bunch_selection)
        if num_bunches == 1 and bunch_selection[0] == 0:
            num_bunches = 0

        self.xoinitialize(
            zeta_slice_centers=_zeta_slice_centers,
            z_min_edge=_zeta_slice_edges[0],
            num_slices=len(_zeta_slice_centers),
            dzeta=_zeta_slice_edges[1] - _zeta_slice_edges[0],
            num_bunches=num_bunches,
            filled_slots=filled_slots,
            bunch_selection=bunch_selection,
            bunch_spacing_zeta=bunch_spacing_zeta,
            num_particles=len(bunch_selection) * len(_zeta_slice_centers),
            **allocated_sizes, **kwargs
        )

    def slice(self, particles, i_slice_particles=None, i_slot_particles=None):

        self.clear()

        if i_slot_particles is not None:
            use_bunch_index_array = 1
        else:
            use_bunch_index_array = 0
            i_slot_particles = particles.particle_id[:1]  # Dummy
        if i_slice_particles is not None:
            use_slice_index_array = 1
        else:
            use_slice_index_array = 0
            i_slice_particles = particles.particle_id[:1]  # Dummy

        self._slice_kernel(
            particles=particles,
            use_bunch_index_array=use_bunch_index_array,
            use_slice_index_array=use_slice_index_array,
            i_slice_particles=i_slice_particles,
            i_slot_particles=i_slot_particles
        )

    def track(self, particles):
        self.slice(particles)

    def clear(self):
        for coord in COORDS:
            getattr(self, '_sum_' + coord)[:] = 0
        for sec_mom in SECOND_MOMENTS:
            getattr(self, '_sum_' + sec_mom)[:] = 0
        self.num_particles[:] = 0

    @property
    def zeta_centers(self):
        """
        Array with the grid points (bin centers).
        """

        ctx2np = self._context.nparray_from_context_array

        out = np.zeros((self.num_bunches, self.num_slices))
        for ii, bunch_num in enumerate(ctx2np(self.bunch_selection)):
            z_offs = ctx2np(self._filled_slots[bunch_num]) * self.bunch_spacing_zeta
            out[ii, :] = (ctx2np(self._zeta_slice_centers) - z_offs)
        return np.atleast_1d(np.squeeze(out))

    @property
    def num_slices(self):
        """
        Number of bins
        """
        return self._num_slices

    @property
    def dzeta(self):
        """
        Bin size in meters.
        """
        return self._dzeta

    @property
    def num_bunches(self):
        """
        Number of bunches
        """
        if self._num_bunches == 0:
            return 1
        return self._num_bunches

    @property
    def zeta_range(self):
        return (self._z_min_edge, self._z_min_edge + self._dzeta * self._num_slices)

    @property
    def filled_slots(self):
        """
        Filled slots
        """
        return self._filled_slots

    @property
    def bunch_selection(self):
        """
        Number of bunches
        """
        return self._bunch_selection

    @property
    def bunch_spacing_zeta(self):
        """
        Spacing between bunches in meters
        """
        return self._bunch_spacing_zeta

    @property
    def moments(self):
        """
        List of moments that are being recorded
        """
        out = []
        for mom_name in COORDS:
            if len(getattr(self._xobject, 'sum_' + mom_name)) > 0:
                out.append(mom_name)
        for sec_mom in SECOND_MOMENTS:
            if len(getattr(self._xobject, 'sum_' + sec_mom)) > 0:
                out.append(sec_mom)

        return out

    @property
    def num_particles(self):
        """
        Number of particles per slice
        """
        return self._reshape_for_multibunch(self._num_particles)

    def sum(self, mom_name, mom_name_2=None):
        """
        Sum of the quantity mom_name per slice
        """
        if mom_name in short_second_mom_names:
            mom_name = short_second_mom_names[mom_name]
        if mom_name_2 is not None:
            mom_name = mom_name + '_' + mom_name_2
        if len(getattr(self._xobject, 'sum_' + mom_name)) == 0:
            raise ValueError(f'Moment `{mom_name}` not recorded')
        return self._reshape_for_multibunch(getattr(self, '_sum_' + mom_name))

    def mean(self, mom_name, mom_name_2=None):
        """
        Mean of the quantity mom_name per slice
        """
        out = 0 * self.num_particles
        mask_nonzero = self.num_particles > 0
        out[mask_nonzero] = (self.sum(mom_name, mom_name_2)[mask_nonzero]
                             / self.num_particles[mask_nonzero])
        return out

    def cov(self, mom_name, mom_name_2=None):
        """
        Covariance between cc1 and cc2 per slice
        """
        if mom_name_2 is None:
            if mom_name in short_second_mom_names:
                mom_name = short_second_mom_names[mom_name]
            mom_name, mom_name_2 = mom_name.split('_')
        return (self.mean(mom_name, mom_name_2) -
                self.mean(mom_name) * self.mean(mom_name_2))

    def var(self, mom_name):
        """
        Variance of the quantity cc per slice
        """
        return self.cov(mom_name, mom_name)

    def std(self, mom_name):
        """
        Standard deviation of the quantity cc per slice
        """
        return np.sqrt(self.var(mom_name))

    def _reshape_for_multibunch(self, data):
        if self.num_bunches <= 1:
            return data
        else:
            return data.reshape(self.num_bunches, self.num_slices)

    def _to_npbuffer(self):
        tmp_buffer = self._buffer.buffer[self._offset:
                                   self._offset + self._xobject._size]
        return self._context.nparray_from_context_array(tmp_buffer).astype(np.int8)

    @classmethod
    def _from_npbuffer(cls, buffer):

        assert isinstance(buffer, np.ndarray)
        assert buffer.dtype == np.int8
        xobuffer = xo.context_default.new_buffer(capacity=len(buffer))
        xobuffer.buffer = buffer
        offset = xobuffer.allocate(size=len(buffer))
        assert offset == 0
        xo_struct = cls._XoStruct
        return cls(_xobject=xo_struct._from_buffer(xobuffer))

    def __iadd__(self, other):

        assert isinstance(other, UniformBinSlicer)
        assert self.num_slices == other.num_slices
        assert self.dzeta == other.dzeta
        assert (self.filled_slots == other.filled_slots).all()
        assert (self.bunch_selection == other.bunch_selection).all()

        for cc in COORDS:
            if len(getattr(self, '_sum_' + cc)) > 0:
                assert len(getattr(other, '_sum_' + cc)) > 0
                getattr(self, '_sum_' + cc)[:] += getattr(other, '_sum_' + cc)
        for ss in SECOND_MOMENTS:
            if len(getattr(self, '_sum_' + ss)) > 0:
                assert len(getattr(other, '_sum_' + ss)) > 0
                getattr(self, '_sum_' + ss)[:] += getattr(other, '_sum_' + ss)
        self.num_particles[:] += other.num_particles

        return self

    def __add__(self, other):
        if other == 0:
            return self.copy()
        out = self.copy()
        out += other
        return out

    def __radd__(self, other):
        if other == 0:
            return self.copy()
        return self.__add__(other)
