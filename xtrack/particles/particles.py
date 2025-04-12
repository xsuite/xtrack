# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2023.                 #
# ######################################### #
import json

import numpy as np
from pathlib import Path

from scipy.constants import e as qe
from scipy.constants import c as clight
from scipy.constants import epsilon_0

import xobjects as xo
from xobjects.general import Print
from xobjects import BypassLinked

from .masses import PROTON_MASS_EV
from .masses import __dict__ as mass__dict__
from .pdg import get_pdg_id_from_name, get_properties_from_pdg_id, \
                 get_mass_from_pdg_id

LAST_INVALID_STATE = -999999999


class Particles(xo.HybridClass):
    _cname = 'ParticlesData'

    size_vars = (
        (xo.Int64, '_capacity'),
        (xo.Int64, '_num_active_particles'),
        (xo.Int64, '_num_lost_particles'),
        (xo.Int64, 'start_tracking_at_element'),
    )
    # Capacity is always kept up to date
    # the other two are placeholders to be used if needed
    # i.e. on ContextCpu

    scalar_vars = (
        (xo.Float64, 'q0'),
        (xo.Float64, 'mass0'),
        (xo.Float64, 't_sim'),
    )

    part_energy_vars = (
        (xo.Float64, 'ptau'),
        (xo.Float64, 'delta'),
        (xo.Float64, 'rpp'),
        (xo.Float64, 'rvv'),
    )

    per_particle_vars = (
        (
            (xo.Float64, 'p0c'),
            (xo.Float64, 'gamma0'),
            (xo.Float64, 'beta0'),
            (xo.Float64, 's'),
            (xo.Float64, 'zeta'),
            (xo.Float64, 'x'),
            (xo.Float64, 'y'),
            (xo.Float64, 'px'),
            (xo.Float64, 'py'),
        )
        + part_energy_vars +
        (
            (xo.Float64, 'chi'),
            (xo.Float64, 'charge_ratio'),
            (xo.Float64, 'weight'),
            (xo.Float64, 'ax'),
            (xo.Float64, 'ay'),
            (xo.Int64, 'pdg_id'),
            (xo.Int64, 'particle_id'),
            (xo.Int64, 'at_element'),
            (xo.Int64, 'at_turn'),
            (xo.Int64, 'state'),
            (xo.Int64, 'parent_particle_id'),
            (xo.UInt32, '_rng_s1'),
            (xo.UInt32, '_rng_s2'),
            (xo.UInt32, '_rng_s3'),
            (xo.UInt32, '_rng_s4')
        )
    )

    _xofields = {
        **{nn: tt for tt, nn in size_vars + scalar_vars},
        **{nn: tt[:] for tt, nn in per_particle_vars},
    }

    _extra_c_sources = [
        Path(__file__).parent.joinpath('rng_src', 'base_rng.h'),
        Path(__file__).parent.joinpath('rng_src', 'particles_rng.h'),
        '\n /*placeholder_for_local_particle_src*/ \n'
    ]

    _rename = {
        'delta': '_delta',
        'ptau': '_ptau',
        'rvv': '_rvv',
        'rpp': '_rpp',
        'p0c': '_p0c',
        'gamma0': '_gamma0',
        'beta0': '_beta0',
    }

    _kernels = {
        'Particles_initialize_rand_gen': xo.Kernel(
            c_name="Particles_initialize_rand_gen",
            args=[
                xo.Arg(xo.ThisClass, name='particles'),
                xo.Arg(xo.UInt32, pointer=True, name='seeds'),
                xo.Arg(xo.Int32, name='n_init')],
            n_threads='n_init')
    }

    def __init__(
            self,
            _capacity=None,
            _no_reorganize=False,
            **kwargs,
    ):


        """
        The Particles class contains coordinates and other data associated to a
        set of particles. Parameters can be provided as arrays of the same
        length or as scalars. If arrays are provided, the length of the arrays
        must be equal to the number of particles. If scalars are provided, the
        same value is assigned to all particles. When parameters are not
        provided, they are initialized to default values, or inferred from the
        other parameters.

        Parameters
        ----------
        _capacity: int
            The maximum number of particles that can be stored in the object.
            If not provided, it is inferred from the size of the provided
            coordinates arrays.
        s : array_like of float, optional
            Reference accumulated path length [m]
        x : array_like of float, optional
            Horizontal position [m]
        px : array_like of float, optional
            Px / (m/m0 * p0c) = beta_x gamma /(beta0 gamma0)
        y : array_like of float, optional
            Vertical position [m]
        py : array_like of float, optional
            Py / (m/m0 * p0c)
        delta : array_like of float, optional
            (Pc m0/m - p0c) /p0c
        ptau : array_like of float, optional
            (Energy m0/m - Energy0) / p0c
        pzeta : array_like of float, optional
            ptau / beta0
        rvv : array_like of float, optional
            beta / beta0
        rpp : array_like of float, optional
            m/m0 P0c / Pc = 1/(1+delta)
        zeta : array_like of float, optional
            (s - beta0 c t)
        tau : array_like of float, optional
            (s / beta0 - ct)
        mass0 : float, optional
            Reference rest mass [eV]
        q0 : float, optional
            Reference charge [e]
        p0c : array_like of float, optional
            Reference momentum [eV]
        energy0 : array_like of float, optional
            Reference energy [eV]
        gamma0 : array_like of float, optional
            Reference relativistic gamma
        beta0 : array_like of float, optional
            Reference relativistic beta
        mass_ratio : array_like of float, optional
            mass/mass0 (this is used to track particles of
            different species. Note that mass is the rest mass
            of the considered particle species and not the
            relativistic mass)
        chi : array_like of float, optional
            q / q0 * m0 / m = qratio / mratio
        charge_ratio : array_like of float, optional
            q / q0
        particle_id : array_like of int, optional
            Identifier of the particle
        at_turn : array_like of int, optional
            Number of tracked turns
        state : array_like of int, optional
            It is <= 0 if the particle is lost, > 0 otherwise
            (different values are used to record information on how the particle
            is lost or generated)
        pdg_id : array_like of float, optional
            PDG id of the particle under consideration (needed when tracking
            ions to distinguish different particle types). The default is 0
            (undefined)
        weight : array_like of float, optional
            Particle weight in number of particles (for collective simulations)
        at_element : array_like of int, optional
            Identifier of the last element through which the particle has been
        parent_particle_id : array_like of int, optional
            Identifier of the parent particle (secondary production processes)
        t_sim : float, optional
            Simulation frame time (typically one revolution period)
        """
        if '_xobject' in kwargs.keys():
            # Initialize xobject
            self.xoinitialize(**kwargs)
            return

        if 'sigma' in kwargs.keys():
            raise NameError('`sigma` is not supported anymore. '
                            'Please use `zeta` instead.')

        if 'psigma' in kwargs.keys():
            raise NameError('`psigma` is not supported anymore.'
                            'Please use `pzeta` instead.')

        accepted_args = set(self._xofields.keys()) | {
            'energy0', 'tau', 'pzeta', 'mass_ratio', 'mass', 'kinetic_energy0',
            '_context', '_buffer', '_offset', 'p0',
        }
        if set(kwargs.keys()) - accepted_args:
            raise NameError(f'Invalid argument(s) provided: '
                            f'{set(kwargs.keys()) - accepted_args}')

        per_part_input_vars = (
            self.per_particle_vars +
            ((xo.Float64, 'energy0'),
             (xo.Float64, 'tau'),
             (xo.Float64, 'pzeta'),
             (xo.Float64, 'mass_ratio'))
        )

        # Determine the number of particles and the capacity, so we can allocate
        # the xobject of the right size
        input_length = 1
        for _, field in per_part_input_vars:
            if field not in kwargs.keys():
                continue
            if np.isscalar(kwargs[field]) or len(kwargs[field]) == 1:
                continue
            if len(kwargs[field]) != input_length and input_length > 1:
                raise ValueError(
                    'All per particle vars have to be of the '
                    'same length.'
                )
            input_length = len(kwargs[field])

        # Validate _capacity if given explicitly, if not assume it based on input
        if _capacity is not None:
            if _capacity <= 0:
                raise ValueError('Explicitly provided `_capacity` has to be'
                                 'greater than zero.')

            if _capacity < input_length:
                raise ValueError(
                    f'Capacity ({_capacity}) has to be greater or equal to the '
                    f'number of particles ({input_length}).'
                )
        else:
            _capacity = input_length

        # Allocate the xobject of the right size
        self.xoinitialize(
            _context=kwargs.pop('_context', None),
            _buffer=kwargs.pop('_buffer', None),
            _offset=kwargs.pop('_offset', None),
            **{field: _capacity for _, field in self.per_particle_vars}
        )
        self._capacity = _capacity
        self._num_active_particles = -1  # To be filled in only on CPU
        self._num_lost_particles = -1  # To be filled in only on CPU

        # Initialize the fields to preset values
        for type_, field in self.per_particle_vars:
            raw_field = self._rename.get(field, field)
            if raw_field.startswith('_rng'):
                setattr(self, raw_field, 0)
            else:
                setattr(self, raw_field, LAST_INVALID_STATE)

        np_to_ctx = self._context.nparray_to_context_array

        # Mask out the unallocated space from now on
        # (match the length of the input arrays)
        self.hide_first_n_particles(input_length)

        # Start populating the object with the input values
        state = kwargs.get('state', 1)
        if np.isscalar(state) or len(state) == 1:
            state = np.array(state).item()
        else:
            state = np_to_ctx(np.array(state))
        self.state = state
        input_mask = self.state > LAST_INVALID_STATE

        particle_ids = kwargs.get('particle_id', np.arange(input_length))
        particle_ids = np.atleast_1d(particle_ids)
        self.particle_id = np_to_ctx(particle_ids)

        parent_particle_id = np.atleast_1d(kwargs.get('parent_particle_id',
                                                      particle_ids))
        self.parent_particle_id = np_to_ctx(parent_particle_id)

        for field in ('state', 'particle_id', 'parent_particle_id'):
            kwargs.pop(field, None)

        # Init scalar vars
        self.q0 = kwargs.get('q0', 1.0)
        self.mass0 = kwargs.get('mass0', PROTON_MASS_EV)
        self.t_sim = kwargs.get('t_sim', 0)
        self.start_tracking_at_element = kwargs.get(
                            'start_tracking_at_element', -1)

        # Init refs
        if 'kinetic_energy0' in kwargs.keys():
            assert kwargs.get('energy0') is None
            kwargs['energy0'] = kwargs.pop('kinetic_energy0') + self.mass0

        # Ensure that all per particle inputs are numpy arrays of the same
        # length, and move them to the target context
        for xotype, field in per_part_input_vars:
            if field not in kwargs.keys():
                continue

            if np.isscalar(kwargs[field]) or len(kwargs[field]) == 1:
                value = np.array(kwargs[field]).item()
                kwargs[field] = np.full(input_length, value)
            else:
                kwargs[field] = np.array(kwargs[field])

            # Coerce the right type so that we can allocate the right array
            # in the target context. PyOpenCL gets fussy if types don't match
            # in calculations.
            if kwargs[field].dtype != xotype._dtype:
                kwargs[field] = kwargs[field].astype(xotype._dtype)
            kwargs[field] = np_to_ctx(kwargs[field])



        # Init independent per particle vars
        self.init_independent_per_part_vars(kwargs)


        self._update_refs(
            p0c=kwargs.get('p0c'),
            energy0=kwargs.get('energy0'),
            gamma0=kwargs.get('gamma0'),
            beta0=kwargs.get('beta0'),
            mask=input_mask,
        )

        # Init chi and charge ratio
        self._update_chi_charge_ratio(
            chi=kwargs.get('chi'),
            charge_ratio=kwargs.get('charge_ratio'),
            mass_ratio=kwargs.get('mass_ratio'),
            mask=input_mask,
        )

        # Init energy deviations
        self._update_energy_deviations(
            delta=kwargs.get('delta'),
            ptau=kwargs.get('ptau'),
            pzeta=kwargs.get('pzeta'),
            _rpp=kwargs.get('rpp'),
            _rvv=kwargs.get('rvv'),
            mask=input_mask,
        )

        # Init zeta
        self._update_zeta(
            zeta=kwargs.get('zeta'),
            tau=kwargs.get('tau'),
            mask=input_mask,
        )

        self.unhide_first_n_particles()
        if isinstance(self._context, xo.ContextCpu) and not _no_reorganize:
            self.reorganize()


    @classmethod
    def from_dict(cls, dct, load_rng_state=True, **kwargs):

        """
        Create a new Particles object from a dictionary.

        Parameters
        ----------
        dct : dict
            The dictionary to load the Particles object from.
        load_rng_state : bool, optional
            Whether to load the state of the random number generator  from the
            dictionary. Defaults to True.
        _context : Context, optional
            The context to load the Particles object into. If not provided,
            the xobjects default context will be used.
        _buffer : Buffer, optional
            The buffer to load the Particles object into. If not provided,
            a new buffer will be allocated from the context.

        Returns
        -------
        particles : Particles
            The newly created Particles object.
        """

        part = cls(**dct, **kwargs)
        np_to_ctx = part._context.nparray_to_context_array

        def array_to_ctx(ary, default=0):
            if ary is not None and not np.isscalar(ary):
                return np_to_ctx(np.array(ary, dtype='uint32'))
            else:
                return ary or default

        if load_rng_state:
            part._rng_s1 = array_to_ctx(dct.get('_rng_s1'))
            part._rng_s2 = array_to_ctx(dct.get('_rng_s2'))
            part._rng_s3 = array_to_ctx(dct.get('_rng_s3'))
            part._rng_s4 = array_to_ctx(dct.get('_rng_s4'))

        return part

    def to_dict(self, copy_to_cpu=True,
                remove_underscored=None,
                remove_unused_space=None,
                remove_redundant_variables=None,
                keep_rng_state=None,
                compact=False):

        """
        Convert the Particles object to a dictionary.

        Parameters
        ----------
        copy_to_cpu : bool, optional
            Whether to copy the Particles object to the CPU before converting
            it to a dictionary. Defaults to True.
        compact:
            Whether to minimize the size of the dictionary. Defaults to False.
        remove_underscored : bool, optional
            Whether to remove underscored variables from the dictionary.
            Defaults to True.
        remove_unused_space : bool, optional
            Whether to remove unused space from the arrays. Defaults to
            the value of `compact`.
        remove_redundant_variables : bool, optional
            Whether to remove redundant variables from the dictionary.
            Defaults to the value of `compact`.
        keep_rng_state : bool, optional
            Whether to keep the state of the random number generator in the
            dictionary. Defaults to true if `compact` is False.

        Returns
        -------
        dct : dict
            The dictionary containing the data from Particles object.
        """

        if remove_underscored is None:
            remove_underscored = True

        if remove_unused_space is None:
            remove_unused_space = compact

        if remove_redundant_variables is None:
            remove_redundant_variables = compact

        if keep_rng_state is None:
            keep_rng_state = not compact

        p_for_dict = self

        if copy_to_cpu:
            p_for_dict = p_for_dict.copy(_context=xo.context_default)

        if remove_unused_space:
            p_for_dict = p_for_dict.remove_unused_space()

        dct = xo.HybridClass.to_dict(p_for_dict)
        dct['delta'] = p_for_dict.delta
        dct['ptau'] = p_for_dict.ptau
        dct['rvv'] = p_for_dict.rvv
        dct['rpp'] = p_for_dict.rpp
        dct['p0c'] = p_for_dict._p0c
        dct['beta0'] = p_for_dict._beta0
        dct['gamma0'] = p_for_dict._gamma0
        dct['start_tracking_at_element'] = p_for_dict.start_tracking_at_element

        if remove_underscored:
            for kk in list(dct.keys()):
                if kk.startswith('_'):
                    if keep_rng_state and kk.startswith('_rng'):
                        continue
                    del (dct[kk])

        if remove_redundant_variables:
            for kk in ['ptau', 'rpp', 'rvv', 'gamma0', 'beta0']:
                del (dct[kk])

        return dct

    def to_json(self, filename, indent=None, **kwargs):
        """
        Save the Particles object to a JSON file.

        Parameters
        ----------
        filename : str
            The name of the file to save the Particles object to.
        **kwargs : dict
            Additional keyword arguments to pass to the json.to_dict method.
        """

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                return json.JSONEncoder.default(self, obj)

        dct = self.to_dict(**kwargs)

        with open(filename, 'w') as f:
            json.dump(dct, f, cls=NumpyEncoder, indent=indent)

    @classmethod
    def from_pandas(cls, df, _context=None, _buffer=None, _offset=None, load_rng_state=True, **kwargs):

        """
        Create a new Particles object from a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to load the Particles object from.
        _context : Context, optional
            The context to load the Particles object into. If not provided,
            the xobjects default context will be used.
        _buffer : Buffer, optional
            The buffer to load the Particles object into. If not provided,
            a new buffer will be allocated from the context.

        Returns
        -------
        particles : Particles
            The newly created Particles object.
        """

        dct = df.to_dict(orient='list')
        for tt, nn in cls.scalar_vars + cls.size_vars:
            if nn in dct.keys() and not np.isscalar(dct[nn]):
                dct[nn] = dct[nn][0]
        return cls.from_dict(dct, load_rng_state=load_rng_state,
                             _context=_context, _buffer=_buffer,
                             _offset=_offset, **kwargs)

    def to_pandas(self,
                  remove_underscored=None,
                  remove_unused_space=None,
                  remove_redundant_variables=None,
                  keep_rng_state=None,
                  compact=False):

        """
        Convert the Particles object to a pandas DataFrame.

        Parameters
        ----------
        compact:
            Whether to minimize the size of the dictionary. Defaults to False.
        remove_underscored : bool, optional
            Whether to remove underscored variables from the dictionary.
            Defaults to True.
        remove_unused_space : bool, optional
            Whether to remove unused space from the arrays. Defaults to
            the value of `compact`.
        remove_redundant_variables : bool, optional
            Whether to remove redundant variables from the dictionary.
            Defaults to the value of `compact`.
        keep_rng_state : bool, optional
            Whether to keep the state of the random number generator in the
            dictionary. Defaults to true if `compact` is False.

        Returns
        -------
        df : pandas.DataFrame
            The DataFrame containing the data from Particles object.

        """

        dct = self.to_dict(
            remove_underscored=remove_underscored,
            remove_unused_space=remove_unused_space,
            remove_redundant_variables=remove_redundant_variables,
            keep_rng_state=keep_rng_state,
            compact=compact)
        import pandas as pd
        return pd.DataFrame(dct)

    def to_table(self):

        """
        Get a Table object with the Particles coordinates.

        Returns
        -------
        table : Table
            The Table object containing the data from Particles object.

        """

        import xtrack as xt
        out_dct = self.to_dict(compact=False)

        for kk in list(out_dct.keys()):
            if not hasattr(out_dct[kk], '__len__'):
                out_dct.pop(kk)
            elif len(out_dct[kk]) != len(out_dct['particle_id']):
                out_dct.pop(kk)

        # Prettier ordering
        col_names = ['s', 'x', 'px', 'y', 'py', 'zeta', 'delta', 'particle_id']
        for nn in col_names.copy():
            if nn not in out_dct.keys():
                col_names.remove(nn)
        col_names += [kk for kk in out_dct.keys() if kk not in col_names]

        return xt.Table(out_dct, index='particle_id', col_names=col_names)

    def get_table(self):

        """
        Get a Table object with the Particles coordinates.

        Returns
        -------
        table : Table
            The Table object containing the data from Particles object.

        """

        return self.to_table()


    @classmethod
    def merge(cls, lst, _context=None, _buffer=None, _offset=None):

        """
        Merge a list of Particles objects into a single one.

        Parameters
        ----------
        lst : list of Particles
            The list of Particles objects to merge.
        _context : Context, optional
            The context to load the Particles object into. If not provided,
            the xobjects default context will be used.
        _buffer : Buffer, optional
            The buffer to load the Particles object into. If not provided,
            a new buffer will be allocated from the context.

        Returns
        -------
        particles : Particles
            The newly created Particles object.

        """

        # TODO For now the merge is performed on CPU for add contexts.
        # Slow for objects on GPU (transferred to CPU for the merge).

        # Move everything to cpu
        cpu_lst = []
        for pp in lst:
            assert isinstance(pp, cls)
            if isinstance(pp._buffer.context, xo.ContextCpu):
                cpu_lst.append(pp)
            else:
                cpu_lst.append(pp.copy(_context=xo.context_default))

        # Check that scalar variable are compatible
        for tt, nn in cls.scalar_vars:
            vals = [getattr(pp, nn) for pp in cpu_lst]
            assert np.allclose(vals, getattr(cpu_lst[0], nn),
                               rtol=0, atol=1e-14)

        # Make new particle on CPU
        capacity = np.sum([pp._capacity for pp in cpu_lst])
        new_part_cpu = cls(_capacity=capacity)

        # Copy scalar vars from first particle
        for tt, nn in cls.scalar_vars:
            setattr(new_part_cpu, nn, getattr(cpu_lst[0], nn))

        # Copy per-particle vars
        first = 0
        max_id_curr = -1
        with new_part_cpu._bypass_linked_vars():
            for pp in cpu_lst:
                for tt, nn in cls.per_particle_vars:
                    if not (nn == 'particle_id' or nn == 'parent_id'):
                        getattr(new_part_cpu, nn)[
                        first:first + pp._capacity] = getattr(pp, nn)

                # Handle particle_ids and parent_ids
                mask = pp.particle_id >= 0
                new_id = pp.particle_id.copy()
                new_parent_id = pp.parent_particle_id.copy()
                if np.min(new_id[mask]) <= max_id_curr:
                    new_id[mask] += (max_id_curr + 1)
                    new_parent_id[mask] += (max_id_curr + 1)
                new_part_cpu.particle_id[first:first + len(new_id)] = new_id
                new_part_cpu.parent_particle_id[
                first:first + len(new_id)] = new_parent_id

                max_id_curr = np.max(new_id)
                first += pp._capacity

        # Reorganize
        new_part_cpu.reorganize()

        # Copy to appropriate context
        if _context is None and _buffer is None:
            # Use constext of first particle
            if isinstance(lst[0]._buffer.context, xo.ContextCpu):
                new_part_cpu._buffer.context = lst[0]._buffer.context
                return new_part_cpu
            else:
                return new_part_cpu.copy(_context=lst[0]._buffer.context)
        else:
            return new_part_cpu.copy(_context=_context, _buffer=_buffer,
                                     _offset=_offset)

    def filter(self, mask):

        """
        Select a subset of particles satisfying a logical condition.

        Parameters
        ----------
        mask : array of bool
            The logical condition to apply to the particles.

        Returns
        -------
        particles : Particles
            The newly created Particles object.
        """

        if isinstance(self._buffer.context, xo.ContextCpu):
            self_cpu = self
        else:
            self_cpu = self.copy(_context=xo.context_default)

        # copy mask to cpu is needed
        if isinstance(mask, self._buffer.context.nplike_array_type):
            mask = self._buffer.context.nparray_from_context_array(mask)

            # Pyopencl returns int8 instead of bool
            if (isinstance(self._buffer.context, xo.ContextPyopencl) and
                    mask.dtype == np.int8):
                assert np.all((mask >= 0) & (mask <= 1))
                mask = mask > 0

        # Make new particle on CPU
        test_x = self_cpu.x[mask]
        capacity = len(test_x)
        new_part_cpu = self.__class__(_capacity=capacity)

        new_part_cpu.start_tracking_at_element = self.start_tracking_at_element

        # Copy scalar vars from first particle
        for tt, nn in self.scalar_vars:
            setattr(new_part_cpu, nn, getattr(self_cpu, nn))

        # Copy per-particle vars
        for tt, nn in self.per_particle_vars:
            with new_part_cpu._bypass_linked_vars():
                getattr(new_part_cpu, nn)[:] = getattr(self_cpu, nn)[mask]

        # Reorganize
        new_part_cpu.reorganize()

        # Copy to original context
        target_ctx = self._buffer.context
        if isinstance(target_ctx, xo.ContextCpu):
            new_part_cpu.move(_context=target_ctx)
            return new_part_cpu
        else:
            return new_part_cpu.copy(_context=target_ctx)

    def sort(self, by='particle_id', interleave_lost_particles=False):

        """
        Sort the particles by a given variable.

        Parameters
        ----------
        by : str
            The name of the variable to sort by. Default is 'particle_id'.
        interleave_lost_particles : bool
            If True, lost particles are interleaved with active particles.
            If False, lost particles are moved to the end of the array.

        Returns
        -------
        sorted_index : array of int
            The index of the sorted particles.

        """

        if not isinstance(self._buffer.context, xo.ContextCpu):
            raise NotImplementedError('Sorting only works on CPU for now')

        if self.lost_particles_are_hidden:
            restore_hidden = True
            self.unhide_lost_particles()
        else:
            restore_hidden = False

        n_active, n_lost = self.reorganize()

        n_used = n_active + n_lost
        sort_key_var = getattr(self, by)[:n_used].copy()
        if not interleave_lost_particles:
            max_id_active = np.max(self.particle_id[:n_active])
            sort_key_var[n_active:] = 10 + max_id_active + sort_key_var[n_active:]

        sorted_index = np.argsort(sort_key_var)

        with self._bypass_linked_vars():
            for tt, nn in self.per_particle_vars:
                vv = getattr(self, nn)
                vv[:n_used] = vv[:n_used][sorted_index]

        if interleave_lost_particles:
            self._num_active_particles = -2
            self._num_lost_particles = -2
        elif restore_hidden:
            self.hide_lost_particles(_assume_reorganized=True)

    def reorganize(self):

        """
        Reorganize the particles object so that all active particles are at the
        beginning of the arrays.

        Returns
        -------
        n_active : int
            The number of active particles.
        n_lost : int
            The number of lost particles.
        """

        if self.lost_particles_are_hidden:
            restore_hidden = True
            self.unhide_lost_particles()
        else:
            restore_hidden = False

        if isinstance(self._context, xo.ContextPyopencl):
            # Needs special treatment because masking does not work with pyopencl
            # Going to for the masking for now, could be replaced by a kernel in the future.
            state_cpu = self.state.get()
            mask_active_cpu = state_cpu > 0
            mask_lost_cpu = (state_cpu < 1) & (state_cpu > LAST_INVALID_STATE)
            mask_active = self._context.nparray_to_context_array(
                np.where(mask_active_cpu)[0])
            mask_lost = self._context.nparray_to_context_array(
                np.where(mask_lost_cpu)[0])
            n_active = int(np.sum(mask_active_cpu))
            n_lost = int(np.sum(mask_lost_cpu))
            needs_reorganization = not mask_active_cpu[:n_active].all()
        else:
            mask_active = self.state > 0
            mask_lost = (self.state < 1) & (self.state > LAST_INVALID_STATE)
            n_active = int(np.sum(mask_active))
            n_lost = int(np.sum(mask_lost))
            needs_reorganization = not mask_active[:n_active].all()

        if needs_reorganization:
            # Reorganize particles
            with self._bypass_linked_vars():
                for tt, nn in self.per_particle_vars:
                    if nn.startswith('_rng'):
                        continue
                    vv = getattr(self, nn)
                    vv_active = vv[mask_active]
                    vv_lost = vv[mask_lost]

                    vv[:n_active] = vv_active
                    vv[n_active:n_active + n_lost] = vv_lost
                    vv[n_active + n_lost:] = tt._dtype.type(LAST_INVALID_STATE)

        if isinstance(self._buffer.context, xo.ContextCpu):
            self._num_active_particles = n_active
            self._num_lost_particles = n_lost

        if restore_hidden:
            self.hide_lost_particles(_assume_reorganized=True)

        return n_active, n_lost


    def hide_lost_particles(self, _assume_reorganized=False):
        """
        Hide lost particles in the particles object.
        """
        self._lim_arrays_name = '_num_active_particles'
        if not _assume_reorganized:
            n_active, _ = self.reorganize()
            self._num_active_particles = n_active

    def unhide_lost_particles(self):
        """
        Unhide lost particles in the particles object.
        """
        if hasattr(self, '_lim_arrays_name'):
            del self._lim_arrays_name
        if not isinstance(self._context, xo.ContextCpu):
            self._num_active_particles = -1

    def remove_unused_space(self):

        """
        Return a new particles object with removed unused space in the
        particle arrays (when the number of particles is smaller than the
        capacity of the particles object).
        """

        return self.filter(self.state > LAST_INVALID_STATE)


    def add_particles(self, part, keep_lost=False):
        """
        Add particles to the particles object.

        Parameters
        ----------
        part : Particles
            The particles to add.
        keep_lost : bool, optional
            If True, lost particles are also added. Default is False.
        """

        if keep_lost:
            raise NotImplementedError
        assert not isinstance(self._buffer.context, xo.ContextPyopencl), (
            'Masking does not work with pyopencl')

        mask_copy = part.state > 0
        n_copy = np.sum(mask_copy)

        n_active, n_lost = self.reorganize()
        i_start_copy = n_active + n_lost
        n_free = self._capacity - n_active - n_lost

        max_id = np.max(self.particle_id[:n_active + n_lost])

        if n_copy > n_free:
            raise NotImplementedError("Out of space, need to regenerate xobject")

        for tt, nn in self.scalar_vars:
            if nn == 't_sim':
                continue
            assert np.isclose(getattr(self, nn), getattr(part, nn),
                              rtol=1e-14, atol=1e-14)

        with self._bypass_linked_vars():
            for tt, nn in self.per_particle_vars:
                if nn.startswith('_rng'):
                    continue
                vv = getattr(self, nn)
                vv_copy = getattr(part, nn)[mask_copy]
                vv[i_start_copy:i_start_copy + n_copy] = vv_copy

        new_ids = self._context.nplike_lib.arange(
            int(max_id) + 1, int(max_id) + 1 + int(n_copy), dtype=np.int64)
        self.particle_id[i_start_copy:i_start_copy + n_copy] = new_ids

        self.reorganize()

    def get_active_particle_id_range(self):
        """
        Get the range of particle ids of active particles.
        """
        ctx2np = self._buffer.context.nparray_from_context_array
        mask_active = ctx2np(self.state) > 0
        ids_active_particles = ctx2np(self.particle_id)[mask_active]
        # Behaves as python range (+1)
        return np.min(ids_active_particles), np.max(ids_active_particles) + 1

    def init_pipeline(self, name):

        self.name = name
        self._needs_pipeline = True

    def show(self):

        """
        Print particle properties.
        """

        df = self.to_pandas()
        dash = '-' * 55
        print("PARTICLES:\n\n")
        print('{:<27} {:>12}'.format("Property", "Value"))
        print(dash)
        for column in df:
            print('{:<27} {:>12}'.format(df[column].name, df[column].values[0]))
        print(dash)
        print('\n')

    def get_classical_particle_radius0(self):

        """
        Get classical particle radius of the reference particle.
        """

        m0 = self.mass0 * qe / (clight ** 2)  # electron volt - kg conversion
        r0 = (self.q0 * qe) ** 2 / (4 * np.pi * epsilon_0 * m0 * clight ** 2)  # 1.5347e-18 is default for protons
        return r0

    def _bypass_linked_vars(self):
        return BypassLinked(self)

    def _has_valid_rng_state(self):
        # I check only the first particle
        if (self._xobject._rng_s1[0] == 0
                and self._xobject._rng_s2[0] == 0
                and self._xobject._rng_s3[0] == 0
                and self._xobject._rng_s4[0] == 0):
            return False
        else:
            return True

    def _init_random_number_generator(self, seeds=None):
        """
        Initialize state of the random number generator (possibility to providing
        a seed for each particle).
        """
        context = self._buffer.context
        if context.allow_prebuilt_kernels:
            _print_state = Print.suppress
            Print.suppress = True
            try:
                from xsuite import (
                    get_suitable_kernel,
                    XSK_PREBUILT_KERNELS_LOCATION,
                )
                kernel_info = get_suitable_kernel({}, ())
            except ImportError:
                kernel_info = None

            Print.suppress = _print_state
            if kernel_info:
                module_name, _ = kernel_info
                kernels = context.kernels_from_file(
                    module_name=module_name,
                    containing_dir=XSK_PREBUILT_KERNELS_LOCATION,
                    kernel_descriptions=self._kernels,
                )
                context.kernels.update(kernels)
        self.compile_kernels(only_if_needed=True)

        if seeds is None:
            seeds = np.random.randint(low=1, high=4e9,
                                      size=self._capacity, dtype=np.uint32)
        else:
            assert len(seeds) == self._capacity
            if not hasattr(seeds, 'dtype') or seeds.dtype != np.uint32:
                seeds = np.array(seeds, dtype=np.uint32)

        seeds_dev = context.nparray_to_context_array(seeds)
        kernel = context.kernels['Particles_initialize_rand_gen']
        kernel(particles=self, seeds=seeds_dev, n_init=self._capacity)

    def hide_first_n_particles(self, num_particles):
        """
        Hide first `num_particles` particles in the particles object.
        """
        self._lim_arrays_name = '_num_shown_particles'
        self._num_shown_particles = num_particles

    def unhide_first_n_particles(self):
        """
        Unhide the particles in the particles object.
        """
        if hasattr(self, '_num_shown_particles'):
            del self._lim_arrays_name
        if not isinstance(self._context, xo.ContextCpu):
            self._num_shown_particles = -1

    @property
    def lost_particles_are_hidden(self):
        return (hasattr(self, '_lim_arrays_name') and
                self._lim_arrays_name == '_num_active_particles')

    def _contains_lost_or_unallocated_particles(self):
        ctx = self._buffer.context
        # TODO: check and handle behavior with hidden lost particles
        if isinstance(ctx, xo.ContextPyopencl):
            npstate = ctx.nparray_from_context_array(self.state)
            return np.any(npstate <= 0)
        else:
            return ctx.nplike_lib.any(self.state <= 0)

    def update_delta(self, new_delta_value):

        """
        Update the `delta` value of the particles object. `ptau` and `rvv` and
        `rpp` are updated accordingly. If `new_delta_value` contains nans, these
        values are not updated.
        """

        isnan = self._context.nplike_lib.isnan
        # The comparison with False is needed as mask consists of int8 on opencl
        mask = (isnan(new_delta_value) == False) & (self.state > 0)  # noqa

        self._update_energy_deviations(
            delta=new_delta_value,
            mask=mask
        )

    @property
    def delta(self):
        return self._buffer.context.linked_array_type.from_array(
            self._delta,
            mode='setitem_from_container',
            container=self,
            container_setitem_name='_delta_setitem')

    @delta.setter
    def delta(self, value):
        self.delta[:] = value

    def _delta_setitem(self, indx, val):
        ctx = self._buffer.context
        temp_delta = ctx.zeros(shape=self._delta.shape, dtype=np.float64)
        temp_delta[:] = np.nan
        temp_delta[indx] = val
        self.update_delta(temp_delta)

    def update_ptau(self, new_ptau):

        """
        Update the `ptau` value of the particles object. `delta` and `rvv` and
        `rpp` are updated accordingly. If `new_ptau` contains nans, these values
        are not updated.
        """

        isnan = self._context.nplike_lib.isnan
        # The comparison with False is needed as mask consists of int8 on opencl
        mask = (isnan(new_ptau) == False) & (self.state > 0)  # noqa

        self._update_energy_deviations(
            ptau=new_ptau,
            mask=mask
        )

    def _ptau_setitem(self, indx, val):
        ctx = self._buffer.context
        temp_ptau = ctx.zeros(shape=self._ptau.shape, dtype=np.float64)
        temp_ptau[:] = np.nan
        temp_ptau[indx] = val
        self.update_ptau(temp_ptau)

    @property
    def ptau(self):
        return self._buffer.context.linked_array_type.from_array(
            self._ptau,
            mode='setitem_from_container',
            container=self,
            container_setitem_name='_ptau_setitem')

    @ptau.setter
    def ptau(self, value):
        self.ptau[:] = value

    def update_p0c(self, new_p0c):

        """
        Update the `p0c` value of the particles object. `gamma0` and `beta0` are
        updated accordingly. If `new_p0c` contains nans, these values
        are not updated.
        """

        isnan = self._context.nplike_lib.isnan
        # The comparison with False is needed as mask consists of int8 on opencl
        mask = (isnan(new_p0c) == False) & (self.state > 0)  # noqa

        self._update_refs(
            p0c=new_p0c,
            mask=mask,
        )

    def _p0c_setitem(self, indx, val):
        ctx = self._buffer.context
        temp_p0c = ctx.zeros(shape=self._p0c.shape, dtype=np.float64)
        temp_p0c[:] = np.nan
        temp_p0c[indx] = val
        self.update_p0c(temp_p0c)

    @property
    def p0c(self):
        return self._buffer.context.linked_array_type.from_array(
            self._p0c,
            mode='setitem_from_container',
            container=self,
            container_setitem_name='_p0c_setitem')

    @p0c.setter
    def p0c(self, value):
        self.p0c[:] = value

    def update_gamma0(self, new_gamma0):

        """
        Update the `gamma0` value of the particles object. `p0c` and `beta0` are
        updated accordingly. If `new_gamma0` contains nans, these values
        are not updated.
        """

        mask = (new_gamma0 == new_gamma0) & (self.state > 0)

        self._update_refs(
            gamma0=new_gamma0,
            mask=mask,
        )

    def _gamma0_setitem(self, indx, val):
        ctx = self._buffer.context
        temp_gamma0 = ctx.zeros(shape=self._gamma0.shape, dtype=np.float64)
        temp_gamma0[:] = np.nan
        temp_gamma0[indx] = val
        self.update_gamma0(temp_gamma0)

    @property
    def gamma0(self):
        return self._buffer.context.linked_array_type.from_array(
            self._gamma0,
            mode='setitem_from_container',
            container=self,
            container_setitem_name='_gamma0_setitem')

    @gamma0.setter
    def gamma0(self, value):
        self.gamma0[:] = value

    def update_beta0(self, new_beta0):

        """
        Update the `beta0` value of the particles object. `p0c` and `gamma0` are
        updated accordingly. If `new_beta0` contains nans, these values
        are not updated.
        """

        mask = (new_beta0 == new_beta0) & (self.state > 0)

        self._update_refs(
            beta0=new_beta0,
            mask=mask,
        )

    def _beta0_setitem(self, indx, val):
        ctx = self._buffer.context
        temp_beta0 = ctx.zeros(shape=self._beta0.shape, dtype=np.float64)
        temp_beta0[:] = np.nan
        temp_beta0[indx] = val
        self.update_beta0(temp_beta0)

    @property
    def beta0(self):
        return self._buffer.context.linked_array_type.from_array(
            self._beta0,
            mode='setitem_from_container',
            container=self,
            container_setitem_name='_beta0_setitem')

    @beta0.setter
    def beta0(self, value):
        self.beta0[:] = value

    @property
    def rvv(self):
        return self._buffer.context.linked_array_type.from_array(
            self._rvv, mode='readonly',
            container=self)

    @property
    def rpp(self):
        return self._buffer.context.linked_array_type.from_array(
            self._rpp, mode='readonly',
            container=self)

    @property
    def energy0(self):
        energy0 = (self.p0c * self.p0c + self.mass0 * self.mass0) ** 0.5
        return self._buffer.context.linked_array_type.from_array(
            energy0, mode='readonly',
            container=self)

    @property
    def kinetic_energy0(self):
        kene0 = self.energy0 - self.mass0
        return self._buffer.context.linked_array_type.from_array(
            kene0, mode='readonly',
            container=self)

    @property
    def energy(self):
        energy = (self.energy0 + self.ptau * self.p0c) * self.mass_ratio  # eV
        return self._buffer.context.linked_array_type.from_array(
            energy, mode='readonly',
            container=self)

    @property
    def mass_ratio(self):
        out = self.charge_ratio / self.chi
        return self._buffer.context.linked_array_type.from_array(
            out, mode='readonly',
            container=self)

    @property
    def mass(self):
        out = self.mass_ratio * self.mass0
        return self._buffer.context.linked_array_type.from_array(
            out, mode='readonly',
            container=self)

    @property
    def charge(self):
        out = self.charge_ratio * self.q0
        return self._buffer.context.linked_array_type.from_array(
            out, mode='readonly',
            container=self)

    @property
    def pzeta(self):
        pzeta = self.ptau / self.beta0
        return self._buffer.context.linked_array_type.from_array(
            pzeta, mode='readonly',
            container=self)

    @property
    def kin_px(self):
        out = self.px - self.ax
        return self._buffer.context.linked_array_type.from_array(
            out,
            mode='readonly',
            container=self)

    @property
    def kin_py(self):
        out = self.py - self.ay
        return self._buffer.context.linked_array_type.from_array(
            out,
            mode='readonly',
            container=self)

    @property
    def kin_ps(self):
        sqrt = self._context.nplike_lib.sqrt
        out = sqrt((1 + self.delta) ** 2 - self.kin_px ** 2 - self.kin_py ** 2)
        return self._buffer.context.linked_array_type.from_array(
            out,
            mode='readonly',
            container=self)

    @property
    def kin_xprime(self):
        out = self.kin_px / self.kin_ps
        return self._buffer.context.linked_array_type.from_array(
            out,
            mode='readonly',
            container=self)

    @property
    def kin_yprime(self):
        out = self.kin_py / self.kin_ps
        return self._buffer.context.linked_array_type.from_array(
            out,
            mode='readonly',
            container=self)

    def add_to_energy(self, delta_energy):
        """
        Add `delta_energy` to the `energy` of the particles object. `delta`,
        'ptau', `rvv` and `rpp` are updated accordingly.
        """
        self.ptau += delta_energy / self.p0c / self.mass_ratio

    def set_particle(self, index, set_scalar_vars=False, **kwargs):
        raise NotImplementedError('This functionality has been removed')

    @classmethod
    def gen_local_particle_api(cls, mode='no_local_copy'):
        if mode != 'no_local_copy':
            raise NotImplementedError

        src_lines = ['']
        for name, mass in mass__dict__.items():
            if name.endswith('_MASS_EV'):
                src_lines.append(f'#define {name} {mass}')

        src_lines.append('typedef struct {')

        for tt, vv in cls.size_vars + cls.scalar_vars:
            src_lines.append('                 ' + tt._c_type + '  ' + vv + ';')

        for tt, vv in cls.per_particle_vars:
            src_lines.append('    /*gpuglmem*/ ' + tt._c_type + '* ' + vv + ';')

        src_lines.append('                 int64_t ipart;')
        src_lines.append('                 int64_t endpart;')
        src_lines.append('    /*gpuglmem*/ int8_t* io_buffer;')
        src_lines.append('} LocalParticle;')
        src_typedef = '\n'.join(src_lines)

        # Get io buffer
        src_lines = []
        src_lines.append('''
            /*gpufun*/
            /*gpuglmem*/ int8_t* LocalParticle_get_io_buffer(LocalParticle* part){
                return part->io_buffer;
            }

            ''')

        # Particles_to_LocalParticle
        src_lines.append('''
            /*gpufun*/
            void Particles_to_LocalParticle(ParticlesData source,
                                            LocalParticle* dest,
                                            int64_t id,
                                            int64_t eid){''')
        for _, vv in cls.size_vars + cls.scalar_vars:
            src_lines.append(
                f'  dest->{vv} = ParticlesData_get_' + vv + '(source);')

        for _, vv in cls.per_particle_vars:
            src_lines.append(
                f'  dest->{vv} = ParticlesData_getp1_' + vv + '(source, 0);')

        src_lines.append('  dest->ipart = id;')
        src_lines.append('  dest->endpart = eid;')
        src_lines.append('}')
        src_particles_to_local = '\n'.join(src_lines)

        # LocalParticle_to_Particles
        src_lines = []
        src_lines.append('''
            /*gpufun*/
            void LocalParticle_to_Particles(
                                            LocalParticle* source,
                                            ParticlesData dest,
                                            int64_t id,
                                            int64_t set_scalar){''')
        src_lines.append('if (set_scalar){')
        for _, vv in cls.size_vars + cls.scalar_vars:
            src_lines.append(
                f'  ParticlesData_set_' + vv + '(dest,'
                                               f'      LocalParticle_get_{vv}(source));')
        src_lines.append('}')

        for _, vv in cls.per_particle_vars:
            src_lines.append(
                f'  ParticlesData_set_' + vv + '(dest, id, '
                                               f'      LocalParticle_get_{vv}(source));')
        src_lines.append('}')
        src_local_to_particles = '\n'.join(src_lines)

        # Adders
        src_lines = []
        for tt, vv in cls.per_particle_vars:
            src_lines.append('''
            /*gpufun*/
            void LocalParticle_add_to_''' + vv + f'(LocalParticle* part, {tt._c_type} value)'
                             + '{')
            src_lines.append(f'#ifndef FREEZE_VAR_{vv}')
            src_lines.append(f'  part->{vv}[part->ipart] += value;')
            src_lines.append('#endif')
            src_lines.append('}\n')
        src_adders = '\n'.join(src_lines)

        # Scalers
        src_lines = []
        for tt, vv in cls.per_particle_vars:
            src_lines.append('''
            /*gpufun*/
            void LocalParticle_scale_''' + vv + f'(LocalParticle* part, {tt._c_type} value)'
                             + '{')
            src_lines.append(f'#ifndef FREEZE_VAR_{vv}')
            src_lines.append(f'  part->{vv}[part->ipart] *= value;')
            src_lines.append('#endif')
            src_lines.append('}\n')
        src_scalers = '\n'.join(src_lines)

        # Setters
        src_lines = []
        for tt, vv in cls.per_particle_vars:
            src_lines.append('''
            /*gpufun*/
            void LocalParticle_set_''' + vv + f'(LocalParticle* part, {tt._c_type} value)'
                             + '{')
            src_lines.append(f'#ifndef FREEZE_VAR_{vv}')
            src_lines.append(f'  part->{vv}[part->ipart] = value;')
            src_lines.append('#endif')
            src_lines.append('}')
        src_setters = '\n'.join(src_lines)

        # Getters
        src_lines = []

        for tt, vv in cls.size_vars + cls.scalar_vars:
            src_lines.append('/*gpufun*/')
            src_lines.append(f'{tt._c_type} LocalParticle_get_' + vv
                             + f'(LocalParticle* part)'
                             + '{')
            src_lines.append(f'  return part->{vv};')
            src_lines.append('}')

        for tt, vv in cls.per_particle_vars:
            src_lines.append('/*gpufun*/')
            src_lines.append(f'{tt._c_type} LocalParticle_get_' + vv
                             + f'(LocalParticle* part)'
                             + '{')
            src_lines.append(f'  return part->{vv}[part->ipart];')
            src_lines.append('}')

        src_getters = '\n'.join(src_lines)

        # Angles
        src_angles_lines = []
        for exact in ['', 'exact_']:
            # xp (py as transverse) and vice versa
            for xx, yy in [['x', 'y'], ['y', 'x']]:
                # Getter
                src_angles_lines.append('/*gpufun*/')
                src_angles_lines.append(f'double LocalParticle_get_{exact}{xx}p(LocalParticle* part){{')
                src_angles_lines.append(f'    double const p{xx} = LocalParticle_get_p{xx}(part);')
                if exact == 'exact_':
                    src_angles_lines.append(f'    double const p{yy} = LocalParticle_get_p{yy}(part);')
                    src_angles_lines.append(f'    double const one_plus_delta = 1. + LocalParticle_get_delta(part);')
                    src_angles_lines.append(
                        f'    double const rpp = 1./sqrt(one_plus_delta*one_plus_delta - px*px - py*py);')
                else:
                    src_angles_lines.append(f'    double const rpp = LocalParticle_get_rpp(part);')
                src_angles_lines.append(f'    // INFO: this is not the angle, but sin(angle)')
                src_angles_lines.append(f'    return p{xx}*rpp;')
                src_angles_lines.append('}')
                src_angles_lines.append('')

            for xx, yy in [['x', 'y'], ['y', 'x']]:
                # Setter
                src_angles_lines.append('/*gpufun*/')
                src_angles_lines.append(f'void LocalParticle_set_{exact}{xx}p(LocalParticle* part, double {xx}p){{')
                src_angles_lines.append(f'#ifndef FREEZE_VAR_p{xx}')
                src_angles_lines.append(f'    double rpp = LocalParticle_get_rpp(part);')
                if exact == 'exact_':
                    src_angles_lines.append(
                        f'    // Careful! If {yy}p also changes, use LocalParticle_set_{exact}xp_yp!')
                    src_angles_lines.append(f'    double const {yy}p = LocalParticle_get_{exact}{yy}p(part);')
                    src_angles_lines.append(f'    rpp *= sqrt(1 + xp*xp + yp*yp);')
                src_angles_lines.append(f'    // INFO: {xx}p is not the angle, but sin(angle)')
                src_angles_lines.append(f'    LocalParticle_set_p{xx}(part, {xx}p/rpp);')
                src_angles_lines.append(f'#endif')
                src_angles_lines.append('}')
                src_angles_lines.append('')

            for xx, yy in [['x', 'y'], ['y', 'x']]:
                # Adder
                src_angles_lines.append('/*gpufun*/')
                src_angles_lines.append(f'void LocalParticle_add_to_{exact}{xx}p(LocalParticle* part, double {xx}p){{')
                src_angles_lines.append(f'#ifndef FREEZE_VAR_p{xx}')
                src_angles_lines.append(f'    LocalParticle_set_{exact}{xx}p(part, '
                                        + f'LocalParticle_get_{exact}{xx}p(part) + {xx}p);')
                src_angles_lines.append(f'#endif')
                src_angles_lines.append('}')
                src_angles_lines.append('')
                # Scaler
                src_angles_lines.append('/*gpufun*/')
                src_angles_lines.append(f'void LocalParticle_scale_{exact}{xx}p(LocalParticle* part, double value){{')
                src_angles_lines.append(f'#ifndef FREEZE_VAR_p{xx}')
                src_angles_lines.append(f'    LocalParticle_set_{exact}{xx}p(part, '
                                        + f'LocalParticle_get_{exact}{xx}p(part) * value);')
                src_angles_lines.append(f'#endif')
                src_angles_lines.append('}')
                src_angles_lines.append('')
            # Double setter, adder, scaler
            src_angles_lines.append('/*gpufun*/')
            src_angles_lines.append(f'void LocalParticle_set_{exact}xp_yp(LocalParticle* part, double xp, double yp){{')
            src_angles_lines.append(f'    double rpp = LocalParticle_get_rpp(part);')
            if exact == 'exact_':
                src_angles_lines.append(f'    rpp *= sqrt(1 + xp*xp + yp*yp);')
            for xx in ['x', 'y']:
                src_angles_lines.append(f'#ifndef FREEZE_VAR_p{xx}')
                src_angles_lines.append(f'    LocalParticle_set_p{xx}(part, {xx}p/rpp);')
                src_angles_lines.append(f'#endif')
            src_angles_lines.append('}')
            src_angles_lines.append('')
            src_angles_lines.append('/*gpufun*/')
            src_angles_lines.append(
                f'void LocalParticle_add_to_{exact}xp_yp(LocalParticle* part, double xp, double yp){{')
            src_angles_lines.append(f'    LocalParticle_set_{exact}xp_yp(part, '
                                    + f'LocalParticle_get_{exact}xp(part) + xp, '
                                    + f'LocalParticle_get_{exact}yp(part) + yp);')
            src_angles_lines.append('}')
            src_angles_lines.append('')
            src_angles_lines.append('/*gpufun*/')
            src_angles_lines.append(
                f'void LocalParticle_scale_{exact}xp_yp(LocalParticle* part, double value_x, double value_y){{')
            src_angles_lines.append(f'    LocalParticle_set_{exact}xp_yp(part, '
                                    + f'LocalParticle_get_{exact}xp(part) * value_x, '
                                    + f'LocalParticle_get_{exact}yp(part) * value_y);')
            src_angles_lines.append('}')
        src_angles = '\n'.join(src_angles_lines)

        # Particle exchangers
        src_exchange = '''
        /*gpufun*/
        void LocalParticle_exchange(LocalParticle* part, int64_t i1, int64_t i2){
        '''
        for tt, vv in cls.per_particle_vars:
            src_exchange += '\n'.join([
                '\n    {',
                f'    {tt._c_type} temp = part->{vv}[i2];',
                f'    part->{vv}[i2] = part->{vv}[i1];',
                f'    part->{vv}[i1] = temp;',
                '     }'])
        src_exchange += '}\n'

        custom_source = '''
        /*gpufun*/
        double LocalParticle_get_energy0(LocalParticle* part){

            double const p0c = LocalParticle_get_p0c(part);
            double const m0  = LocalParticle_get_mass0(part);

            return sqrt( p0c * p0c + m0 * m0 );
        }

        /*gpufun*/
        void LocalParticle_update_ptau(LocalParticle* part, double new_ptau_value){

            double const beta0 = LocalParticle_get_beta0(part);

            double const ptau = new_ptau_value;

            double const irpp = sqrt(ptau*ptau + 2*ptau/beta0 +1);

            double const new_rpp = 1./irpp;
            LocalParticle_set_delta(part, irpp - 1.);

            double const new_rvv = irpp/(1 + beta0*ptau);
            LocalParticle_set_rvv(part, new_rvv);
            LocalParticle_set_ptau(part, ptau);

            LocalParticle_set_rpp(part, new_rpp );
        }

        /*gpufun*/
        void LocalParticle_update_delta(LocalParticle* part, double new_delta_value){
            double const beta0 = LocalParticle_get_beta0(part);
            double const delta_beta0 = new_delta_value * beta0;
            double const ptau_beta0  = sqrt( delta_beta0 * delta_beta0 +
                                        2. * delta_beta0 * beta0 + 1. ) - 1.;

            double const one_plus_delta = 1. + new_delta_value;
            double const rvv    = ( one_plus_delta ) / ( 1. + ptau_beta0 );
            double const rpp    = 1. / one_plus_delta;
            double const ptau = ptau_beta0 / beta0;

            LocalParticle_set_delta(part, new_delta_value);

            LocalParticle_set_rvv(part, rvv );
            LocalParticle_set_rpp(part, rpp );
            LocalParticle_set_ptau(part, ptau );

        }

        /*gpufun*/
        double LocalParticle_get_pzeta(LocalParticle* part){

            double const ptau = LocalParticle_get_ptau(part);
            double const beta0 = LocalParticle_get_beta0(part);

            return ptau/beta0;

        }

        /*gpufun*/
        void LocalParticle_update_pzeta(LocalParticle* part, double new_pzeta_value){

            double const beta0 = LocalParticle_get_beta0(part);
            LocalParticle_update_ptau(part, beta0*new_pzeta_value);

        }

        /*gpufun*/
        void increment_at_element(LocalParticle* part0, int64_t const increment){

            //start_per_particle_block (part0->part)
                LocalParticle_add_to_at_element(part, increment);
            //end_per_particle_block

        }

        /*gpufun*/
        void increment_at_turn(LocalParticle* part0, int flag_reset_s){

            //start_per_particle_block (part0->part)
            LocalParticle_add_to_at_turn(part, 1);
            LocalParticle_set_at_element(part, 0);
            if (flag_reset_s>0){
                LocalParticle_set_s(part, 0.);
            }
            //end_per_particle_block
        }

        /*gpufun*/
        void increment_at_turn_backtrack(LocalParticle* part0, int flag_reset_s,
                                         double const line_length,
                                         int64_t const num_elements){

            //start_per_particle_block (part0->part)
            LocalParticle_add_to_at_turn(part, -1);
            LocalParticle_set_at_element(part, num_elements);
            if (flag_reset_s>0){
                LocalParticle_set_s(part, line_length);
            }
            //end_per_particle_block
        }

        // check_is_active has different implementation on CPU and GPU

        #define CPU_SERIAL_IMPLEM //only_for_context cpu_serial
        #define CPU_OMP_IMPLEM //only_for_context cpu_openmp

        #ifdef CPU_SERIAL_IMPLEM

        /*gpufun*/
        int64_t check_is_active(LocalParticle* part) {
            int64_t ipart=0;
            while (ipart < part->_num_active_particles){
                #ifdef XSUITE_RESTORE_LOSS
                ipart++;
                #else
                if (part->state[ipart]<1){
                    LocalParticle_exchange(
                        part, ipart, part->_num_active_particles-1);
                    part->_num_active_particles--;
                    part->_num_lost_particles++;
                }
                else{
                    ipart++;
                }
                #endif
            }

            if (part->_num_active_particles==0){
                return 0;//All particles lost
            } else {
                return 1; //Some stable particles are still present
            }
        }

        #else // not CPU_SERIAL_IMPLEM
        #ifdef CPU_OMP_IMPLEM

        /*gpufun*/
        int64_t check_is_active(LocalParticle* part) {
        #ifndef SKIP_SWAPS
            int64_t ipart = part->ipart;
            int64_t endpart = part->endpart;

            int64_t left = ipart;
            int64_t right = endpart - 1;
            int64_t swap_made = 0;
            int64_t has_alive = 0;

            if (left == right) return part->state[left] > 0;

            while (left < right) {
                if (part->state[left] > 0) {
                    left++;
                    has_alive = 1;
                }
                else if (part->state[right] <= 0) right--;
                else {
                    LocalParticle_exchange(part, left, right);
                    left++;
                    right--;
                    swap_made = 1;
                }
            }

            return swap_made || has_alive;
        #else
            return 1;
        #endif
        }

        /*gpufun*/
        void count_reorganized_particles(LocalParticle* part) {
            int64_t num_active = 0;
            int64_t num_lost = 0;

            for (int64_t i = part->ipart; i < part->endpart; i++) {
                if (part->state[i] <= -999999999) break;
                else if (part->state[i] > 0) num_active++;
                else num_lost++;
            }

            part->_num_active_particles = num_active;
            part->_num_lost_particles = num_lost;
        }

        #else // not CPU_SERIAL_IMPLEM and not CPU_OMP_IMPLEM

        /*gpufun*/
        int64_t check_is_active(LocalParticle* part) {
            return LocalParticle_get_state(part)>0;
        };

        #endif // CPU_OMP_IMPLEM
        #endif // CPU_SERIAL_IMPLEM

        #undef CPU_SERIAL_IMPLEM //only_for_context cpu_serial
        #undef CPU_OMP_IMPLEM //only_for_context cpu_openmp


        '''

        source = '\n\n'.join([src_typedef, src_adders, src_getters,
                                 src_setters, src_scalers, src_exchange,
                                 src_particles_to_local, src_local_to_particles,
                                 src_angles, custom_source])

        source += """
                    #ifdef XTRACK_GLOBAL_XY_LIMIT

                    /*gpufun*/
                    void global_aperture_check(LocalParticle* part0) {
                        //start_per_particle_block (part0->part)
                            double const x = LocalParticle_get_x(part);
                            double const y = LocalParticle_get_y(part);

                        int64_t const is_alive = (int64_t)(
                                  (x >= -XTRACK_GLOBAL_XY_LIMIT) &&
                                  (x <=  XTRACK_GLOBAL_XY_LIMIT) &&
                                  (y >= -XTRACK_GLOBAL_XY_LIMIT) &&
                                  (y <=  XTRACK_GLOBAL_XY_LIMIT) );

                        // I assume that if I am in the function is because
                            if (!is_alive){
                               LocalParticle_set_state(part, -1);
                        }
                        //end_per_particle_block
                    }

                    #endif

                    /*gpufun*/
                    void LocalParticle_add_to_energy(LocalParticle* part, double delta_energy, int pz_only ){
                        double ptau = LocalParticle_get_ptau(part);
                        double const p0c = LocalParticle_get_p0c(part);
                        double const charge_ratio = LocalParticle_get_charge_ratio(part);
                        double const chi = LocalParticle_get_chi(part);
                        double const mass_ratio = charge_ratio / chi;

                        ptau += delta_energy/p0c / mass_ratio;

                        double const old_rpp = LocalParticle_get_rpp(part);

                        LocalParticle_update_ptau(part, ptau);

                        if (!pz_only) {
                            double const new_rpp = LocalParticle_get_rpp(part);
                            double const f = old_rpp / new_rpp;
                            LocalParticle_scale_px(part, f);
                            LocalParticle_scale_py(part, f);
                        }
                    }


                    /*gpufun*/
                    void LocalParticle_update_p0c(LocalParticle* part, double new_p0c_value){

                        double const mass0 = LocalParticle_get_mass0(part);
                        double const old_p0c = LocalParticle_get_p0c(part);
                        double const old_delta = LocalParticle_get_delta(part);
                        double const old_beta0 = LocalParticle_get_beta0(part);

                        double const ppc = old_p0c * old_delta + old_p0c;
                        double const new_delta = (ppc - new_p0c_value)/new_p0c_value;

                        double const new_energy0 = sqrt(new_p0c_value*new_p0c_value + mass0 * mass0);
                        double const new_beta0 = new_p0c_value / new_energy0;
                        double const new_gamma0 = new_energy0 / mass0;

                        LocalParticle_set_p0c(part, new_p0c_value);
                        LocalParticle_set_gamma0(part, new_gamma0);
                        LocalParticle_set_beta0(part, new_beta0);

                        LocalParticle_update_delta(part, new_delta);

                        LocalParticle_scale_px(part, old_p0c/new_p0c_value);
                        LocalParticle_scale_py(part, old_p0c/new_p0c_value);

                        LocalParticle_scale_zeta(part, new_beta0/old_beta0);

                    }

                    /*gpufun*/
                    void LocalParticle_kill_particle(LocalParticle* part, int64_t kill_state) {
                        LocalParticle_set_x(part, 1e30);
                        LocalParticle_set_px(part, 1e30);
                        LocalParticle_set_y(part, 1e30);
                        LocalParticle_set_py(part, 1e30);
                        LocalParticle_set_zeta(part, 1e30);
                        LocalParticle_update_delta(part, -1);  // zero energy
                        LocalParticle_set_state(part, kill_state);
                    }
                """
        return source

    @classmethod
    def part_energy_varnames(cls):
        return [vv for tt, vv in cls.part_energy_vars]

    def init_independent_per_part_vars(self, kwargs):
        self.s = kwargs.get('s', 0)
        self.at_turn = kwargs.get('at_turn', 0)
        self.at_element = kwargs.get('at_element', 0)
        self.weight = kwargs.get('weight', 1)
        self.x = kwargs.get('x', 0)
        self.y = kwargs.get('y', 0)
        self.px = kwargs.get('px', 0)
        self.py = kwargs.get('py', 0)
        self.ax = kwargs.get('ax', 0)
        self.ay = kwargs.get('ay', 0)

        pdg_id = kwargs.get('pdg_id')
        try:
            pdg_id = get_pdg_id_from_name(pdg_id)
            if not np.isscalar(pdg_id):
                pdg_id = self._context.nparray_to_context_array(pdg_id)
            self.pdg_id = pdg_id
        except ModuleNotFoundError:
            if pdg_id is not None:
                raise ValueError("In order to specify `pdg_id` you must have "
                                 "xpart installed, however, it's not currently "
                                 "available.")
            self.pdg_id = 0

    @classmethod
    def reference_from_pdg_id(cls, pdg_id, **kwargs):
        pdg_id = get_pdg_id_from_name(pdg_id)
        kwargs['pdg_id'] = pdg_id
        q0 = kwargs.get('q0')
        mass0 = kwargs.get('mass0')
        if q0 is None:
            q, _, _, _ = get_properties_from_pdg_id(pdg_id)
            kwargs['q0'] = q
        if mass0 is None:
            kwargs['mass0'] = get_mass_from_pdg_id(pdg_id)

        particle_ref = cls(**kwargs)
        if particle_ref._capacity > 1:
            raise ValueError("The method `reference_from_pdg_id` should have "
                           + "a `_capacity` of 1. Make sure all properties are "
                           + "single entries!")
        return particle_ref

    def _allclose(self, a, b, rtol=1e-05, atol=1e-08, mask=None):
        """Substitute for np.allclose that works with all contexts, and
        allows for masking. Mask is expected to be an integer array on pyopencl,
        and a boolean array on other contexts.
        """
        if isinstance(self._context, xo.ContextPyopencl):
            # PyOpenCL does not support np.allclose
            whr = _mask_to_where(mask, ctx=self._context)
            if not np.isscalar(a) or not len(a.shape) == 0:
                a = a[whr]
            if not np.isscalar(b) or not len(b.shape) == 0:
                b = b[whr]
            c = abs(a - b)
            # We use the same formula as in numpy:
            return not bool((c > (atol + rtol * abs(b))).any())
        else:
            if mask is not None:
                a = a[mask]
                b = b[mask]
            return np.allclose(a, b, rtol, atol, equal_nan=True)

    def _assert_values_consistent(self, given_value, computed_value, mask=None):
        """Check if the given value is consistent with the computed value."""
        if given_value is None:
            return
        if not self._allclose(given_value, computed_value, mask=mask):
            raise ValueError(
                f'The given value {given_value} is not consistent with the '
                f'computed value {computed_value}. Difference: '
                f'{abs(given_value - computed_value)}.'
            )

    def _setattr_if_consistent(self, varname, given_value, computed_value,
                               mask=None):
        """Update field values that may be both given and computed from others.

        In case of small differences between the given value and them computed
        value, the given value will prevail to preserve numerical stability.
        This is useful when two or more dependent variables are given as input.
        """
        self._assert_values_consistent(given_value, computed_value, mask)
        target_val = given_value if given_value is not None else computed_value

        # The simple case
        if mask is None:
            setattr(self, varname, target_val)
            return

        # Assign with a mask
        if isinstance(self._context, xo.ContextPyopencl):  # PyOpenCL array
            mask = _mask_to_where(mask, self._context)

        getattr(self, varname)[mask] = target_val[mask]

    def _update_refs(self, p0c=None, energy0=None, gamma0=None, beta0=None,
                     mask=None):
        if not any(ff is not None for ff in (p0c, energy0, gamma0, beta0)):
            self._p0c = 1e9
            p0c = self._p0c

        _sqrt = self._context.nplike_lib.sqrt

        if p0c is not None:
            _energy0 = _sqrt(p0c ** 2 + self.mass0 ** 2)
            _beta0 = p0c / _energy0
            _gamma0 = _energy0 / self.mass0
            _p0c = p0c
        elif energy0 is not None:
            _p0c = _sqrt(energy0 ** 2 - self.mass0 ** 2)
            _beta0 = _p0c / energy0
            _gamma0 = energy0 / self.mass0
        elif gamma0 is not None:
            _beta0 = _sqrt(1 - 1 / gamma0 ** 2)
            _energy0 = self.mass0 * gamma0
            _p0c = _energy0 * _beta0
            _gamma0 = gamma0
        elif beta0 is not None:
            _gamma0 = 1 / _sqrt(1 - beta0 ** 2)
            _energy0 = self.mass0 * _gamma0
            _p0c = _energy0 * beta0
            _beta0 = beta0
        else:
            raise RuntimeError('This statement is unreachable.')

        self._assert_values_consistent(energy0, self.mass0 * _gamma0, mask)
        self._setattr_if_consistent('_p0c',
                                    given_value=p0c,
                                    computed_value=_p0c,
                                    mask=mask)
        self._setattr_if_consistent('_gamma0',
                                    given_value=gamma0,
                                    computed_value=_gamma0,
                                    mask=mask)
        self._setattr_if_consistent('_beta0',
                                    given_value=beta0,
                                    computed_value=_beta0,
                                    mask=mask)

    def _update_energy_deviations(self, delta=None, ptau=None, pzeta=None,
                                  _rpp=None, _rvv=None, mask=None):
        if all(ff is None for ff in (delta, ptau, pzeta)):
            if _rpp is not None or _rvv is not None:
                raise ValueError('Setting `delta` and `ptau` by only giving '
                                 '`_rpp` and `_rvv` is not supported.')
            if any(self.mass_ratio != 1.0):
                raise ValueError('Need to provide `delta` or `ptau` with '
                                 'non-default mass ratios.')
            self._delta = 0.0
            delta = self._delta  # Cupy complains if we later assign LinkedArray

        _sqrt = self._context.nplike_lib.sqrt

        beta0 = self._beta0
        if delta is not None:
            _delta = delta
            _ptau = _sqrt(_delta ** 2 + 2 * _delta + 1 / beta0 ** 2) - 1 / beta0
            _pzeta = _ptau / beta0
        elif ptau is not None:
            _ptau = ptau
            _delta = _sqrt(_ptau ** 2 + 2 * _ptau / beta0 + 1) - 1
            _pzeta = _ptau / beta0
        elif pzeta is not None:
            _pzeta = pzeta
            _ptau = _pzeta * beta0
            _delta = _sqrt(_ptau ** 2 + 2 * _ptau / beta0 + 1) - 1
        else:
            raise RuntimeError('This statement is unreachable.')

        self._assert_values_consistent(pzeta, _pzeta, mask)
        self._setattr_if_consistent('_delta',
                                    given_value=delta,
                                    computed_value=_delta,
                                    mask=mask)
        self._setattr_if_consistent('_ptau',
                                    given_value=ptau,
                                    computed_value=_ptau,
                                    mask=mask)

        delta = self._delta  # Cupy complains if we later assign LinkedArray
        delta_beta0 = delta * beta0
        ptau_beta0 = _sqrt(delta_beta0 ** 2 + 2 * delta_beta0 * beta0 + 1) - 1
        new_rvv = (1 + delta) / (1 + ptau_beta0)
        new_rpp = 1 / (1 + delta)
        self._setattr_if_consistent('_rpp',
                                    given_value=_rpp,
                                    computed_value=new_rpp,
                                    mask=mask)
        self._setattr_if_consistent('_rvv',
                                    given_value=_rvv,
                                    computed_value=new_rvv,
                                    mask=mask)

    def _update_zeta(self, zeta=None, tau=None, mask=None):
        if zeta is None and tau is None:
            self.zeta = 0.0
            zeta = self.zeta

        beta0 = self._beta0

        if zeta is not None:
            _zeta = zeta
            _tau = zeta / beta0
        elif tau is not None:
            _tau = tau
            _zeta = beta0 * _tau
        else:
            raise RuntimeError('This statement is unreachable.')

        self._assert_values_consistent(tau, _tau, mask)
        self._setattr_if_consistent('zeta',
                                    given_value=zeta,
                                    computed_value=_zeta,
                                    mask=mask)

    def _update_chi_charge_ratio(self, chi=None, charge_ratio=None,
                                 mass_ratio=None, mask=None):
        num_args = sum(ff is not None for ff in (chi, charge_ratio, mass_ratio))

        if num_args == 0:
            self.chi = 1.0
            self.charge_ratio = 1.0
            return

        elif num_args == 1:
            raise ValueError('Two of `chi`, `charge_ratio` and `mass_ratio` '
                                 'must be provided.')

        elif num_args == 2:
            if chi is None:
                _charge_ratio, _mass_ratio = charge_ratio, mass_ratio
                _chi = charge_ratio / mass_ratio
            elif charge_ratio is None:
                _chi, _mass_ratio = chi, mass_ratio
                _charge_ratio = chi * mass_ratio
            elif mass_ratio is None:
                _chi, _charge_ratio = chi, charge_ratio
                _mass_ratio = charge_ratio / chi
            else:
                raise RuntimeError('This statement is unreachable.')

        else:  # num_args == 3
            _chi, _charge_ratio, _mass_ratio = chi, charge_ratio, mass_ratio

        self._assert_values_consistent(mass_ratio, _mass_ratio, mask)
        self._setattr_if_consistent('chi',
                                    given_value=chi,
                                    computed_value=_chi,
                                    mask=mask)
        self._setattr_if_consistent('charge_ratio',
                                    given_value=charge_ratio,
                                    computed_value=_charge_ratio,
                                    mask=mask)

    def update_p0c_and_energy_deviations(self, p0c, update_pxpy=False):

        assert np.isscalar(p0c), 'p0c must be a scalar'

        # Assign with a mask
        mask = self.state > 0

        old_p0c = self.p0c.copy()
        old_beta0 = self.beta0.copy()
        old_delta = self.delta.copy()

        PC = (old_delta + 1) * old_p0c
        new_delta = PC / p0c - 1

        new_p0c = p0c + 0 * old_p0c

        self._update_refs(p0c=new_p0c, mask=mask)
        self._update_energy_deviations(mask=mask, delta=new_delta)
        self._update_zeta(mask=mask, zeta=self.zeta * self.beta0 / old_beta0)

        if update_pxpy:
            if isinstance(self._context, xo.ContextPyopencl):
                raise NotImplementedError # Issue wiht masking
            scale_pxpy = old_p0c[mask] / new_p0c[mask]
            self.px[mask] *= scale_pxpy
            self.py[mask] *= scale_pxpy


def _mask_to_where(mask, ctx):
    if hasattr(mask, 'get'):
        mask = mask.get()
    whr = np.where(mask)[0]
    whr = ctx.nparray_to_context_array(whr)
    return whr


def reference_from_pdg_id(pdg_id, **kwargs):
    return Particles.reference_from_pdg_id(pdg_id, **kwargs)
