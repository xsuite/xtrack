import numpy as np
from copy import deepcopy

import xdeps as xd
import xtrack as xt

class MultilineLegacy:

    '''
    Class to manage multiple beam lines (they can optionally share the xdeps vars).

    Parameters
    ----------
    lines: dict
        Dictionary with the lines objects
    link_vars: bool
        If True, the variables are linked between the lines.

    '''

    def __init__(self, lines: dict, link_vars=True):
        raise ValueError('MultilineLegacy is deprecated, use Environment instead')

        self.lines = {}
        self.lines.update(lines)

        line_names = list(self.lines.keys())
        self.line_names = line_names
        line_list = [self.lines[nn] for nn in line_names]
        if link_vars:
            self._var_sharing = xt.multiline_legacy.VarSharing(
                lines=line_list, names=line_names)
            self._multiline_vars = xt.line.EnvVars(self)
        else:
            self._var_sharing = None

        for nn, ll in zip(line_names, line_list):
            ll._in_multiline = self
            ll._name_in_multiline = nn

        self.metadata = {}

    def to_dict(self, include_var_management=True):

        '''
        Save the multiline to a dictionary.

        Parameters
        ----------
        include_var_management: bool
            If True, the variable management data is included in the dictionary.

        Returns
        -------
        dct: dict
            The dictionary with the multiline data.
        '''

        dct = {}
        if include_var_management:
            dct['_var_manager'] = self._var_sharing.manager.dump()
            dct['_var_management_data'] = self._var_sharing.data
        dct['lines'] = {}
        for nn, ll in self.lines.items():
            dct['lines'][nn] = ll.to_dict(include_var_management=False)

        dct["metadata"] = deepcopy(self.metadata)

        return dct

    @classmethod
    def from_dict(cls, dct, with_progress=True):

        '''
        Load a multiline from a dictionary.

        Parameters
        ----------
        dct: dict
            The dictionary with the multiline data.
        with_progress : bool, optional
            Whether to show progress while deserializing line elements.
            Defaults to ``True``.

        Returns
        -------
        new_multiline: Multiline
            The multiline object.
        '''

        lines = {}
        for nn, ll in dct['lines'].items():
            lines[nn] = xt.Line.from_dict(
                ll, with_progress=with_progress)

        new_multiline = cls(lines=lines, link_vars=('_var_manager' in dct))

        if '_var_manager' in dct:
            for kk in dct['_var_management_data'].keys():
                new_multiline._var_sharing.data[kk].update(
                                                dct['_var_management_data'][kk])
            new_multiline._var_sharing.manager.load(dct['_var_manager'])

        if "metadata" in dct:
            new_multiline.metadata = dct["metadata"]

        return new_multiline

    def to_json(self, file, indent=1, **kwargs):
        '''Save the multiline to a json file.

        Parameters
        ----------
        file: str or file-like object
            The file to save to. If a string is provided, a file is opened and
            closed. If filename ends with '.gz' file is compressed.
            If a file-like object is provided, it is used directly.
        **kwargs: dict
            Additional keyword arguments are passed to the `Line.to_dict` method.
        '''
        xt.json.dump(self.to_dict(**kwargs), file, indent=indent)

    @classmethod
    def from_json(cls, file, **kwargs):
        '''Load a multiline from a json file.

        Parameters
        ----------
        file: str or file-like object
            The file to load from. If a string is provided, a file is opened and
            closed. If the string endswith '.gz' the file is decompressed.
            If a file-like object is provided, it is used directly.
        **kwargs: dict

        Returns
        -------
        new_multiline: Multiline
            The multiline object.
        '''
        return cls.from_dict(xt.json.load(file), **kwargs)

    @classmethod
    def from_madx(cls, filename=None, madx=None, stdout=None, return_lines=False, **kwargs):
        '''
        Load a multiline from a MAD-X file.

        Parameters
        ----------
        file: str
            The MAD-X file to load from.
        **kwargs: dict
            Additional keyword arguments are passed to the `Line.from_madx_sequence`
            method.

        Returns
        -------
        new_multiline: Multiline
            The multiline object.
        '''
        return _multiline_from_madx(cls, filename=filename, madx=madx, stdout=stdout,
                             return_lines=return_lines, **kwargs)

    def copy(self, with_progress=True):
        '''
        Returns a deep copy of the multiline.
        '''
        return self.__class__.from_dict(
            self.to_dict(), with_progress=with_progress)

    def __getstate__(self):
        out = self.__dict__.copy()
        return out

    def __setstate__(self, state):
        self.__dict__.update(state)

    def build_trackers(self, _context=None, _buffer=None, **kwargs):
        '''
        Build the trackers for the lines.

        Parameters
        ----------
        _context: xobjects.Context
            The context in which the trackers are built.
        _buffer: xobjects.Buffer
            The buffer in which the trackers are built.
        **kwargs: dict
            Additional keyword arguments are passed to the `Line.build_tracker`
            method.

        '''

        for nn, ll in self.lines.items():
            ll.build_tracker(_context=_context, _buffer=_buffer, **kwargs)


    def discard_trackers(self):
        '''
        Discard the trackers associated to the lines.
        '''

        for nn, ll in self.lines.items():
            ll.discard_tracker()

    def twiss(self, lines=None, **kwargs):

        '''
        Compute the twiss parameters for the lines.

        Parameters
        ----------
        lines: list of str
            The lines for which the twiss parameters are computed. If None,
            the twiss parameters are computed for all lines.
        **kwargs: dict
            Additional keyword arguments are passed to the `Line.twiss` method.

        Returns
        -------
        out: MultiTwiss
            A MultiTwiss object containing the twiss parameters for the lines.
        '''

        for old, new in zip(['ele_start', 'ele_stop', 'ele_init', 'twiss_init'],
                            ['start', 'end', 'init_at', 'init']):
            if old in kwargs:
                raise ValueError(f'`{old}` is deprecated. Please use `{new}`.')

        out = MultiTwiss()
        if lines is None:
            lines = self.line_names

        kwargs, kwargs_per_twiss = _dispatch_twiss_kwargs(kwargs, lines)

        for ii, nn in enumerate(lines):
            this_kwargs = kwargs.copy()
            for kk in kwargs_per_twiss.keys():
                this_kwargs[kk] = kwargs_per_twiss[kk][ii]
            out[nn] = self.lines[nn].twiss(**this_kwargs)

        out._line_names = lines

        return out

    def match(self, vary, targets, restore_if_fail=True, solver=None,
              verbose=False, check_limits=True, **kwargs):

        '''
        Change a set of knobs in the beam lines in order to match assigned targets.

        Parameters
        ----------
        vary : list of str or list of Vary objects
            List of knobs to be varied. Each knob can be a string or a Vary object
            including the knob name and the step used for computing the Jacobian
            for the optimization.
        targets : list of Target objects
            List of targets to be matched.
        restore_if_fail : bool
            If True, the beamline is restored to its initial state if the matching
            fails.
        solver : str
            Solver to be used for the matching.
        check_limits : bool
            If True (default), the limits of the knobs are checked before the
            optimization. If False, if the knobs are out of limits, the optimization
            knobs are set to the limits on the first iteration.
        verbose : bool
            If True, the matching steps are printed.
        **kwargs : dict
            Additional arguments to be passed to the twiss.

        Returns
        -------
        result_info : dict
            Dictionary containing information about the matching result.

        '''

        for old, new in zip(['ele_start', 'ele_stop', 'ele_init', 'twiss_init'],
                            ['start', 'end', 'init_at', 'init']):
            if old in kwargs:
                raise ValueError(f'`{old}` is deprecated. Please use `{new}`.')

        line_names = kwargs.get('lines', self.line_names)
        kwargs, kwargs_per_twiss = _dispatch_twiss_kwargs(kwargs, line_names)
        kwargs.update(kwargs_per_twiss)

        return xt.match.match_line(self, vary, targets,
                          restore_if_fail=restore_if_fail,
                          solver=solver, check_limits=check_limits,
                          verbose=verbose, **kwargs)

    def match_knob(self, knob_name, vary, targets,
                knob_value_start=0, knob_value_end=1,
                **kwargs):

        '''
        Match a new knob in the beam line such that the specified targets are
        matched when the knob is set to the value `knob_value_end` and the
        state of the line before tha matching is recovered when the knob is
        set to the value `knob_value_start`.

        Parameters
        ----------
        knob_name : str
            Name of the knob to be matched.
        vary : list of str or list of Vary objects
            List of existing knobs to be varied.
        targets : list of Target objects
            List of targets to be matched.
        knob_value_start : float
            Value of the knob before the matching. Defaults to 0.
        knob_value_end : float
            Value of the knob after the matching. Defaults to 1.

        '''

        opt = xt.match.match_knob_line(self, vary=vary, targets=targets,
                        knob_name=knob_name, knob_value_start=knob_value_start,
                        knob_value_end=knob_value_end, **kwargs)

        return opt

    def __getitem__(self, key: str):
        if key in self.vars:
            return self.vars.val[key]

        if key in self.lines:
            return self.lines[key]

        raise KeyError(f'Name {key} not found')

    def __setitem__(self, key: str, value):
        if key in self.lines:
            raise ValueError(
                'Cannot create a var `{key}` using __setitem__, as there is '
                'already a line of that name in this multiline.')

        if not np.isscalar(value) and not xd.refs.is_ref(value):
            raise ValueError('Only scalars or references are allowed')

        self.vars[key] = value

    def __dir__(self):
        return list(self.lines.keys()) + object.__dir__(self)

    def __getattr__(self, key):
        if key == 'lines':
            return object.__getattribute__(self, 'lines')
        if key in self.lines:
            return self.lines[key]
        else:
            raise AttributeError(f"Multiline object has no attribute `{key}`.")

    def set(self, key, value):
        self.__setitem__(key, value)

    def get(self, key):
        return self.__getitem__(key)

    def info(self, key, limit=12):
        self.vars[key]._info(limit=limit)

    eval = xt.Line.eval
    get_expr = xt.Line.get_expr
    new_expr = xt.Line.new_expr

    @property
    def _xdeps_eval(self):
        try:
            eva_obj = self._xdeps_eval_obj
        except AttributeError:
            eva_obj = xd.madxutils.MadxEval(variables=self._xdeps_vref,
                                            functions=self.functions,
                                            elements={})
            self._xdeps_eval_obj = eva_obj

        return eva_obj

    def ref_manager(self):
        return self._var_sharing.manager

    @property
    def vars(self):
        return self._multiline_vars

    @property
    def varval(self):
        return self.vars.val

    @property
    def vv(self): # alias for varval
        return self.vars.val

    @property
    def functions(self):
        return self._xdeps_fref

    @property
    def _xdeps_vref(self):
        if self._var_sharing is not None:
            return self._var_sharing._vref

    @property
    def _xdeps_fref(self):
        if self._var_sharing is not None:
            return self._var_sharing._fref

    @property
    def _xdeps_manager(self):
        if self._var_sharing is not None:
            return self._var_sharing.manager

class MultiTwiss(dict):

    def __init__(self):
        self.__dict__ = self

def _dispatch_twiss_kwargs(kwargs, lines):
    kwargs_per_twiss = {}
    for arg_name in ['start', 'end', 'init_at', 'init',
                        '_keep_initial_particles',
                        '_initial_particles', '_ebe_monitor']:
        if arg_name not in kwargs:
            continue
        if not isinstance(kwargs[arg_name], (list, tuple)):
            kwargs_per_twiss[arg_name] = len(lines) * [kwargs[arg_name]]
            kwargs.pop(arg_name)
        else:
            assert len(kwargs[arg_name]) == len(lines), \
                f'Length of {arg_name} must be equal to the number of lines'
            kwargs_per_twiss[arg_name] = list(kwargs[arg_name])
            kwargs.pop(arg_name)
    return kwargs, kwargs_per_twiss

def _multiline_from_madx(cls, filename=None, madx=None, stdout=None, return_lines=False, **kwargs):
    '''
    Load a multiline from a MAD-X file.

    Parameters
    ----------
    file: str
        The MAD-X file to load from.
    **kwargs: dict
        Additional keyword arguments are passed to the `Line.from_madx_sequence`
        method.

    Returns
    -------
    new_multiline: Multiline
        The multiline object.
    '''
    if madx is None:
        from cpymad.madx import Madx
        madx = Madx(stdout=stdout)
    if filename is not None:
        madx.call(filename)
    lines = {}
    for nn in madx.sequence.keys():
        lines[nn] = xt.Line.from_madx_sequence(
            madx.sequence[nn],
            allow_thick=True,
            deferred_expressions=True,
            **kwargs)

        lines[nn].particle_ref = xt.Particles(
            mass0=madx.sequence[nn].beam.mass*1e9,
            q0=madx.sequence[nn].beam.charge,
            gamma0=madx.sequence[nn].beam.gamma)

        if madx.sequence[nn].beam.bv == -1:
            lines[nn].twiss_default['reverse'] = True

    if return_lines:
        return lines
    else:
        out = cls(lines=lines)
        for nn in lines.keys():
            out.lines[nn].twiss_default.update(lines[nn].twiss_default)
        return out
