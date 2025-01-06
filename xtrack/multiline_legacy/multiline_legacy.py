import pandas as pd
import numpy as np
from copy import deepcopy

import xdeps as xd
import xobjects as xo
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
            self._multiline_vars = xt.line.LineVars(self)
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

        if hasattr(self, '_bb_config') and self._bb_config is not None:
            dct['_bb_config'] = {}
            for nn, vv in self._bb_config.items():
                if nn == 'dataframes':
                    dct['_bb_config'][nn] = {}
                    for kk, vv in vv.items():
                        if vv is not None:
                            dct['_bb_config'][nn][kk] = vv.to_dict()
                        else:
                            dct['_bb_config'][nn][kk] = None
                else:
                    dct['_bb_config'][nn] = vv

        dct["metadata"] = deepcopy(self.metadata)

        return dct

    @classmethod
    def from_dict(cls, dct):

        '''
        Load a multiline from a dictionary.

        Parameters
        ----------
        dct: dict
            The dictionary with the multiline data.

        Returns
        -------
        new_multiline: Multiline
            The multiline object.
        '''

        lines = {}
        for nn, ll in dct['lines'].items():
            lines[nn] = xt.Line.from_dict(ll)

        new_multiline = cls(lines=lines, link_vars=('_var_manager' in dct))

        if '_var_manager' in dct:
            for kk in dct['_var_management_data'].keys():
                new_multiline._var_sharing.data[kk].update(
                                                dct['_var_management_data'][kk])
            new_multiline._var_sharing.manager.load(dct['_var_manager'])

        if '_bb_config' in dct:
            new_multiline._bb_config = dct['_bb_config']
            for nn, vv in dct['_bb_config']['dataframes'].items():
                if vv is not None:
                    df = pd.DataFrame(vv)
                else:
                    df = None
                new_multiline._bb_config[
                    'dataframes'][nn] = df

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

    def copy(self):
        '''
        Returns a deep copy of the multiline.
        '''
        return self.__class__.from_dict(self.to_dict())

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
            return self.vv[key]

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

    def install_beambeam_interactions(self, clockwise_line, anticlockwise_line,
                                      ip_names,
                                      num_long_range_encounters_per_side,
                                      num_slices_head_on,
                                      harmonic_number, bunch_spacing_buckets,
                                      sigmaz,
                                      delay_at_ips_slots=None):

        '''
        Install beam-beam elements in the lines. Elements are inserted in the
        lines in the appropriate positions. They are not configured and are kept
        inactive.

        Parameters
        ----------
        clockwise_line: str
            Name of the line in which the beam-beam elements for the clockwise
            beam are installed.
        anticlockwise_line: xt.Line
            Name of the line in which the beam-beam elements for the
            anticlockwise beam are installed.
        ip_names: list
            The names of the IPs in the lines around which the beam-beam
            elements need to be installed.
        num_long_range_encounters_per_side: dict
            The number of long range encounters per side for each IP.
        num_slices_head_on: int
            The number of slices to be used for  the head-on beam-beam interaction.
        harmonic_number: int
            The harmonic number of the machine.
        bunch_spacing_buckets: float
            The bunch spacing in buckets.
        sigmaz: float
            The longitudinal size of the beam.
        delay_at_ips_slots: list
            Delay between the two beams in bunch slots for each IP. It specifies
            which bunch of the anticlockwise beam interacts with bunch zero of
            the clockwise beam.

        '''

        if isinstance(num_long_range_encounters_per_side, dict):
            num_long_range_encounters_per_side = [
                num_long_range_encounters_per_side[nn] for nn in ip_names]

        # Trackers need to be invalidated to add elements
        for nn, ll in self.lines.items():
            ll.unfreeze()

        if clockwise_line is not None and anticlockwise_line is not None:
            circumference_cw = self.lines[clockwise_line].get_length()
            circumference_acw = self.lines[anticlockwise_line].get_length()
            assert np.isclose(circumference_cw, circumference_acw,
                              atol=1e-4, rtol=0)

        import xfields as xf
        bb_df_cw, bb_df_acw = xf.install_beambeam_elements_in_lines(
            line_b1=self.lines.get(clockwise_line, None),
            line_b4=self.lines.get(anticlockwise_line, None),
            ip_names=ip_names,
            num_long_range_encounters_per_side=num_long_range_encounters_per_side,
            num_slices_head_on=num_slices_head_on,
            harmonic_number=harmonic_number,
            bunch_spacing_buckets=bunch_spacing_buckets,
            sigmaz_m=sigmaz, delay_at_ips_slots=delay_at_ips_slots)

        self._bb_config = {
            'dataframes': {
                'clockwise': bb_df_cw,
                'anticlockwise': bb_df_acw
            },
            'ip_names': ip_names,
            'clockwise_line': clockwise_line,
            'anticlockwise_line': anticlockwise_line,
            'bunch_spacing_buckets': bunch_spacing_buckets,
            'harmonic_number': harmonic_number
        }

    def configure_beambeam_interactions(self, num_particles,
                                    nemitt_x, nemitt_y, crab_strong_beam=True,
                                    use_antisymmetry=False,
                                    separation_bumps=None):

        '''
        Configure the beam-beam elements in the lines.

        Parameters
        ----------
        num_particles: float
            The number of particles per bunch.
        nemitt_x: float
            The normalized emittance in the horizontal plane.
        nemitt_y: float
            The normalized emittance in the vertical plane.
        crab_strong_beam: bool
            If True, crabbing of the strong beam is taken into account.
        use_antisymmetry: bool
            If True, the antisymmetry of the optics and orbit is used to compute
            the momenta of the beam-beam interaction (in the absence of the
            counter-rotating beam)
        separation_bumps: dict
            Dictionary previding the plane of the separation bump in the IPs
            where separation is present. The keys are the IP names and the
            values are the plane ("x" or "y"). This information needs to be
            provided only when use_antisymmetry is True.

        '''

        # Check that the context in which the trackers are built is on CPU
        for nn in ["clockwise", "anticlockwise"]:
            if self._bb_config[f"{nn}_line"] is None:
                continue
            line = self.lines[self._bb_config[f"{nn}_line"]]
            if not isinstance(line.tracker._context, xo.ContextCpu):
                raise ValueError(
                    "The trackers need to be built on CPU before "
                    "configuring the beam-beam elements."
                )

        if self._bb_config['dataframes']['clockwise'] is not None:
            bb_df_cw = self._bb_config['dataframes']['clockwise'].copy()
        else:
            bb_df_cw = None

        if self._bb_config['dataframes']['anticlockwise'] is not None:
            bb_df_acw = self._bb_config['dataframes']['anticlockwise'].copy()
        else:
            bb_df_acw = None

        import xfields as xf
        xf.configure_beam_beam_elements(
            bb_df_cw=bb_df_cw,
            bb_df_acw=bb_df_acw,
            line_cw=self.lines.get(self._bb_config['clockwise_line'], None),
            line_acw=self.lines.get(self._bb_config['anticlockwise_line'], None),
            num_particles=num_particles,
            nemitt_x=nemitt_x, nemitt_y=nemitt_y,
            crab_strong_beam=crab_strong_beam,
            ip_names=self._bb_config['ip_names'],
            use_antisymmetry=use_antisymmetry,
            separation_bumps=separation_bumps)

        self.vars['beambeam_scale'] = 1.0

        for nn in ['clockwise', 'anticlockwise']:
            if self._bb_config[f'{nn}_line'] is  None: continue

            line = self.lines[self._bb_config[f'{nn}_line']]
            df = self._bb_config['dataframes'][nn]

            for bbnn in df.index:
                self.vars[f'{bbnn}_scale_strength'] = self.vars['beambeam_scale']
                line.element_refs[bbnn].scale_strength = self.vars[f'{bbnn}_scale_strength']

    def apply_filling_pattern(self, filling_pattern_cw, filling_pattern_acw,
                             i_bunch_cw, i_bunch_acw):

        '''
        Enable only he beam-beam elements corresponding to actual encounters
        for the given filling pattern and the selected bunches.

        Parameters
        ----------

        filling_pattern_cw: list or array
            The filling pattern for the clockwise beam.
        filling_pattern_acw: list or array
            The filling pattern for the anticlockwise beam.
        i_bunch_cw: int
            The index of the bunch to be simulated for the clockwise beam.
        i_bunch_acw: int
            The index of the bunch to be simulated for the anticlockwise beam.
        '''

        import xfields as xf
        apply_filling_pattern = (
            xf.config_tools.beambeam_config_tools.config_tools.apply_filling_pattern)

        apply_filling_pattern(collider=self, filling_pattern_cw=filling_pattern_cw,
                            filling_pattern_acw=filling_pattern_acw,
                            i_bunch_cw=i_bunch_cw, i_bunch_acw=i_bunch_acw)


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
