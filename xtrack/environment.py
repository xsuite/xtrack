from collections import Counter, UserDict
from functools import cmp_to_key
from typing import Literal
from weakref import WeakSet
from copy import deepcopy

import numpy as np
import pandas as pd

import xobjects as xo
import xtrack as xt
from xdeps.refs import is_ref
from .multiline_legacy.multiline_legacy import MultilineLegacy

ReferType = Literal['entry', 'center']

def _flatten_components(components, refer: ReferType = 'center'):
    if refer not in {'entry', 'center', 'centre', 'exit'}:
        raise ValueError(
            f'Allowed values for refer are "entry", "center" and "exit". Got "{refer}".'
        )

    flatt_components = []
    for nn in components:
        if isinstance(nn, Place) and isinstance(nn.name, xt.Line):
            if refer == 'exit':
                raise NotImplementedError(
                    'Refer "exit" is not yet implemented for lines.'
                )
            line = nn.name
            if not line.element_names:
                continue
            sub_components = list(line.element_names).copy()
            if nn.at is not None:
                if isinstance(nn.at, str):
                    at = line._xdeps_eval.eval(nn.at)
                else:
                    at = nn.at
                if refer == 'center' or refer == 'centre':
                    at_first_element = at - line.get_length() / 2 + line[0].length / 2
                else:
                    at_first_element = at
                sub_components[0] = Place(sub_components[0], at=at_first_element, from_=nn.from_)
            flatt_components += sub_components
        elif isinstance(nn, xt.Line):
            flatt_components += nn.element_names
        else:
            flatt_components.append(nn)

    return flatt_components

class Environment:
    def __init__(self, element_dict=None, particle_ref=None, _var_management=None,
                 lines=None):
        self._element_dict = element_dict or {}
        self.particle_ref = particle_ref

        if _var_management is not None:
            self._var_management = _var_management
        else:
            self._init_var_management()

        self.lines = EnvLines(self)
        self._lines_weakrefs = WeakSet()
        self._drift_counter = 0
        self.ref = EnvRef(self)

        if lines is not None:
            for nn, ll in lines.items():
                self.import_line(line=ll, suffix_for_common_elements='__'+nn,
                    line_name=nn)

        self.metadata = {}

    def __getstate__(self):
        out = self.__dict__.copy()
        out.pop('_lines_weakrefs')
        out.pop('_xdeps_eval_obj', None)
        return out

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lines_weakrefs = WeakSet()

    def new(self, name, parent, mode=None, at=None, from_=None,
            anchor=None, from_anchor=None,
            extra=None,
            mirror=False, import_from=None, **kwargs):

        '''
        Create a new element or line.

        Parameters
        ----------
        name : str
            Name of the new element or line
        parent : str or class
            Parent class or name of the parent element
        mode : str, optional
             - clone: clone the parent element or line.
               The parent element or line is copied, together with the associated
               expressions.
             - replica: replicate the parent elements or lines are made.
             - import: clone from a different environment. `import_from` must be
               provided.
        at : float or str, optional
            Position of the created object.
        from_: str, optional
            Name of the element from which the position is calculated (its center
            is used as reference).
        mirror : bool, optional
            Can only be used when cloning lines. If True, the order of the elements
            is reversed.
        import_from : Environment, optional. Only to be used when mode is 'import'.

        Returns
        -------
        str or Place
            Name of the created element or line or a Place object if at or from_ is
            provided.
        '''

        if from_ is not None or at is not None:
            all_kwargs = locals()
            all_kwargs.pop('self')
            all_kwargs.pop('at')
            all_kwargs.pop('from_')
            all_kwargs.pop('anchor')
            all_kwargs.pop('from_anchor')
            all_kwargs.pop('kwargs')
            all_kwargs.update(kwargs)
            return Place(self.new(**all_kwargs), at=at, from_=from_,
                         anchor=anchor, from_anchor=from_anchor)

        _ALLOWED_ELEMENT_TYPES_IN_NEW = xt.line._ALLOWED_ELEMENT_TYPES_IN_NEW
        _ALLOWED_ELEMENT_TYPES_DICT = xt.line._ALLOWED_ELEMENT_TYPES_DICT
        _STR_ALLOWED_ELEMENT_TYPES_IN_NEW = xt.line._STR_ALLOWED_ELEMENT_TYPES_IN_NEW

        if parent in self.lines:
            parent = self.lines[parent]

        if isinstance(parent, xt.Line):
            assert len(kwargs) == 0, 'No kwargs allowed when creating a line'
            if mode == 'replica':
                assert name is not None, 'Name must be provided when replicating a line'
                return parent.replicate(name=name, mirror=mirror)
            else:
                assert mode in [None, 'clone'], f'Unknown mode {mode}'
                assert name is not None, 'Name must be provided when cloning a line'
                return parent.clone(name=name, mirror=mirror)

        assert mirror is False, 'mirror=True only allowed when cloning lines.'

        if parent is xt.Line or (parent=='Line' and (
            'Line' not in self.lines and 'Line' not in self.element_dict)):
            assert mode is None, 'Mode not allowed when cls is Line'
            return self.new_line(name=name, **kwargs)

        if mode == 'replica':
            assert parent in self.element_dict, f'Element {parent} not found, cannot replicate'
            kwargs['parent_name'] = xo.String(parent)
            parent = xt.Replica
        elif mode == 'clone':
            assert parent in self.element_dict, f'Element {parent} not found, cannot clone'
        else:
            assert mode is None, f'Unknown mode {mode}'

        _eval = self._xdeps_eval.eval

        assert isinstance(parent, str) or parent in _ALLOWED_ELEMENT_TYPES_IN_NEW, (
            'Only '
            + _STR_ALLOWED_ELEMENT_TYPES_IN_NEW
            + ' elements are allowed in `new` for now.')

        needs_instantiation = True
        parent_element = None
        if isinstance(parent, str):
            if parent in self.element_dict:
                # Clone an existing element
                self.element_dict[name] = xt.Replica(parent_name=parent)
                xt.Line.replace_replica(self, name)

                parent_element = self.element_dict[name]
                parent = type(parent_element)
                needs_instantiation = False
            elif parent in _ALLOWED_ELEMENT_TYPES_DICT:
                parent = _ALLOWED_ELEMENT_TYPES_DICT[parent]
                needs_instantiation = True
            else:
                raise ValueError(f'Element type {parent} not found')

        if parent == xt.Bend and ('angle' in kwargs or 'rbarc' in kwargs):
            kwargs = _handle_bend_kwargs(kwargs, _eval, env=self)
        kwargs.pop('rbarc', None)
        kwargs.pop('rbend', None)

        ref_kwargs, value_kwargs = _parse_kwargs(parent, kwargs, _eval)

        if needs_instantiation: # Parent is a class and not another element
            self.element_dict[name] = parent(**value_kwargs)

        _set_kwargs(name=name, ref_kwargs=ref_kwargs, value_kwargs=value_kwargs,
                    element_dict=self.element_dict, element_refs=self.element_refs)

        if extra is not None:
            assert isinstance(extra, dict)
            self.element_dict[name].extra = extra

        return name

    def _init_var_management(self, dct=None):

        self._var_management = xt.line._make_var_management(element_dict=self.element_dict,
                                               dct=dct)
        self._line_vars = xt.line.LineVars(self)


    def new_line(self, components=None, name=None, refer: ReferType = 'center'):

        '''
        Create a new line.

        Parameters
        ----------
        components : list, optional
            List of components to be added to the line. It can include strings,
            place objects, and lines.
        name : str, optional
            Name of the new line.

        Returns
        -------
        line
            The new line.

        Examples
        --------
        .. code-block:: python

            env = xt.Environment()
            env['a'] = 3 # Define a variable
            env.new('mq1', xt.Quadrupole, length=0.3, k1='a')  # Create an element
            env.new('mq2', xt.Quadrupole, length=0.3, k1='-a')  # Create another element

            ln = env.new_line(name='myline', components=[
                'mq',  # Add the element 'mq' at the start of the line
                env.new('mymark', xt.Marker, at=10.0),  # Create a marker at s=10
                env.new('mq1_clone', 'mq1', k1='2a'),   # Clone 'mq1' with a different k1
                env.place('mq2', at=20.0, from='mymark'),  # Place 'mq2' at s=20
                ])
        '''

        out = xt.Line()
        out.particle_ref = self.particle_ref
        out.env = self
        out._element_dict = self.element_dict # Avoid copying
        if components is None:
            components = []

        for ii, nn in enumerate(components):
            if (isinstance(nn, Place) and isinstance(nn.name, str)
                    and nn.name in self.lines):
                nn.name = self.lines[nn.name]
            if isinstance(nn, str) and nn in self.lines:
                components[ii] = self.lines[nn]

        flattened_components = _flatten_components(components, refer=refer)
        out.element_names = handle_s_places(flattened_components, self, refer=refer)
        out._var_management = self._var_management
        out._name = name
        out.builder = Builder(env=self, components=components)

        # Temporary solution to keep consistency in multiline
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            out._var_management = None
            out._in_multiline = self._in_multiline
            out._name_in_multiline = self._name_in_multiline

        self._lines_weakrefs.add(out) # Weak references
        if name is not None:
            self.lines[name] = out

        return out

    def place(self, name, at=None, from_=None, anchor=None, from_anchor=None):
        '''
        Create a place object.

        Parameters
        ----------
        name : str or Line
            Name of the element or line to be placed.
        at : float or str, optional
            Position of the created object.
        from_: str, optional
            Name of the element from which the position is calculated (its center
            is used as reference).

        Returns
        -------
        Place
            The new place object.
        '''

        return Place(name, at=at, from_=from_, anchor=anchor, from_anchor=from_anchor)

    def new_builder(self, components=None, name=None, refer: ReferType = 'center'):
        '''
        Create a new builder.

        Parameters
        ----------
        components : list, optional
            List of components to be added to the builder. It can include strings,
            place objects, and lines.
        name : str, optional
            Name of the line that will be built by the builder.

        Returns
        -------
        Builder
            The new builder.
        '''

        return Builder(env=self, components=components, name=name, refer=refer)

    def call(self, filename):
        '''
        Call a file with xtrack commands.

        Parameters
        ----------
        filename : str
            Name of the file to be called.
        '''
        with open(filename) as fid:
            code = fid.read()
        import xtrack
        xtrack._passed_env = self
        try:
            exec(code)
        except Exception as ee:
            xtrack._passed_env = None
            raise ee
        xtrack._passed_env = None

    def copy_element_from(self, name, source, new_name=None):
        return xt.Line.copy_element_from(self, name, source, new_name)

    def import_line(
            self,
            line,
            suffix_for_common_elements=None,
            line_name=None,
            overwrite_vars=False,
    ):
        """Import a line into this environment.

        Parameters
        ----------
        suffix_for_common_elements : str, optional
            Suffix to be added to the names of the elements that are common to
            the imported line and the line in this environment. If None,
            '_{source_line_name}' is used.
        line_name : str, optional
            Name of the new line. If None, the name of the imported line is used.
        overwrite_vars : bool, optional
            If True, the variables in the imported line will overwrite the
            variables with the same name in this environment. Default is False.
        """
        line_name = line_name or line.name
        if suffix_for_common_elements is None:
            suffix_for_common_elements = f'_{line_name}'

        new_var_values = line.ref_manager.containers['vars']._owner
        if not overwrite_vars:
            new_var_values = new_var_values.copy()
            new_var_values.update(self.ref_manager.containers['vars']._owner)
        self.ref_manager.containers['vars']._owner.update(new_var_values)

        self.ref_manager.copy_expr_from(line.ref_manager, 'vars', overwrite=overwrite_vars)
        self.ref_manager.run_tasks()

        components = []
        for name in line.element_names:
            new_name = name
            if (name in self.element_dict and not
                (isinstance(line[name], xt.Marker)
                and isinstance(self.element_dict.get(name), xt.Marker))):
                new_name += suffix_for_common_elements

            self.copy_element_from(name, line, new_name=new_name)

            components.append(new_name)

        out = self.new_line(components=components, name=line_name)

        if line.particle_ref is not None:
            out.particle_ref = line.particle_ref.copy()

    def _ensure_tracker_consistency(self, buffer):
        for ln in self._lines_weakrefs:
            if ln._has_valid_tracker() and ln._buffer is not buffer:
                ln.discard_tracker()

    def _get_a_drift_name(self):
        self._drift_counter += 1
        nn = f'drift_{self._drift_counter}'
        if nn not in self.element_dict:
            return nn
        else:
            return self._get_a_drift_name()

    def __setitem__(self, key, value):

        if isinstance(value, xt.Line):
            assert value.env is self, 'Line must be in the same environment'
            if key in self.lines:
                raise ValueError(f'There is already a line with name {key}')
            if key in self.element_dict:
                raise ValueError(f'There is already an element with name {key}')
            self.lines[key] = value
        else:
            xt.Line.__setitem__(self, key, value)

    def to_dict(self, include_var_management=True):

        out = {}
        out["elements"] = {k: el.to_dict() for k, el in self.element_dict.items()}

        if self.particle_ref is not None:
            out['particle_ref'] = self.particle_ref.to_dict()
        if self._var_management is not None and include_var_management:
            if hasattr(self, '_in_multiline') and self._in_multiline is not None:
                raise ValueError('The line is part ot a MultiLine object. '
                    'To save without expressions please use '
                    '`line.to_dict(include_var_management=False)`.\n'
                    'To save also the deferred expressions please save the '
                    'entire multiline.\n ')

            out.update(self._var_management_to_dict())

        if hasattr(self, '_bb_config') and self._bb_config is not None:
            out['_bb_config'] = {}
            for nn, vv in self._bb_config.items():
                if nn == 'dataframes':
                    out['_bb_config'][nn] = {}
                    for kk, vv in vv.items():
                        if vv is not None:
                            out['_bb_config'][nn][kk] = vv.to_dict()
                        else:
                            out['_bb_config'][nn][kk] = None
                else:
                    out['_bb_config'][nn] = vv

        out["metadata"] = deepcopy(self.metadata)

        out['xsuite_data_type'] = 'Environment'

        out['lines'] = {}

        for nn, ll in self.lines.items():
            out['lines'][nn] = ll.to_dict(include_element_dict=False,
                                        include_var_management=False)

        return out

    @classmethod
    def from_dict(cls, dct):
        cls = xt.Environment

        ldummy = xt.Line.from_dict(dct)
        out = cls(element_dict=ldummy.element_dict, particle_ref=ldummy.particle_ref,
                _var_management=ldummy._var_management)
        out._line_vars = xt.line.LineVars(out)

        for nn in dct['lines'].keys():
            ll = xt.Line.from_dict(dct['lines'][nn], env=out, verbose=False)
            out[nn] = ll

        if '_bb_config' in dct:
            out._bb_config = dct['_bb_config']
            for nn, vv in dct['_bb_config']['dataframes'].items():
                if vv is not None:
                    df = pd.DataFrame(vv)
                else:
                    df = None
                out._bb_config[
                    'dataframes'][nn] = df

        if "metadata" in dct:
            out.metadata = dct["metadata"]

        return out

    @classmethod
    def from_json(cls, file, **kwargs):

        """Constructs an environment from a json file.

        Parameters
        ----------
        file : str or file-like object
            Path to the json file or file-like object.
            If filename ends with '.gz' file is decompressed.
        **kwargs : dict
            Additional keyword arguments passed to `Environment.from_dict`.

        Returns
        -------
        environment : Environment
            Environment object.

        """

        dct = xt.json.load(file)

        return cls.from_dict(dct, **kwargs)


    def to_json(self, file, indent=1, **kwargs):
        '''Save the environment to a json file.

        Parameters
        ----------
        file: str or file-like object
            The file to save to. If a string is provided, a file is opened and
            closed. If a file-like object is provided, it is used directly.
        **kwargs:
            Additional keyword arguments are passed to the `Environment.to_dict` method.

        '''

        xt.json.dump(self.to_dict(**kwargs), file, indent=indent)

    @property
    def line_names(self):
        return list(self.lines.keys())

    @property
    def functions(self):
        return self._xdeps_fref

    def __getattr__(self, key):
        if key == 'lines':
            return object.__getattribute__(self, 'lines')
        if key in self.lines:
            return self.lines[key]
        else:
            raise AttributeError(f"Environment object has no attribute `{key}`.")

    def __dir__(self):
        return [nn for nn  in list(self.lines.keys()) if '.' not in nn
                    ] + object.__dir__(self)

    element_dict = xt.Line.element_dict
    _xdeps_vref = xt.Line._xdeps_vref
    _xdeps_fref = xt.Line._xdeps_fref
    _xdeps_manager = xt.Line._xdeps_manager
    _xdeps_eval = xt.Line._xdeps_eval
    element_refs = xt.Line.element_refs
    vars = xt.Line.vars
    varval = xt.Line.varval
    vv = xt.Line.vv
    __getitem__ = xt.Line.__getitem__
    set = xt.Line.set
    get = xt.Line.get
    eval = xt.Line.eval
    info = xt.Line.info
    get_expr = xt.Line.get_expr
    new_expr = xt.Line.new_expr
    ref_manager = xt.Line.ref_manager
    _var_management_to_dict = xt.Line._var_management_to_dict

    twiss = MultilineLegacy.twiss
    discard_trackers = MultilineLegacy.discard_trackers
    build_trackers = MultilineLegacy.build_trackers
    match = MultilineLegacy.match
    match_knob = MultilineLegacy.match_knob
    from_madx = MultilineLegacy.from_madx
    install_beambeam_interactions = MultilineLegacy.install_beambeam_interactions
    configure_beambeam_interactions =  MultilineLegacy.configure_beambeam_interactions
    apply_filling_pattern = MultilineLegacy.apply_filling_pattern


class Place:

    def __init__(self, name, at=None, from_=None, anchor=None, from_anchor=None):

        if isinstance(at, str) and '@' in at:
            at_parts = at.split('@')
            assert len(at_parts) == 2
            assert from_ is None
            assert from_anchor is None
            at = 0
            from_ = at_parts[1]
            from_anchor = at_parts[0]

        if from_ is not None:
            assert isinstance(from_, str)
            if '@' in from_:
                from_parts = from_.split('@')
                assert len(from_parts) == 2
                from_ = from_parts[1]
                from_anchor = from_parts[0]

        self.name = name
        self.at = at
        self.from_ = from_
        self.anchor = anchor
        self.from_anchor = from_anchor
        self._before = False

    def __repr__(self):
        out = f'Place({self.name}'
        if self.at is not None: out += f', at={self.at}'
        if self.from_ is not None: out += f', from_={self.from_}'
        if self.anchor is not None: out += f', anchor={self.anchor}'
        if self.from_anchor is not None: out += f', anchor={self.from_anchor}'

        if self._before:
            out += ', before=True'
        out += ')'
        return out

    def copy(self):
        out = Place('dummy')
        out.__dict__ = self.__dict__.copy()
        return out

def _all_places(seq):
    seq_all_places = []
    for ss in seq:
        if isinstance(ss, Place):
            seq_all_places.append(ss)
        elif not isinstance(ss, str) and hasattr(ss, '__iter__'):
            # Find first place
            i_first = None
            for ii, sss in enumerate(ss):
                if isinstance(sss, Place):
                    i_first = ii
                    break
                assert isinstance(sss, str) or isinstance(sss, xt.Line), (
                    'Only places, elements, strings or Lines are allowed in sequences')
            ss_aux = _all_places(ss)
            if i_first is not None:
                for ii in range(i_first):
                    ss_aux[ii]._before = True
            seq_all_places.extend(ss_aux)
        else:
            assert isinstance(ss, str) or isinstance(ss, xt.Line), (
                'Only places, elements, strings or Lines are allowed in sequences')
            seq_all_places.append(Place(ss, at=None, from_=None))
    return seq_all_places

# In case we want to allow for the length to be an expression
# def _length_expr_or_val(name, line):
#     if isinstance(line[name], xt.Replica):
#         name = line[name].resolve(line, get_name=True)

#     if not line[name].isthick:
#         return 0

#     if line.element_refs[name]._expr is not None:
#         return line.element_refs[name]._expr
#     else:
#         return line[name].length

def _compute_one_s(at, anchor, from_anchor, self_length, from_length, s_entry_from,
                   default_anchor):

    if is_ref(at):
        at = at._value

    if anchor is None:
        anchor = default_anchor

    if from_anchor is None:
        from_anchor = default_anchor

    s_from = 0
    if from_length is not None:
        s_from = s_entry_from
        if from_anchor == 'center' or from_anchor == 'centre':
            s_from += from_length / 2
        elif from_anchor == 'end':
            s_from += from_length

    ds_self = 0
    if anchor == 'center' or anchor=='centre':
        ds_self = self_length / 2
    elif anchor == 'end':
        ds_self = self_length

    s_entry_self = s_from + at - ds_self

    return s_entry_self

def _resolve_s_positions(seq_all_places, env, refer: ReferType = 'center',
                         allow_duplicate_places=True, s_tol=1e-10):

    # Handle duplicate places
    if len(seq_all_places) != len(set(seq_all_places)):
        if allow_duplicate_places:
            # Make copies
            seq_all_places = [ss.copy() for ss in seq_all_places]
        else:
            raise ValueError('Duplicate places detected and `allow_duplicates` is False')

    names_unsorted = [ss.name for ss in seq_all_places]

    aux_line = env.new_line(components=names_unsorted, refer=refer)
    temp_tt = aux_line.get_table()
    temp_tt['length'] = np.diff(temp_tt.s, append=temp_tt.s[-1])
    temp_tt = temp_tt.rows[:-1] # Remove endpoint
    tt_lengths = xt.Table({'name': temp_tt.env_name, 'length': temp_tt.length})

    s_entry_for_place = {}  # entry positions
    place_for_name = {}
    n_resolved = 0
    n_resolved_prev = -1

    assert len(seq_all_places) == len(set(seq_all_places)), 'Duplicate places detected'

    if seq_all_places[0].at is None and not seq_all_places[0]._before:
        # In case we want to allow for the length to be an expression
        s_entry_for_place[seq_all_places[0]] = 0
        place_for_name[seq_all_places[0].name] = seq_all_places[0]
        n_resolved += 1

    while n_resolved != n_resolved_prev:
        n_resolved_prev = n_resolved
        for ii, ss in enumerate(seq_all_places):
            if ss in s_entry_for_place:  # Already resolved
                continue
            if ss.at is None and not ss._before:
                ss_prev = seq_all_places[ii-1]
                if ss_prev in s_entry_for_place:
                    s_entry_for_place[ss] = (s_entry_for_place[ss_prev]
                                             + tt_lengths['length', ss_prev.name])
                    place_for_name[ss.name] = ss
                    n_resolved += 1
            elif ss.at is None and ss._before:
                ss_next = seq_all_places[ii+1]
                if ss_next in s_entry_for_place:
                    s_entry_for_place[ss] = (s_entry_for_place[ss_next]
                                            - tt_lengths['length', ss.name])
                    place_for_name[ss.name] = ss
                    n_resolved += 1
            else:
                if isinstance(ss.at, str):
                    at = aux_line._xdeps_eval.eval(ss.at)
                else:
                    at = ss.at

                from_length=None
                s_entry_from=None
                if ss.from_ is not None:
                    from_length = tt_lengths['length', ss.from_]
                    s_entry_from=s_entry_for_place[place_for_name[ss.from_]]

                s_entry_for_place[ss] = _compute_one_s(at, anchor=ss.anchor,
                    from_anchor=ss.from_anchor,
                    self_length=tt_lengths['length', ss.name],
                    from_length=from_length,
                    s_entry_from=s_entry_from,
                    default_anchor=refer)

                place_for_name[ss.name] = ss
                n_resolved += 1

    if n_resolved != len(seq_all_places):
        unresolved_pos = set(seq_all_places) - set(s_entry_for_place.keys())
        raise ValueError(f'Could not resolve all s positions: {unresolved_pos}')

    if n_resolved != len(seq_all_places):
        unresolved_pos = set(seq_all_places) - set(s_entry_for_place.keys())
        raise ValueError(f'Could not resolve all s positions: {unresolved_pos}')

    # Sorting
    aux_tt = temp_tt

    aux_s_entry = np.array([s_entry_for_place[ss] for ss in seq_all_places])
    aux_s_center = aux_s_entry + aux_tt['length'] / 2 # Need to sort the centers to avoid issues
                                                      # with thin + thick elements at the same s_entry
    aux_tt['s_entry'] = aux_s_entry
    aux_tt['s_center'] = aux_s_center

    aux_tt['from_'] = np.array([ss.from_ for ss in seq_all_places])
    aux_tt['from_anchor'] = np.array([ss.from_anchor for ss in seq_all_places])
    aux_tt['i_place'] = np.arange(len(seq_all_places))

    all_from = []
    for ss in seq_all_places:
        if ss.from_ is not None and ss.from_ not in all_from:
            all_from.append(ss.from_)

    # Sort by s_center
    iii = _argsort_s(aux_tt.s_center, tol=10e-10)
    aux_tt = aux_tt.rows[iii]

    group_id = np.zeros(len(aux_tt), dtype=int)
    group_id[0] = 0
    for ii in range(1, len(aux_tt)):
        if abs(aux_tt.s_center[ii] - aux_tt.s_center[ii-1]) < s_tol:
            group_id[ii] = group_id[ii-1]
        else:
            group_id[ii] = group_id[ii-1] + 1

    aux_tt['group_id'] = group_id
    aux_tt.show(cols=['group_id', 's_center', 'name', 'from_', 'from_anchor', 'i_place'])

    n_places = len(seq_all_places)
    i_start_group = 0
    names_sorted = []
    while i_start_group < n_places:
        i_group = aux_tt['group_id', i_start_group]
        i_end_group = i_start_group + 1
        while i_end_group < n_places and aux_tt['group_id', i_end_group] == i_group:
            i_end_group += 1
        print(f'Group {i_group}: {aux_tt.name[i_start_group:i_end_group]}')

        n_group = i_end_group - i_start_group
        if n_group == 1: # Single element
            names_sorted.append(aux_tt.name[i_start_group])
            i_start_group = i_end_group
            continue

        if np.all(aux_tt.from_anchor[i_start_group:i_end_group] == None): # Nothing to do
            names_sorted.extend(list(aux_tt.name[i_start_group:i_end_group]))
            i_start_group = i_end_group

        tt_group = aux_tt.rows[i_start_group:i_end_group]
        tt_group.show(cols=['s_center', 'name', 'from_', 'from_anchor'])

        for ff in tt_group.from_:
            i_from_global = aux_tt.rows.indices[ff][0] - i_start_group
            key_sort = np.zeros(n_group, dtype=int)

            if i_from_global < 0:
                key_sort[:] = 2
            elif i_from_global >= n_group:
                key_sort[:] = -2
            else:
                i_local = tt_group.rows.indices[ff][0] # I need to use this because it might change in the group resortings
                key_sort[i_local] = 0
                key_sort[:i_local] = -2
                key_sort[i_local+1:] = 2

            from_present = tt_group['from_']
            from_anchor_present = tt_group['from_anchor']

            mask_pack_before = (from_present == ff) & (from_anchor_present == 'start')
            mask_pack_after = (from_present == ff) & (from_anchor_present == 'end')
            key_sort[mask_pack_before] = -1
            key_sort[mask_pack_after] = 1

            # tt_group[f'key_sort_{ff}'] = key_sort
            tt_group = tt_group.rows[np.argsort(key_sort, kind='stable')]

        names_sorted.extend(list(tt_group.name))
        i_start_group = i_end_group

    tt_sorted = aux_tt.rows[names_sorted]

    tt_sorted['s_center'] = tt_sorted['s_entry'] + tt_sorted['length'] / 2
    tt_sorted['s_exit'] = tt_sorted['s_entry'] + tt_sorted['length']

    tt_sorted['ds_upstream'] = 0 * tt_sorted['s_entry']
    tt_sorted['ds_upstream'][1:] = tt_sorted['s_entry'][1:] - tt_sorted['s_exit'][:-1]
    tt_sorted['ds_upstream'][0] = tt_sorted['s_entry'][0]
    tt_sorted['s'] = tt_sorted['s_entry']
    assert np.all(tt_sorted.name == np.array(names_sorted))

    return tt_sorted

def _generate_element_names_with_drifts(env, tt_sorted, s_tol=1e-10):

    names_with_drifts = []
    # Create drifts
    for ii, nn in enumerate(tt_sorted.name):
        ds_upstream = tt_sorted['ds_upstream', ii]
        if np.abs(ds_upstream) > s_tol:
            assert ds_upstream > 0, f'Negative drift length: {ds_upstream}, upstream of {nn}'
            drift_name = env._get_a_drift_name()
            env.new(drift_name, xt.Drift, length=ds_upstream)
            names_with_drifts.append(drift_name)
        names_with_drifts.append(nn)

    return list(map(str, names_with_drifts))

def handle_s_places(seq, env, refer: ReferType = 'center'):

    if np.array([isinstance(ss, str) for ss in seq]).all():
        return [str(ss) for ss in seq]

    seq_all_places = _all_places(seq)
    tab_sorted = _resolve_s_positions(seq_all_places, env, refer=refer)
    names = _generate_element_names_with_drifts(env, tab_sorted)

    return names

def _parse_kwargs(cls, kwargs, _eval):
    ref_kwargs = {}
    value_kwargs = {}
    for kk in kwargs:
        if hasattr(kwargs[kk], '_value'):
            ref_kwargs[kk] = kwargs[kk]
            value_kwargs[kk] = kwargs[kk]._value
        elif (hasattr(cls, '_xofields') and kk in cls._xofields
                and xo.array.is_array(cls._xofields[kk])):
            assert hasattr(kwargs[kk], '__iter__'), (
                f'{kk} should be an iterable for {cls} element')
            ref_vv = []
            value_vv = []
            for ii, vvv in enumerate(kwargs[kk]):
                if hasattr(vvv, '_value'):
                    ref_vv.append(vvv)
                    value_vv.append(vvv._value)
                elif isinstance(vvv, str):
                    ref_vv.append(_eval(vvv))
                    if hasattr(ref_vv[-1], '_value'):
                        value_vv.append(ref_vv[-1]._value)
                    else:
                        value_vv.append(ref_vv[-1])
                else:
                    ref_vv.append(None)
                    value_vv.append(vvv)
            ref_kwargs[kk] = ref_vv
            value_kwargs[kk] = value_vv
        elif (isinstance(kwargs[kk], str) and hasattr(cls, '_xofields')
            and (kk not in cls._xofields or cls._xofields[kk].__name__ != 'String')):
            ref_kwargs[kk] = _eval(kwargs[kk])
            if hasattr(ref_kwargs[kk], '_value'):
                value_kwargs[kk] = ref_kwargs[kk]._value
            else:
                value_kwargs[kk] = ref_kwargs[kk]
        elif isinstance(kwargs[kk], xo.String):
            vvv = kwargs[kk].to_str()
            value_kwargs[kk] = vvv
        else:
            value_kwargs[kk] = kwargs[kk]

    return ref_kwargs, value_kwargs

def _set_kwargs(name, ref_kwargs, value_kwargs, element_dict, element_refs):
    for kk in value_kwargs:
        if hasattr(value_kwargs[kk], '__iter__') and not isinstance(value_kwargs[kk], str):
            len_value = len(value_kwargs[kk])
            getattr(element_dict[name], kk)[:len_value] = value_kwargs[kk]
            if kk in ref_kwargs:
                for ii, vvv in enumerate(value_kwargs[kk]):
                    if ref_kwargs[kk][ii] is not None:
                        getattr(element_refs[name], kk)[ii] = ref_kwargs[kk][ii]
        elif kk in ref_kwargs:
            setattr(element_refs[name], kk, ref_kwargs[kk])
        else:
            setattr(element_dict[name], kk, value_kwargs[kk])

class EnvRef:
    def __init__(self, env):
        self.env = env

    def __getitem__(self, name):
        if hasattr(self.env, 'lines') and name in self.env.lines:
            return self.env.lines[name].ref
        elif name in self.env.element_dict:
            return self.env.element_refs[name]
        elif name in self.env.vars:
            return self.env.vars[name]
        else:
            raise KeyError(f'Name {name} not found.')

    def __setitem__(self, key, value):
        if isinstance(value, xt.Line):
            assert value.env is self.env, 'Line must be in the same environment'
            if key in self.env.lines:
                raise ValueError(f'There is already a line with name {key}')
            if key in self.env.element_dict:
                raise ValueError(f'There is already an element with name {key}')
            self.env.lines[key] = value

        if hasattr(value, '_value'):
            val_ref = value
            val_value = value._value
        else:
            val_ref = value
            val_value = value

        if np.isscalar(val_value):
            if key in self.env.element_dict:
                raise ValueError(f'There is already an element with name {key}')
            self.env.vars[key] = val_ref
        else:
            if key in self.env.vars:
                raise ValueError(f'There is already a variable with name {key}')
            self.element_refs[key] = val_ref

def _handle_bend_kwargs(kwargs, _eval, env=None, name=None):
    kwargs = kwargs.copy()
    rbarc = kwargs.pop('rbarc', True)
    rbend = kwargs.pop('rbend', False)

    if rbarc:
        assert 'angle' in kwargs, 'Angle must be specified for a bend with rbarc'

    if env is not None and name is not None:
        for kk in 'h length edge_entry_angle edge_exit_angle'.split():
            if kk not in kwargs:
                expr = getattr(env.ref[name], kk)._expr
                if expr is not None:
                    kwargs[kk] = expr
                else:
                    kwargs[kk] = getattr(env.get(name), kk)
        if 'angle' in kwargs:
            kwargs.pop('h')

    length = kwargs.get('length', 0)
    if isinstance(length, str):
        length = _eval(length)

    if 'angle' in kwargs:
        assert 'h' not in kwargs, 'Cannot specify both angle and h'
        assert 'length' in kwargs, 'Length must be specified for a bend'

        angle = kwargs.pop('angle')

        if isinstance(angle, str):
            angle = _eval(angle)

        kwargs['h'] = angle / length

        if rbend and rbarc:
            fsin = env._xdeps_fref['sin']
            fsinc = env._xdeps_fref['sinc']
            kwargs['h'] = fsin(0.5*angle) / (0.5 * length) # here length is the straight line
            kwargs['length'] = length / fsinc(0.5*angle)
    else:
        angle = kwargs.get('h', 0) * length

    if rbend:
        edge_entry_angle = kwargs.pop('edge_entry_angle', 0.)
        if isinstance(edge_entry_angle, str):
            edge_entry_angle = _eval(edge_entry_angle)
        edge_exit_angle = kwargs.pop('edge_exit_angle', 0.)
        if isinstance(edge_exit_angle, str):
            edge_exit_angle = _eval(edge_exit_angle)

        edge_entry_angle += angle / 2
        edge_exit_angle += angle / 2

        kwargs['edge_entry_angle'] = edge_entry_angle
        kwargs['edge_exit_angle'] = edge_exit_angle

    if kwargs.pop('k0_from_h', False):
        kwargs['k0'] = kwargs.get('h', 0)

    return kwargs


class Builder:
    def __init__(self, env, components=None, name=None, refer: ReferType = 'center'):
        self.env = env
        self.components = components or []
        self.name = name
        self.refer = refer

    def __repr__(self):
        return f'Builder({self.name}, components={self.components!r})'

    def new(self, name, cls, at=None, from_=None, extra=None, **kwargs):
        out = self.env.new(
            name, cls, at=at, from_=from_, extra=extra, **kwargs)
        self.components.append(out)
        return out

    def place(self, name, at=None, from_=None, anchor=None, from_anchor=None):
        out = self.env.place(
            name, at=at, from_=from_, anchor=anchor, from_anchor=from_anchor)
        self.components.append(out)
        return out

    def build(self, name=None):
        if name is None:
            name = self.name
        out =  self.env.new_line(components=self.components, name=name, refer=self.refer)
        out.builder = self
        return out

    def set(self, *args, **kwargs):
        self.components.append(self.env.set(*args, **kwargs))

    def get(self, *args, **kwargs):
        return self.env.get(*args, **kwargs)

    @property
    def element_dict(self):
        return self.env.element_dict

    @property
    def ref(self):
        return self.env.ref

    @property
    def vars(self):
        return self.env.vars

    def __getitem__(self, key):
        return self.env[key]

    def __setitem__(self, key, value):
        self.env[key] = value


class EnvLines(UserDict):

    def __init__(self, env):
        self.data = {}
        self.env = env

    def __setitem__(self, key, value):
        self.env._lines_weakrefs.add(value)
        UserDict.__setitem__(self, key, value)

def get_environment(verbose=False):
    import xtrack
    if hasattr(xtrack, '_passed_env') and xtrack._passed_env is not None:
        if verbose:
            print('Using existing environment')
        return xtrack._passed_env
    else:
        if verbose:
            print('Creating new environment')
        return Environment()

def _argsort_s(seq, tol=10e-10):
    """Argsort, but with a tolerance; `sorted` is stable."""
    seq_indices = np.arange(len(seq))

    def comparator(i, j):
        a, b = seq[i], seq[j]
        if np.abs(a - b) < tol:
            return 0
        return -1 if a < b else 1

    return sorted(seq_indices, key=cmp_to_key(comparator))