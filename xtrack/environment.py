from collections import Counter, UserDict
from functools import cmp_to_key
from typing import Literal
from weakref import WeakSet

import numpy as np

import xobjects as xo
import xtrack as xt
from xdeps.refs import is_ref

ReferType = Literal['entry', 'centre']


def _argsort(seq, tol=10e-10):
    """Argsort, but with a tolerance; `sorted` is stable."""
    seq_indices = np.arange(len(seq))

    def comparator(i, j):
        a, b = seq[i], seq[j]
        if np.abs(a - b) < tol:
            return 0
        return -1 if a < b else 1

    return sorted(seq_indices, key=cmp_to_key(comparator))


def _flatten_components(components, refer: ReferType = 'centre'):
    if refer not in {'entry', 'centre', 'exit'}:
        raise ValueError(
            f'Allowed values for refer are "entry", "centre" and "exit". Got "{refer}".'
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
                if refer == 'centre':
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
    def __init__(self, element_dict=None, particle_ref=None, _var_management=None):
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

    def __getstate__(self):
        out = self.__dict__.copy()
        out.pop('_lines_weakrefs')
        return out

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lines_weakrefs = WeakSet()

    def new(self, name, parent, mode=None, at=None, from_=None, extra=None,
            mirror=False, **kwargs):

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
        at : float or str, optional
            Position of the created object.
        from_: str, optional
            Name of the element from which the position is calculated (its center
            is used as reference).
        mirror : bool, optional
            Can only be used when cloning lines. If True, the order of the elements
            is reversed.

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
            all_kwargs.pop('kwargs')
            all_kwargs.update(kwargs)
            return Place(self.new(**all_kwargs), at=at, from_=from_)

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


    def new_line(self, components=None, name=None, refer: ReferType = 'centre'):

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

    def new_builder(self, components=None, name=None, refer: ReferType = 'centre'):
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


class Place:

    def __init__(self, name, at=None, from_=None, anchor=None, from_anchor=None):

        if anchor is not None:
            raise ValueError('anchor not implemented')
        if from_anchor is not None:
            raise ValueError('from_anchor not implemented')

        self.name = name
        self.at = at
        self.from_ = from_
        self.anchor = anchor
        self.from_anchor = from_anchor
        self._before = False

    def __repr__(self):
        out = f'Place({self.name}, at={self.at}, from_={self.from_}'
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


def _resolve_s_positions(seq_all_places, env, refer: ReferType = 'center'):

    if len(seq_all_places) != len(set(seq_all_places)):
        seq_all_places = [ss.copy() for ss in seq_all_places]

    names_unsorted = [ss.name for ss in seq_all_places]

    # identify duplicates
    if len(names_unsorted) != len(set(names_unsorted)):
        counter = Counter(names_unsorted)
        duplicates = set([name for name, count in counter.items() if count > 1])
    else:
        duplicates = set()

    aux_line = env.new_line(components=names_unsorted, refer=refer)
    aux_tt = aux_line.get_table()
    aux_tt['length'] = np.diff(aux_tt._data['s'], append=0)
    aux_tt.name = aux_tt.env_name  # I want the repeated names here

    s_center_for_place = {}
    s_entry_for_place = {}  # entry positions calculated assuming at is also pointing to entry
    s_exit_for_place = {}  # exit positions calculated assuming at is also pointing to exit
    place_for_name = {}
    n_resolved = 0
    n_resolved_prev = -1

    assert len(seq_all_places) == len(set(seq_all_places)), 'Duplicate places detected'

    if seq_all_places[0].at is None and not seq_all_places[0]._before:
        # In case we want to allow for the length to be an expression
        s_center_for_place[seq_all_places[0]] = aux_tt['length', seq_all_places[0].name] / 2
        s_entry_for_place[seq_all_places[0]] = 0
        s_exit_for_place[seq_all_places[0]] = aux_tt['length', seq_all_places[0].name]
        place_for_name[seq_all_places[0].name] = seq_all_places[0]
        n_resolved += 1

    while n_resolved != n_resolved_prev:
        n_resolved_prev = n_resolved
        for ii, ss in enumerate(seq_all_places):
            if ss in s_center_for_place:  # Can this ever happen? We assert no duplicates earlier.
                continue
            if ss.at is None and not ss._before:
                ss_prev = seq_all_places[ii-1]
                if ss_prev in s_center_for_place:
                    # in case we want to allow for the length to be an expression
                    # s_center_dct[ss] = (s_center_dct[ss_prev]
                    #                         + _length_expr_or_val(ss_prev, aux_line) / 2
                    #                         + _length_expr_or_val(ss, aux_line) / 2)
                    s_center_for_place[ss] = (s_center_for_place[ss_prev]
                                              + aux_tt['length', ss_prev.name] / 2
                                              + aux_tt['length', ss.name] / 2)
                    s_entry_for_place[ss] = (s_entry_for_place[ss_prev]
                                             + aux_tt['length', ss_prev.name])
                    s_exit_for_place[ss] = (s_exit_for_place[ss_prev]
                                            + aux_tt['length', ss.name])
                    place_for_name[ss.name] = ss
                    n_resolved += 1
            elif ss.at is None and ss._before:
                ss_next = seq_all_places[ii+1]
                if ss_next in s_center_for_place:
                    # in case we want to allow for the length to be an expression
                    # s_center_dct[ss] = (s_center_dct[ss_next]
                    #                         - _length_expr_or_val(ss_next, aux_line) / 2
                    #                         - _length_expr_or_val(ss, aux_line) / 2)
                    s_center_for_place[ss] = (s_center_for_place[ss_next]
                                              - aux_tt['length', ss_next.name] / 2
                                              - aux_tt['length', ss.name] / 2)
                    s_entry_for_place[ss] = (s_entry_for_place[ss_next]
                                            - aux_tt['length', ss.name])
                    s_exit_for_place[ss] = (s_exit_for_place[ss_next]
                                            - aux_tt['length', ss_next.name])
                    place_for_name[ss.name] = ss
                    n_resolved += 1
            else:
                if isinstance(ss.at, str):
                    at = aux_line._xdeps_eval.eval(ss.at)
                else:
                    at = ss.at

                if ss.from_ is None:
                    s_center_for_place[ss] = at
                    s_entry_for_place[ss] = at
                    s_exit_for_place[ss] = at
                    place_for_name[ss.name] = ss
                    n_resolved += 1
                elif ss.from_ in place_for_name:
                    if ss.from_ in duplicates:
                        assert ss.name in duplicates, (
                            f'Cannot resolve from_ for {ss.name} as {ss.from_} is duplicated')
                    s_center_for_place[ss] = s_center_for_place[place_for_name[ss.from_]] + at
                    s_entry_for_place[ss] = s_entry_for_place[place_for_name[ss.from_]] + at
                    s_exit_for_place[ss] = s_exit_for_place[place_for_name[ss.from_]] + at
                    place_for_name[ss.name] = ss
                    n_resolved += 1

    if n_resolved != len(seq_all_places):
        unresolved_pos = set(seq_all_places) - set(s_center_for_place.keys())
        raise ValueError(f'Could not resolve all s positions: {unresolved_pos}')

    if n_resolved != len(seq_all_places):
        unresolved_pos = set(seq_all_places) - set(s_center_for_place.keys())
        raise ValueError(f'Could not resolve all s positions: {unresolved_pos}')

    aux_s_center_expr = np.array([s_center_for_place[ss] for ss in seq_all_places])
    aux_s_entry_expr = np.array([s_entry_for_place[ss] for ss in seq_all_places])
    aux_s_exit_expr = np.array([s_exit_for_place[ss] for ss in seq_all_places])
    aux_s_center = [ss._value if is_ref(ss) else ss for ss in aux_s_center_expr]
    aux_s_entry = [ss._value if is_ref(ss) else ss for ss in aux_s_entry_expr]
    aux_s_exit = [ss._value if is_ref(ss) else ss for ss in aux_s_exit_expr]

    if refer == 'centre':
        aux_tt['s_center'] = np.concatenate([aux_s_center, [0]])

        i_sorted = _argsort(aux_s_center)

        name_sorted = [str(aux_tt.name[ii]) for ii in i_sorted]

        # Temporary, should be replaced by aux_tt.rows[i_sorted], when table is fixed
        data_sorted = {kk: aux_tt[kk][i_sorted] for kk in aux_tt._col_names}
        tt_sorted = xt.Table(data_sorted)

        tt_sorted['s_entry'] = tt_sorted['s_center'] - tt_sorted['length'] / 2
        tt_sorted['s_exit'] = tt_sorted['s_center'] + tt_sorted['length'] / 2
        anchor_pos_dct = s_center_for_place
    elif refer == 'entry':
        aux_tt['s_entry'] = np.concatenate([aux_s_entry, [0]])

        i_sorted = _argsort(aux_s_entry)

        name_sorted = [str(aux_tt.name[ii]) for ii in i_sorted]

        # Temporary, should be replaced by aux_tt.rows[i_sorted], when table is fixed
        data_sorted = {kk: aux_tt[kk][i_sorted] for kk in aux_tt._col_names}
        tt_sorted = xt.Table(data_sorted)

        tt_sorted['s_center'] = tt_sorted['s_entry'] + tt_sorted['length'] / 2
        tt_sorted['s_exit'] = tt_sorted['s_entry'] + tt_sorted['length']
        anchor_pos_dct = s_entry_for_place
    elif refer == 'exit':
        aux_tt['s_exit'] = np.concatenate([aux_s_exit, [0]])

        i_sorted = _argsort(aux_s_exit)

        name_sorted = [str(aux_tt.name[ii]) for ii in i_sorted]

        # Temporary, should be replaced by aux_tt.rows[i_sorted], when table is fixed
        data_sorted = {kk: aux_tt[kk][i_sorted] for kk in aux_tt._col_names}
        tt_sorted = xt.Table(data_sorted)

        tt_sorted['s_center'] = tt_sorted['s_exit'] - tt_sorted['length'] / 2
        tt_sorted['s_entry'] = tt_sorted['s_exit'] - tt_sorted['length']
        anchor_pos_dct = s_entry_for_place
    else:
        raise ValueError(f'Unknown refer value: {refer}')

    tt_sorted['ds_upstream'] = 0 * tt_sorted['s_entry']
    tt_sorted['ds_upstream'][1:] = tt_sorted['s_entry'][1:] - tt_sorted['s_exit'][:-1]
    tt_sorted['ds_upstream'][0] = tt_sorted['s_entry'][0]
    tt_sorted['s'] = tt_sorted['s_entry']
    assert np.all(tt_sorted.name == np.array(name_sorted))

    tt_sorted._data['s_entry_dct'] = anchor_pos_dct

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

def handle_s_places(seq, env, refer: ReferType = 'centre'):

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
            and kk in cls._xofields and cls._xofields[kk].__name__ != 'String'):
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
    def __init__(self, env, components=None, name=None, refer: ReferType = 'centre'):
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