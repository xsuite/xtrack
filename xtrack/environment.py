import xtrack as xt
import xobjects as xo
import numpy as np
from weakref import WeakSet
from collections import Counter, UserDict

def _flatten_components(components):
    flatt_components = []
    for nn in components:
        if isinstance(nn, Place) and isinstance(nn.name, xt.Line):
            line = nn.name
            components = list(line.element_names).copy()
            if nn.at is not None:
                if isinstance(nn.at, str):
                    at = line._xdeps_eval.eval(nn.at)
                else:
                    at = nn.at
                at_first_element = at - line.get_length() / 2 + line[0].length / 2
                components[0] = Place(components[0], at=at_first_element, from_=nn.from_)
            flatt_components += components
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

    def new_line(self, components=None, name=None):
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

        flattened_components = _flatten_components(components)
        out.element_names = handle_s_places(flattened_components, self)
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

    def new_builder(self, components=None, name=None):
        return Builder(env=self, components=components, name=name)

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

    def new(self, name, cls, mode=None, at=None, from_=None, extra=None,
            mirror=False, **kwargs):

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

        if cls in self.lines:
            cls = self.lines[cls]

        if isinstance(cls, xt.Line):
            assert len(kwargs) == 0, 'No kwargs allowed when creating a line'
            if mode == 'replica':
                assert name is not None, 'Name must be provided when replicating a line'
                return cls.replicate(name=name, mirror=mirror)
            else:
                assert mode in [None, 'clone'], f'Unknown mode {mode}'
                assert name is not None, 'Name must be provided when cloning a line'
                return cls.clone(name=name, mirror=mirror)

        assert mirror is False, 'mirror=True only allowed when cloning or  lines.'

        if cls is xt.Line or (cls=='Line' and (
            'Line' not in self.lines and 'Line' not in self.element_dict)):
            assert mode is None, 'Mode not allowed when cls is Line'
            return self.new_line(name=name, **kwargs)

        if mode == 'replica':
            assert cls in self.element_dict, f'Element {cls} not found, cannot replicate'
            kwargs['parent_name'] = xo.String(cls)
            cls = xt.Replica
        elif mode == 'clone':
            assert cls in self.element_dict, f'Element {cls} not found, cannot clone'
        else:
            assert mode is None, f'Unknown mode {mode}'

        _eval = self._xdeps_eval.eval

        assert isinstance(cls, str) or cls in _ALLOWED_ELEMENT_TYPES_IN_NEW, (
            'Only '
            + _STR_ALLOWED_ELEMENT_TYPES_IN_NEW
            + ' elements are allowed in `new` for now.')

        needs_instantiation = True
        if isinstance(cls, str):
            if cls in self.element_dict:
                # Clone an existing element
                self.element_dict[name] = xt.Replica(parent_name=cls)
                self.replace_replica(name)
                cls = type(self.element_dict[name])
                needs_instantiation = False
            elif cls in _ALLOWED_ELEMENT_TYPES_DICT:
                cls = _ALLOWED_ELEMENT_TYPES_DICT[cls]
                needs_instantiation = True
            else:
                raise ValueError(f'Element type {cls} not found')

        ref_kwargs, value_kwargs = _parse_kwargs(cls, kwargs, _eval)

        if needs_instantiation: # Parent is a class and not another element
            self.element_dict[name] = cls(**value_kwargs)

        _set_kwargs(name=name, ref_kwargs=ref_kwargs, value_kwargs=value_kwargs,
                    element_dict=self.element_dict, element_refs=self.element_refs)

        if extra is not None:
            assert isinstance(extra, dict)
            self.element_dict[name].extra = extra

        return name

    def place(self, name, at=None, from_=None, anchor=None, from_anchor=None):
        return Place(name, at=at, from_=from_, anchor=anchor, from_anchor=from_anchor)

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

Environment.element_dict = xt.Line.element_dict
Environment._init_var_management = xt.Line._init_var_management
Environment._xdeps_vref = xt.Line._xdeps_vref
Environment._xdeps_fref = xt.Line._xdeps_fref
Environment._xdeps_manager = xt.Line._xdeps_manager
Environment._xdeps_eval = xt.Line._xdeps_eval
Environment.element_refs = xt.Line.element_refs
Environment.vars = xt.Line.vars
Environment.varval = xt.Line.varval
Environment.vv = xt.Line.vv
Environment.replace_replica = xt.Line.replace_replica
Environment.__getitem__ = xt.Line.__getitem__
Environment.set = xt.Line.set
Environment.get = xt.Line.get

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


def _resolve_s_positions(seq_all_places, env):

    if len(seq_all_places) != len(set(seq_all_places)):
        seq_all_places = [ss.copy() for ss in seq_all_places]

    names_unsorted = [ss.name for ss in seq_all_places]

    # identify duplicates
    if len(names_unsorted) != len(set(names_unsorted)):
        counter = Counter(names_unsorted)
        duplicates = set([name for name, count in counter.items() if count > 1])
    else:
        duplicates = set()

    aux_line = env.new_line(components=names_unsorted)
    aux_tt = aux_line.get_table()
    aux_tt['length'] = np.diff(aux_tt._data['s'], append=0)
    aux_tt.name = aux_tt.env_name # I want the repeated names here

    s_center_dct = {}
    s_center_dct_names = {}
    n_resolved = 0
    n_resolved_prev = -1

    assert len(seq_all_places) == len(set(seq_all_places)), 'Duplicate places detected'

    if seq_all_places[0].at is None and not seq_all_places[0]._before:
        # In case we want to allow for the length to be an expression
        s_center_dct[seq_all_places[0]] = aux_tt['length', seq_all_places[0].name] / 2
        # s_center_dct[seq_all_places[0]] = _length_expr_or_val(seq_all_places[0], aux_line) / 2
        n_resolved += 1

    while n_resolved != n_resolved_prev:
        n_resolved_prev = n_resolved
        for ii, ss in enumerate(seq_all_places):
            if ss in s_center_dct:
                continue
            if ss.at is None and not ss._before:
                ss_prev = seq_all_places[ii-1]
                if ss_prev in s_center_dct:
                    # in case we want to allow for the length to be an expression
                    # s_center_dct[ss] = (s_center_dct[ss_prev]
                    #                         + _length_expr_or_val(ss_prev, aux_line) / 2
                    #                         + _length_expr_or_val(ss, aux_line) / 2)
                    s_center_dct[ss] = (s_center_dct[ss_prev]
                                            +  aux_tt['length', ss_prev.name] / 2
                                             + aux_tt['length', ss.name] / 2)
                    s_center_dct_names[ss.name] = s_center_dct[ss]
                    n_resolved += 1
            elif ss.at is None and ss._before:
                ss_next = seq_all_places[ii+1]
                if ss_next in s_center_dct:
                     # in case we want to allow for the length to be an expression
                    # s_center_dct[ss] = (s_center_dct[ss_next]
                    #                         - _length_expr_or_val(ss_next, aux_line) / 2
                    #                         - _length_expr_or_val(ss, aux_line) / 2)
                    s_center_dct[ss] = (s_center_dct[ss_next]
                                            - aux_tt['length', ss_next.name] / 2
                                            - aux_tt['length', ss.name] / 2)
                    s_center_dct_names[ss.name] = s_center_dct[ss]
                    n_resolved += 1
            else:
                if isinstance(ss.at, str):
                    at = aux_line._xdeps_eval.eval(ss.at)
                else:
                    at = ss.at

                if ss.from_ is None:
                    s_center_dct[ss] = at
                    s_center_dct_names[ss.name] = at
                    n_resolved += 1
                elif ss.from_ in s_center_dct_names:
                    if ss.from_ in duplicates:
                        assert ss.name in duplicates, (
                            f'Cannot resolve from_ for {ss.name} as {ss.from_} is duplicated')
                    s_center_dct[ss] = s_center_dct_names[ss.from_] + at
                    s_center_dct_names[ss.name] = s_center_dct[ss]
                    n_resolved += 1

    assert n_resolved == len(seq_all_places), 'Not all positions resolved'

    aux_s_center_expr = np.array([s_center_dct[ss] for ss in seq_all_places])
    aux_s_center = []
    for ss in aux_s_center_expr:
        if hasattr(ss, '_value'):
            aux_s_center.append(ss._value)
        else:
            aux_s_center.append(ss)
    aux_tt['s_center'] = np.concatenate([aux_s_center, [0]])

    i_sorted = np.argsort(aux_s_center, stable=True)

    name_sorted = [str(aux_tt.name[ii]) for ii in i_sorted]

    # Temporary, should be replaced by aux_tt.rows[i_sorted], when table is fixed
    data_sorted = {kk: aux_tt[kk][i_sorted] for kk in aux_tt._col_names}
    tt_sorted = xt.Table(data_sorted)

    tt_sorted['s_entry'] = tt_sorted['s_center'] - tt_sorted['length'] / 2
    tt_sorted['s_exit'] = tt_sorted['s_center'] + tt_sorted['length'] / 2
    tt_sorted['ds_upstream'] = 0 * tt_sorted['s_entry']
    tt_sorted['ds_upstream'][1:] = tt_sorted['s_entry'][1:] - tt_sorted['s_exit'][:-1]
    tt_sorted['ds_upstream'][0] = tt_sorted['s_entry'][0]
    tt_sorted['s'] = tt_sorted['s_center']
    assert np.all(tt_sorted.name == np.array(name_sorted))

    tt_sorted._data['s_center_dct'] = s_center_dct

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

def handle_s_places(seq, env):

    if np.array([isinstance(ss, str) for ss in seq]).all():
        return [str(ss) for ss in seq]

    seq_all_places = _all_places(seq)
    tab_sorted = _resolve_s_positions(seq_all_places, env)
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
                    value_vv.append(ref_vv[-1]._value)
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
        else:
            if kk in ref_kwargs:
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
    rbarc = kwargs.pop('rbarc', False)
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

    if isinstance(kwargs['length'], str):
        length = _eval(kwargs['length'])
    else:
        length = kwargs['length']

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

    if kwargs.pop('rbend', False):
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
        kwargs['k0'] = kwargs['h']

    return kwargs


class Builder:
    def __init__(self, env, components=None, name=None):
        self.env = env
        self.components = components or []
        self.name = name

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
        out =  self.env.new_line(components=self.components, name=name)
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
