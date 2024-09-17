import xtrack as xt
import xobjects as xo
import numpy as np
from weakref import WeakSet


def _flatten_components(components):
    flatten_components = []
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
            flatten_components += components
        elif isinstance(nn, xt.Line):
            flatten_components += nn.element_names
        else:
            flatten_components.append(nn)
    return flatten_components

class Environment:
    def __init__(self, element_dict=None, particle_ref=None, _var_management=None):
        self._element_dict = element_dict or {}
        self.particle_ref = particle_ref

        if _var_management is not None:
            self._var_management = _var_management
        else:
            self._init_var_management()

        self.lines = {}
        self._lines = WeakSet()
        self._drift_counter = 0
        self.ref = EnvRef(self)

    def new_line(self, components=None, name=None):
        out = xt.Line()
        out.particle_ref = self.particle_ref
        out.env = self
        out._element_dict = self.element_dict # Avoid copying
        if components is None:
            components = []
        flattened_components = _flatten_components(components)
        out.element_names = handle_s_places(flattened_components, self)
        out._var_management = self._var_management
        out._name = name
        self._lines.add(out) # Weak references
        if name is not None:
            self.lines[name] = out

        return out

    def _ensure_tracker_consistency(self, buffer):
        for ln in self._lines:
            if ln._has_valid_tracker() and ln._buffer is not buffer:
                ln.discard_tracker()

    def _get_a_drift_name(self):
        self._drift_counter += 1
        nn = f'drift_{self._drift_counter}'
        if nn not in self.element_dict:
            return nn
        else:
            return self._get_a_drift_name()

    def new(self, name, cls, at=None, from_=None, **kwargs):

        if from_ is not None or at is not None:
            return Place(at=at, from_=from_,
                         name=self.new(name, cls, **kwargs))

        _eval = self._xdeps_eval.eval

        assert isinstance(cls, str) or cls in [xt.Drift, xt.Bend, xt.
                       Quadrupole, xt.Sextupole, xt.Octupole,
                       xt.Multipole, xt.Marker, xt.Replica], (
            'Only Drift, Dipole, Quadrupole, Sextupole, Octupole, Multipole, Marker, and Replica '
            'elements are allowed in `new` for now.')

        cls_input = cls
        if isinstance(cls, str):
            # Clone an existing element
            assert cls in self.element_dict, f'Element {cls} not found in environment'
            self.element_dict[name] = xt.Replica(parent_name=cls)
            self.replace_replica(name)
            cls = type(self.element_dict[name])

        ref_kwargs, value_kwargs = _parse_kwargs(cls, kwargs, _eval)

        if not isinstance(cls_input, str): # Parent is a class and not another element
            self.element_dict[name] = cls(**value_kwargs)

        _set_kwargs(name=name, ref_kwargs=ref_kwargs, value_kwargs=value_kwargs,
                    element_dict=self.element_dict, element_refs=self.element_refs)

        return name

    def place(self, name, at=None, from_=None, anchor=None, from_anchor=None):
        return Place(name, at=at, from_=from_, anchor=anchor, from_anchor=from_anchor)

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
Environment.__setitem__ = xt.Line.__setitem__
Environment.set = xt.Line.set

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
        return f'Place({self.name}, at={self.at}, from_={self.from_})'

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
            if i_first is None:
                raise ValueError('No Place in sequence')
            ss_aux = _all_places(ss)
            for ii in range(i_first):
                ss_aux[ii]._before = True
            seq_all_places.extend(ss_aux)
        else:
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
    names_unsorted = [ss.name for ss in seq_all_places]
    aux_line = env.new_line(components=names_unsorted)
    aux_tt = aux_line.get_table()
    aux_tt['length'] = np.diff(aux_tt._data['s'], append=0)

    s_center_dct = {}
    n_resolved = 0
    n_resolved_prev = -1

    if seq_all_places[0].at is None and not seq_all_places[0]._before:
        # In case we want to allow for the length to be an expression
        s_center_dct[seq_all_places[0].name] = aux_tt['length', seq_all_places[0].name] / 2
        # s_center_dct[seq_all_places[0].name] = _length_expr_or_val(seq_all_places[0].name, aux_line) / 2
        n_resolved += 1

    while n_resolved != n_resolved_prev:
        n_resolved_prev = n_resolved
        for ii, ss in enumerate(seq_all_places):
            if ss.name in s_center_dct:
                continue
            if ss.at is None and not ss._before:
                ss_prev = seq_all_places[ii-1]
                if ss_prev.name in s_center_dct:
                    # in case we want to allow for the length to be an expression
                    # s_center_dct[ss.name] = (s_center_dct[ss_prev.name]
                    #                         + _length_expr_or_val(ss_prev.name, aux_line) / 2
                    #                         + _length_expr_or_val(ss.name, aux_line) / 2)
                    s_center_dct[ss.name] = (s_center_dct[ss_prev.name]
                                            +  aux_tt['length', ss_prev.name] / 2
                                             + aux_tt['length', ss.name] / 2)
                    n_resolved += 1
            elif ss.at is None and ss._before:
                ss_next = seq_all_places[ii+1]
                if ss_next.name in s_center_dct:
                     # in case we want to allow for the length to be an expression
                    # s_center_dct[ss.name] = (s_center_dct[ss_next.name]
                    #                         - _length_expr_or_val(ss_next.name, aux_line) / 2
                    #                         - _length_expr_or_val(ss.name, aux_line) / 2)
                    s_center_dct[ss.name] = (s_center_dct[ss_next.name]
                                            - aux_tt['length', ss_next.name] / 2
                                            - aux_tt['length', ss.name] / 2)
                    n_resolved += 1
            else:
                if isinstance(ss.at, str):
                    at = aux_line._xdeps_eval.eval(ss.at)
                else:
                    at = ss.at

                if ss.from_ is None:
                    s_center_dct[ss.name] = at
                    n_resolved += 1
                elif ss.from_ in s_center_dct:
                    s_center_dct[ss.name] = s_center_dct[ss.from_] + at
                    n_resolved += 1

    assert n_resolved == len(seq_all_places), 'Not all positions resolved'

    aux_s_center_expr = np.array([s_center_dct[nn] for nn in aux_tt.name[:-1]])
    aux_s_center = []
    for ss in aux_s_center_expr:
        if hasattr(ss, '_value'):
            aux_s_center.append(ss._value)
        else:
            aux_s_center.append(ss)
    aux_tt['s_center'] = np.concatenate([aux_s_center, [0]])

    i_sorted = np.argsort(aux_s_center, stable=True)

    name_sorted = [str(aux_tt.name[ii]) for ii in i_sorted]

    tt_sorted = aux_tt.rows[name_sorted]
    tt_sorted['s_entry'] = tt_sorted['s_center'] - tt_sorted['length'] / 2
    tt_sorted['s_exit'] = tt_sorted['s_center'] + tt_sorted['length'] / 2
    tt_sorted['ds_upstream'] = 0 * tt_sorted['s_entry']
    tt_sorted['ds_upstream'][1:] = tt_sorted['s_entry'][1:] - tt_sorted['s_exit'][:-1]
    tt_sorted['ds_upstream'][0] = tt_sorted['s_entry'][0]
    tt_sorted['s'] = tt_sorted['s_center']
    assert np.all(tt_sorted.name == np.array(name_sorted))

    tt_sorted._data['s_center_dct'] = s_center_dct

    return tt_sorted

def _generate_element_names_with_drifts(env, tt_sorted, s_tol=1e-12):

    names_with_drifts = []
    # Create drifts
    for nn in tt_sorted.name:
        ds_upstream = tt_sorted['ds_upstream', nn]
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
        else:
            value_kwargs[kk] = kwargs[kk]

    return ref_kwargs, value_kwargs

def _set_kwargs(name, ref_kwargs, value_kwargs, element_dict, element_refs):
    for kk in value_kwargs:
        if hasattr(value_kwargs[kk], '__iter__'):
            len_value = len(value_kwargs[kk])
            getattr(element_dict[name], kk)[:len_value] = value_kwargs[kk]
            if kk in ref_kwargs:
                for ii, vvv in enumerate(value_kwargs[kk]):
                    if vvv is not None:
                        getattr(element_refs[name], kk)[ii] = vvv
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
            raise ValueError('Cannot set a Line, please use Envirnoment.new_line')

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