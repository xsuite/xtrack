from collections.abc import Iterable
import numpy as np
import xtrack as xt
import xdeps as xd
from functools import cmp_to_key


class Builder:
    def __init__(self, env, components=None, name=None, length=None,
                 refer='center', s_tol=1e-6,
                 mirror=False):

        if refer is None:
            refer = 'center'
        self.env = env
        self.components = components or []
        self.name = name
        self.refer = refer
        self.length = length
        self.s_tol = s_tol
        self.mirror = mirror

    def copy(self):
        out = self.__class__(self.env)
        out.__dict__.update(self.__dict__)
        out.components = self.components.copy()
        return out

    def __repr__(self):
        parts = []
        if self.name:
            parts.append(f'name={self.name!r}')
        if self.length is not None:
            parts.append(f'length={self.length!r}')
        if self.refer not in {'center', 'centre'}:
            parts.append(f'refer={self.refer!r}')
        if self.mirror:
            parts.append(f'mirror={self.mirror!r}')
        parts.append(f'{len(self.components)} components')
        args_str = ', '.join(parts)
        return f'Builder({args_str})'

    def new(self, name, cls, at=None, from_=None, extra=None, force=False,
            **kwargs):
        out = self.env.new(
            name, cls, at=at, from_=from_, extra=extra, force=force, **kwargs)
        self.components.append(out)
        return out

    def place(self, name, obj=None, at=None, from_=None, anchor=None, from_anchor=None):
        out = self.env.place(name=name, obj=obj, at=at, from_=from_,
                             anchor=anchor, from_anchor=from_anchor)
        self.components.append(out)
        return out

    def build(self, name=None, inplace=None, s_tol=None, line=None):

        if inplace is None and name is None and self.name is not None:
            inplace = True

        if inplace and self.name is None:
            raise ValueError('Inplace build requires the Builder to have a name')

        if inplace:
            name = self.name

        if s_tol is None:
            s_tol = self.s_tol

        if line is not None:
            if line.env is not self.env:
                raise ValueError('Line must belong to the same environment as the Builder')

        components = self.components
        length = self.length

        if isinstance(length, str):
            length = self.env.eval(length)
        elif xd.refs.is_ref(length):
            length = length._value

        refer = self.refer

        components = _resolve_lines_in_components(components, self.env)
        flattened_components = _flatten_components(self.env, components, refer=refer)

        if np.array([isinstance(ss, str) for ss in flattened_components]).all():
            # All elements provided by name
            element_names = [str(ss) for ss in flattened_components]
            if length is not None:
                length_all_elements = self.env.new_line(components=element_names).get_length()
                if length_all_elements > length + s_tol:
                    raise ValueError(f'Line length {length_all_elements} is '
                                     f'greater than the requested length {length}')
                elif length_all_elements < length - s_tol:
                    element_names.append(self.env.new(self.env._get_a_drift_name(), xt.Drift,
                                                  length=length-length_all_elements))
        else:
            seq_all_places = _all_places(flattened_components)
            tab_unsorted = _resolve_s_positions(seq_all_places, self.env, refer=refer)
            tab_sorted = _sort_places(tab_unsorted)
            element_names = _generate_element_names_with_drifts(self.env, tab_sorted,
                                                                length=length,
                                                                s_tol=s_tol)

        if line is None:
            line = xt.Line(env=self.env, element_names=element_names)

        line.element_names = element_names

        if self.mirror:
            line.element_names = line.element_names[::-1]

        if name is not None:
            if name in self.env.lines:
                del self.env.lines[name]
            line._name = name
            self.env.lines[name] = line

        return line

    def __len__(self):
        return len(self.components)

    def resolve_s_positions(self, sort=True):
        components = self.components
        if components is None:
            components = []

        components = _resolve_lines_in_components(components, self.env)
        flattened_components = _flatten_components(self.env, components, refer=self.refer)

        seq_all_places = _all_places(flattened_components)
        tab_unsorted = _resolve_s_positions(seq_all_places, self.env, refer=self.refer)
        if not sort:
            return tab_unsorted
        tab_sorted = _sort_places(tab_unsorted)
        return tab_sorted

    def flatten(self, inplace=False):

        assert not inplace, 'Inplace not yet implemented'

        out = self.__class__(self.env)
        out.__dict__.update(self.__dict__)

        components = _resolve_lines_in_components(self.components, self.env)
        out.components = _flatten_components(self.env,components, refer=self.refer)
        out.components = _all_places(out.components)
        return out

    def to_dict(self):
        dct = {'__class__': self.__class__.__name__}
        dct['components'] = []

        formatter = xd.refs.CompactFormatter(scope=None)

        for cc in self.components:

            if isinstance(cc, str):
                dct['components'].append(cc)
                continue

            if not isinstance(cc, xt.Place):
                raise NotImplementedError('Only Place components are implemented for now')

            name = cc.name
            if hasattr(name, 'to_dict'):
                name = name.to_dict(include_element_dict=False,
                                    include_var_management=False)

            cc_dct = {}
            cc_dct['name'] = name

            if cc.at is not None:
                if xd.refs.is_ref(cc.at):
                    cc_dct['at'] = cc.at._formatted(formatter)
                else:
                    cc_dct['at'] = cc.at

            if cc.from_ is not None:
                cc_dct['from_'] = cc.from_

            if cc.anchor is not None:
                cc_dct['anchor'] = cc.anchor

            if cc.from_anchor is not None:
                cc_dct['from_anchor'] = cc.from_anchor

            dct['components'].append(cc_dct)

        if self.name is not None:
            dct['name'] = self.name

        if self.refer is not None:
            dct['refer'] = self.refer

        if self.length is not None:
            if xd.refs.is_ref(self.length):
                dct['length'] = self.length._formatted(formatter)
            else:
                dct['l'] = self.length

        if self.s_tol is not None:
            dct['s_tol'] = self.s_tol

        if self.mirror:
            dct['mirror'] = self.mirror

        return dct

    @classmethod
    def from_dict(cls, dct, env):

        dct = dct.copy()
        dct.pop('__class__', None)

        out = cls(env=env)
        components = dct.pop('components')
        for cc in components:
            if isinstance(cc, str):
                out.components.append(cc)
                continue

            if isinstance(cc['name'], dict):
                assert cc['name']['__class__'] == 'Line'
                name = xt.Line.from_dict(cc['name'], _env=env)
                cc['name'] = name
            out.place(**cc)
        for kk, vv in dct.items():
            setattr(out, kk, vv)

        return out



def _flatten_components(env, components, refer='center'):

    if refer not in ['start', 'center', 'centre', 'end']:
        raise ValueError(
            f'Allowed values for refer are "start", "center" and "end". Got "{refer}".'
        )

    flatt_components = []
    for nn in components:
        if ((is_line_from_place := (isinstance(nn, Place) and isinstance(nn.name, xt.Line)))
            or (is_line_from_str := (isinstance(nn, str) and isinstance(env[nn], xt.Line)))):

            if is_line_from_place:
                anchor = nn.anchor
                line = nn.name
            elif is_line_from_str:
                anchor = None
                line = env[nn]
            else:
                raise RuntimeError('This should never happen')

            if isinstance(line, xt.Builder):
                line = line.build(name=None, inplace=False)
            elif isinstance(line, xt.Line) and line.mode == 'compose':
                line = line.composer.build(name=None, inplace=False)

            if anchor is None:
                anchor = refer or 'center'

            if not line.element_names:
                continue
            sub_components = list(line.element_names).copy()
            if nn.at is not None:
                if isinstance(nn.at, str):
                    at = line._xdeps_eval.eval(nn.at)
                else:
                    at = nn.at
                if anchor=='center' or anchor=='centre':
                    at_of_start_first_element = at - line.get_length() / 2
                elif anchor=='end':
                    at_of_start_first_element = at - line.get_length()
                elif anchor=='start':
                    at_of_start_first_element = at
                else:
                    raise ValueError(f'Unknown anchor {anchor}')
                sub_components[0] = Place(sub_components[0], at=at_of_start_first_element,
                        anchor='start', from_=nn.from_, from_anchor=nn.from_anchor)
            flatt_components += sub_components
        elif isinstance(nn, xt.Builder):
            flatt_components += nn.build(inplace=False).element_names
        elif isinstance(nn, xt.Line):
            if nn.mode == 'compose':
                nn = nn.composer.build(name=None, inplace=False)
            flatt_components += nn.element_names
        elif isinstance(nn, Iterable) and not isinstance(nn, str):
            flatt_components += _flatten_components(env, nn, refer=refer)
        else:
            flatt_components.append(nn)

    return flatt_components

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
            seq_all_places.extend(ss_aux)
        else:
            assert isinstance(ss, str) or isinstance(ss, xt.Line), (
                'Only places, elements, strings or Lines are allowed in sequences')
            seq_all_places.append(Place(ss, at=None, from_=None))
    return seq_all_places

def _compute_one_s(at, anchor, from_anchor, self_length, from_length, s_start_from,
                   default_anchor):

    if xd.refs.is_ref(at):
        at = at._value

    if anchor is None:
        anchor = default_anchor

    if from_anchor is None:
        from_anchor = default_anchor

    s_from = 0
    if from_length is not None:
        s_from = s_start_from
        if from_anchor == 'center' or from_anchor == 'centre':
            s_from += from_length / 2
        elif from_anchor == 'end':
            s_from += from_length

    ds_self = 0
    if anchor == 'center' or anchor=='centre':
        ds_self = self_length / 2
    elif anchor == 'end':
        ds_self = self_length

    s_start_self = s_from + at - ds_self

    return s_start_self

def _resolve_s_positions(seq_all_places, env, refer='center',
                         allow_duplicate_places=True):

    if not allow_duplicate_places:
        raise NotImplementedError('allow_duplicate_places=False not yet implemented')

    seq_all_places = [ss.copy() for ss in seq_all_places]
    names_unsorted = [ss.name for ss in seq_all_places]

    aux_line = env.new_line(components=names_unsorted, refer=refer)

    # Prepare table for output
    tt_out = aux_line.get_table()
    tt_out['length'] = np.diff(tt_out.s, append=tt_out.s[-1])
    tt_out = tt_out.rows[:-1] # Remove endpoint

    tt_lengths = xt.Table({'name': tt_out.env_name, 'length': tt_out.length})

    s_start_for_place = {}  # start positions
    place_for_name = {}
    n_resolved = 0
    n_resolved_prev = -1

    assert len(seq_all_places) == len(set(seq_all_places)), 'Duplicate places detected'

    if seq_all_places[0].at is None:
        # In case we want to allow for the length to be an expression
        s_start_for_place[seq_all_places[0]] = 0
        place_for_name[seq_all_places[0].name] = seq_all_places[0]
        n_resolved += 1

    while n_resolved != n_resolved_prev:
        n_resolved_prev = n_resolved
        for ii, ss in enumerate(seq_all_places):

            if ss in s_start_for_place:  # Already resolved
                continue

            if ss.from_ is not None or ss.from_anchor is not None:
                if ss.at is None:
                    raise ValueError(
                        f'Cannot specify `from_ `or `from_anchor` without providing `at`.'
                        f'Error in place `{ss}`.')

            if ss.at is None:
                ss_prev = seq_all_places[ii-1]
                if ss_prev in s_start_for_place:
                    s_start_for_place[ss] = (s_start_for_place[ss_prev]
                                             + tt_lengths['length', ss_prev.name])
                    place_for_name[ss.name] = ss
                    if not ss_prev.name.startswith('||drift'):
                        ss.at = 0
                        ss.from_ = ss_prev.name
                        ss.from_anchor = 'end'
                    n_resolved += 1
            else:
                if isinstance(ss.at, str):
                    at = aux_line._xdeps_eval.eval(ss.at)
                elif xd.refs.is_ref(ss.at):
                    at = ss.at._value
                else:
                    at = ss.at

                from_length=None
                s_start_from=None
                if ss.from_ is not None:
                    if ss.from_ not in place_for_name:
                        continue # Cannot resolve yet
                    else:
                        from_length = tt_lengths['length', ss.from_]
                        s_start_from=s_start_for_place[place_for_name[ss.from_]]

                s_start_for_place[ss] = _compute_one_s(at, anchor=ss.anchor,
                    from_anchor=ss.from_anchor,
                    self_length=tt_lengths['length', ss.name],
                    from_length=from_length,
                    s_start_from=s_start_from,
                    default_anchor=refer)

                place_for_name[ss.name] = ss
                n_resolved += 1

    if n_resolved != len(seq_all_places):
        unresolved_pos = set(seq_all_places) - set(s_start_for_place.keys())
        raise ValueError(f'Could not resolve all s positions: {unresolved_pos}')

    if n_resolved != len(seq_all_places):
        unresolved_pos = set(seq_all_places) - set(s_start_for_place.keys())
        raise ValueError(f'Could not resolve all s positions: {unresolved_pos}')

    aux_s_start = np.array([s_start_for_place[ss] for ss in seq_all_places])
    aux_s_center = aux_s_start + tt_out['length'] / 2
    aux_s_end = aux_s_start + tt_out['length']
    tt_out['s_start'] = aux_s_start
    tt_out['s_center'] = aux_s_center
    tt_out['s_end'] = aux_s_end
    tt_out['s']= aux_s_start

    tt_out['from_'] = np.array([ss.from_ for ss in seq_all_places])
    tt_out['from_anchor'] = np.array([ss.from_anchor for ss in seq_all_places])

    return tt_out

# @profile
def _sort_places(tt_unsorted, s_tol=1e-10, allow_non_existent_from=False):

    tt_unsorted['i_place'] = np.arange(len(tt_unsorted))

    # Sort by s_center
    iii = _argsort_s(tt_unsorted.s_center, tol=s_tol)
    tt_s_sorted = tt_unsorted.rows[iii]

    # Identify groups of elements with s_center with the same s position
    # (basically thin elements, if not overlapping)
    group_id = np.zeros(len(tt_s_sorted), dtype=int)
    group_id[0] = 0
    for ii in range(1, len(tt_s_sorted)):
        if abs(tt_s_sorted.s_center[ii] - tt_s_sorted.s_center[ii-1]) > s_tol:
            group_id[ii] = group_id[ii-1] + 1
        elif tt_s_sorted.isthick[ii] and (tt_s_sorted.s_end[ii] - tt_s_sorted.s_start[ii]) != 0:
            # Needed in Line.insert (on the first sorting pass there can be overlapping elements)
            group_id[ii] = group_id[ii-1] + 1
        else:
            group_id[ii] = group_id[ii-1]

    tt_s_sorted['group_id'] = group_id
    # tt_s_sorted.show(cols=['group_id', 's_center', 'name', 'from_', 'from_anchor', 'i_place'])

    # cache indices (indices will change but only within groups, so no need to update in the loop)
    # This trick gives me x40 speedup compared to using tt_s_sorted.rows.indices
    # at each iteration.
    ind_name = {nn: ii for ii, nn in enumerate(tt_s_sorted.name)}

    # Sort elements within each group
    n_places = len(tt_s_sorted)
    i_start_group = 0
    i_place_sorted = []
    while i_start_group < n_places:

        # Identify group edges
        i_group = tt_s_sorted['group_id', i_start_group]
        i_end_group = i_start_group + 1
        while i_end_group < n_places and tt_s_sorted['group_id', i_end_group] == i_group:
            i_end_group += 1

        # Debug
        # print(f'Group {i_group}: {tt_s_sorted.name[i_start_group:i_end_group]}')

        n_group = i_end_group - i_start_group

        if n_group == 1: # Single element
            i_place_sorted.append(tt_s_sorted.i_place[i_start_group])
            i_start_group = i_end_group
            continue

        if np.all(tt_s_sorted.from_anchor[i_start_group:i_end_group] == None): # Nothing to do
            i_place_sorted.extend(list(tt_s_sorted.i_place[i_start_group:i_end_group]))
            i_start_group = i_end_group
            continue

        # Geneal case to sort thin sandwiches:
        #  - elements with from_ before the group go first (in order of appearance)
        #  - elements with no from_ go next (in order of appearance)
        #  - elements with from_ after the group go last (in order of appearance)
        #  - elements with from_ inside the group get inserted based on their from_/from_anchor

        tt_group = tt_s_sorted.rows[i_start_group:i_end_group]

        # Debug
        # tt_group.show(cols=['s_center', 'name', 'from_', 'from_anchor'])

        # Identify subgroups
        subgroup_from_is_before = []
        subgroup_from_is_after = []
        subgroup_from_is_inside = []
        subgroup_no_from = []
        for ii in range(n_group):
            ff = tt_group.from_[ii]
            if ff is None:
                subgroup_no_from.append(ii)
            else:
                if ff not in ind_name:
                    if allow_non_existent_from:
                        subgroup_no_from.append(ii)
                        continue
                    else:
                        raise ValueError(f'Element {ff} not found in the line')
                i_from_global = ind_name[ff]
                if i_from_global < i_start_group:
                    subgroup_from_is_before.append(ii)
                elif i_from_global >= i_end_group:
                    subgroup_from_is_after.append(ii)
                else:
                    subgroup_from_is_inside.append(ii)

        # Build dicts with insertions from subgroup_from_is_inside
        # (dictionary keys are the from_ names)
        insertion_before = {}
        insertion_after = {}
        for ii in subgroup_from_is_inside:
            from_ = tt_group.from_[ii]
            from_anchor = tt_group.from_anchor[ii]
            if from_anchor == 'start' or from_anchor == None:
                if from_ not in insertion_before:
                    insertion_before[from_] = []
                insertion_before[from_].append(ii)
            elif from_anchor == 'end':
                if from_ not in insertion_after:
                    insertion_after[from_] = []
                insertion_after[from_].append(ii)
            else:
                raise ValueError(f'Unknown from_anchor {from_anchor}')

        # Make insertions
        subgroup_from_is_not_inside = (subgroup_from_is_before +
                                subgroup_no_from +
                                subgroup_from_is_after)
        i_subgroup_sorted = subgroup_from_is_not_inside.copy()
        while len(insertion_before) > 0 or len(insertion_after) > 0:
            new_i_subgroup_sorted = []
            for ii in i_subgroup_sorted:
                nn = tt_group.name[ii]
                if nn in insertion_before:
                    new_i_subgroup_sorted.extend(insertion_before[nn])
                    insertion_before.pop(nn)
                new_i_subgroup_sorted.append(ii)
                if nn in insertion_after:
                    new_i_subgroup_sorted.extend(insertion_after[nn])
                    insertion_after.pop(nn)

            if len(new_i_subgroup_sorted) == len(i_subgroup_sorted):
                # No changes -> done
                raise ValueError('Could not sort elements within group; possible circular '
                                 'dependency in from_ specifications')

            i_subgroup_sorted = new_i_subgroup_sorted

        # Sort the group subtable
        tt_group = tt_group.rows[i_subgroup_sorted]

        # Append the sorted indices
        i_place_sorted.extend(list(tt_group.i_place))

        # Move to next group
        i_start_group = i_end_group

    # Sort the entire table according to i_place_sorted
    tt_sorted = tt_unsorted.rows[i_place_sorted]

    tt_sorted['s_center'] = tt_sorted['s_start'] + tt_sorted['length'] / 2
    tt_sorted['s_end'] = tt_sorted['s_start'] + tt_sorted['length']

    tt_sorted['ds_upstream'] = 0 * tt_sorted['s_start']
    tt_sorted['ds_upstream'][1:] = tt_sorted['s_start'][1:] - tt_sorted['s_end'][:-1]
    tt_sorted['ds_upstream'][0] = tt_sorted['s_start'][0]
    tt_sorted['s'] = tt_sorted['s_start']

    return tt_sorted

def _generate_element_names_with_drifts(env, tt_sorted, length=None, s_tol=1e-6):

    names_with_drifts = []
    # Create drifts
    for ii, nn in enumerate(tt_sorted.env_name):
        ds_upstream = tt_sorted['ds_upstream', ii]
        if np.abs(ds_upstream) > s_tol:
            assert ds_upstream > 0, f'Negative drift length: {ds_upstream}, upstream of {nn}'
            drift_name = env._get_drift(ds_upstream)
            names_with_drifts.append(drift_name)
        names_with_drifts.append(nn)

    if length is not None:
        length_line = tt_sorted['s_end'][-1]
        if length_line > length + s_tol:
            raise ValueError(f'Line length {length_line} is greater than the requested length {length}')
        if length_line < length - s_tol:
            drift_name = env._get_drift(length - length_line)
            names_with_drifts.append(drift_name)

    return list(map(str, names_with_drifts))

class Place:

    def __init__(self, name, at=None, from_=None, anchor=None, from_anchor=None):

        if isinstance(at, str) and '@' in at:
            at_parts = at.split('@')
            assert len(at_parts) == 2
            assert from_ is None
            assert from_anchor is None
            at = 0
            from_ = at_parts[0]
            from_anchor = at_parts[1]

        if from_ is not None:
            assert isinstance(from_, str)
            if '@' in from_:
                from_parts = from_.split('@')
                assert len(from_parts) == 2
                from_ = from_parts[0]
                from_anchor = from_parts[1]

        assert anchor in [None, 'center', 'centre', 'start', 'end']
        assert from_anchor in [None, 'center', 'centre', 'start', 'end']

        self.name = name
        self.at = at
        self.from_ = from_
        self.anchor = anchor
        self.from_anchor = from_anchor

    def __repr__(self):
        out = f'Place({self.name}'
        if self.at is not None: out += f', at={self.at}'
        if self.from_ is not None: out += f', from_={self.from_}'
        if self.anchor is not None: out += f', anchor={self.anchor}'
        if self.from_anchor is not None: out += f', from_anchor={self.from_anchor}'

        out += ')'
        return out

    def copy(self):
        out = Place('dummy')
        out.__dict__ = self.__dict__.copy()
        return out





def _argsort_s(seq, tol=10e-10):
    """Argsort, but with a tolerance; `sorted` is stable."""
    seq_indices = np.arange(len(seq))

    def comparator(i, j):
        a, b = seq[i], seq[j]
        if np.abs(a - b) < tol:
            return 0
        return -1 if a < b else 1

    return sorted(seq_indices, key=cmp_to_key(comparator))

def _resolve_lines_in_components(components, env):

    components = list(components) # Make a copy

    for ii, nn in enumerate(components):
        if (isinstance(nn, Place) and isinstance(nn.name, str)
                and nn.name in env.lines):
            nn.name = env.lines[nn.name]
        if isinstance(nn, str) and nn in env.lines:
            components[ii] = env.lines[nn]

    return components