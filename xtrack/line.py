# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import io
import math
import logging
import json
from copy import deepcopy
from pprint import pp
from pathlib import Path

import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt

from .mad_loader import MadLoader
from .beam_elements import element_classes
from . import beam_elements
from .beam_elements import Drift, BeamElement, Marker, Multipole

log = logging.getLogger(__name__)

def mk_class_namespace(extra_classes):
    try:
        import xfields as xf
        all_classes = element_classes + xf.element_classes + extra_classes + (Line,)
    except ImportError:
        all_classes = element_classes + extra_classes
        log.warning("Xfields not installed correctly")

    out = AttrDict()
    for cl in all_classes:
        out[cl.__name__] = cl
    return out


def _is_drift(element): # can be removed if length is zero
    return isinstance(element, (beam_elements.Drift,) )

def _behaves_like_drift(element):
    return hasattr(element, 'behaves_like_drift') and element.behaves_like_drift

def _is_aperture(element):
    return element.__class__.__name__.startswith('Limit')

def _is_thick(element):
    return  hasattr(element, "isthick") and element.isthick

def _allow_backtrack(element):
    return hasattr(element, 'allow_backtrack') and element.allow_backtrack

def _next_name(prefix, names, name_format='{}{}'):
    """Return an available element name by appending a number"""
    if prefix not in names: return prefix
    i = 1
    while name_format.format(prefix, i) in names:
        i += 1
    return name_format.format(prefix, i)

def _dicts_equal(dict1, dict2):
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise ValueError
    if set(dict1.keys()) != set(dict2.keys()):
        return False
    for key in dict1.keys():
        if hasattr(dict1[key], '__iter__'):
            if not hasattr(dict2[key], '__iter__'):
                return False
            elif isinstance(dict1[key], dict):
                if not isinstance(dict2[key], dict):
                    return False
                else:
                    if not _dicts_equal(dict1[key], dict2[key]):
                        return False
            elif not np.array_equal(dict1[key], dict2[key]):
                return False
        elif dict1[key] != dict2[key]:
            return False
    return True

def _apertures_equal(ap1, ap2):
    if not _is_aperture(ap1) or not _is_aperture(ap2):
        raise ValueError(f"Element {ap1} or {ap2} not an aperture!")
    if ap1.__class__ != ap2.__class__:
        return False
    ap1 = ap1.to_dict()
    ap2 = ap2.to_dict()
    return _dicts_equal(ap1, ap2)

def _lines_equal(line1, line2):
    return _dicts_equal(line1.to_dict(), line2.to_dict())


DEG2RAD = np.pi / 180.

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class Node:
    def __init__(self, s, what, *, from_=0, name=None):
        """Holds the location of an element or sequence for use with Line.from_sequence

        Args:
            s (float): Location (in m) of what relative to from_.
            what (str, BeamElement or list): Object to place here. Can be an instance of a BeamElement,
                another sequence given as list of At, or the name of a named element.
            from_ (float or str, optional): Reference location for placement, can be the s coordinate (in m)
                or the name of an element or sequence whose location is used.
            name (str, optional): Name of the element to place here. If None, a name is chosen automatically.

        """
        self.s = s
        self.from_ = from_
        self.what = what
        self.name = name

    def __repr__(self):
        return f"Node({self.s}, {self.what}, from_={self.from_}, name={self.name})"

At = Node



def flatten_sequence(nodes, elements={}, sequences={}, copy_elements=False, naming_scheme='{}{}'):
    """Flatten the sequence definition

    Named elements and nested sequences are replaced recursively.
    Node locations are made absolute.

    See Line.from_sequence for details
    """
    flat_nodes = []
    for node in nodes:
        # determine absolute position
        s = node.s
        if isinstance(node.from_, str):
            # relative to another element
            for n in flat_nodes:
                if node.from_ == n.name:
                    s += n.s
                    break
            else:
                raise ValueError(f'Unknown element name {node.from_} passed as from_')
        else:
            s += node.from_

        # find a unique name
        name = node.name or (node.what if isinstance(node.what, str) else 'element')
        name = _next_name(name, [n.name for n in flat_nodes], naming_scheme)

        # determine what to place here
        element = None
        sequence = None
        if isinstance(node.what, str):
            if node.what in elements:
                element = elements[node.what]
                if copy_elements:
                    element = element.copy()
            elif node.what in sequences:
                sequence = sequences[node.what]
            else:
                raise ValueError(f'Unknown element or sequence name {node.what}')
        elif isinstance(node.what, BeamElement):
            element = node.what
        elif hasattr(node.what, '__iter__'):
            sequence = node.what
        else:
            raise ValueError(f'Unknown element type {node.what}')

        # place elements
        if element is not None:
            flat_nodes.append(Node(s, element, name=name))

        # place nested sequences by recursion
        if sequence is not None:
            flat_nodes.append(Node(s, Marker(), name=name))
            for sub in flatten_sequence(sequence, elements=elements, sequences=sequences, copy_elements=copy_elements, naming_scheme=naming_scheme):
                sub_name = naming_scheme.format(name, sub.name)
                flat_nodes.append(Node(s + sub.s, sub.what, name=sub_name))

    return flat_nodes




class Line:

    '''
    Beam line object. `Line.element_names` contains the ordered list of beam
    elements, `Line.element_dict` is a dictionary associating to each name the
    corresponding beam element object.
    '''

    _element_dict = None

    @classmethod
    def from_dict(cls, dct, _context=None, _buffer=None, classes=()):
        '''
        Build a Line from a dictionary. `_context` and `_buffer` can be
        optionally specified.
        '''
        class_dict = mk_class_namespace(classes)

        _buffer = xo.get_a_buffer(context=_context, buffer=_buffer,size=8)

        if isinstance(dct['elements'], dict):
            elements = {}
            num_elements = len(dct['elements'].keys())
            for ii, (kk, ee) in enumerate(dct['elements'].items()):
                if ii % 100 == 0:
                    print('Loading line from dict: '
                        f'{round(ii/num_elements*100):2d}%  ',end="\r", flush=True)
                elements[kk] = _deserialize_element(ee, class_dict, _buffer)
        elif isinstance(dct['elements'], list):
            elements = []
            num_elements = len(dct['elements'])
            for ii, ee in enumerate(dct['elements']):
                if ii % 100 == 0:
                    print('Loading line from dict: '
                        f'{round(ii/num_elements*100):2d}%  ',end="\r", flush=True)
                elements.append(_deserialize_element(ee, class_dict, _buffer))
        else:
            raise ValueError('Field `elements` must be a dict or a list')

        self = cls(elements=elements, element_names=dct['element_names'])

        if 'particle_ref' in dct.keys():
            self.particle_ref = xp.Particles.from_dict(dct['particle_ref'],
                                    _context=_buffer.context)

        if '_var_manager' in dct.keys():
            self._init_var_management(dct=dct)

        print('Done loading line from dict.           ')

        return self

    @classmethod
    def from_json(cls, file, **kwargs):

        if isinstance(file, io.IOBase):
            dct = json.load(file)
        else:
            with open(file, 'r') as fid:
                dct = json.load(fid)

        if 'line' in dct.keys():
            dct_line = dct['line']
        else:
            dct_line = dct

        return cls.from_dict(dct_line, **kwargs)

    @classmethod
    def from_sequence(cls, nodes=None, length=None, elements=None, sequences=None, copy_elements=False,
                      naming_scheme='{}{}', auto_reorder=False, **kwargs):
        """Constructs a line from a sequence definition, inserting drift spaces as needed

        Args:
            nodes (list of Node): Sequence definition.
            length: Total length (in m) of line. Determines drift behind last element.
            elements: Dictionary with named elements, which can be refered to in the sequence definion by name.
            sequences: Dictionary with named sub-sequences, which can be refered to in the sequence definion by name.
            copy_elements (bool): Whether to make copies of elements or not. By default, named elements are
                re-used which is memory efficient but does not allow to change parameters individually.
            naming_scheme (str): Naming scheme to name sub-sequences. A format string accepting two names to be joined.
            auto_reorder (bool): If false (default), nodes must be defined in order of increasing s coordinate,
                otherwise an exception is thrown. If true, nodes can be defined in any order and are re-ordered
                as neseccary. Useful to place additional elements inside of sub-sequences.
            kwargs: Arguments passed to constructor of the line

        Returns:
            An instance of Line

        Examples:
            >>> from xtrack import Line, Node, Multipole
            >>> elements = {
                    'quad': Multipole(length=0.3, knl=[0, +0.50]),
                    'bend': Multipole(length=0.5, knl=[np.pi / 12], hxl=[np.pi / 12]),
                }
            >>> sequences = {
                    'arc': [Node(1, 'quad'), Node(5, 'bend')],
                }
            >>> monitor = ParticlesMonitor(...)
            >>>
            >>> line = Line.from_sequence([
                    # direct element definition
                    Node(3, xt.Multipole(...)),
                    Node(7, xt.Multipole(...), name='quad1'),
                    Node(1, xt.Multipole(...), name='bend1', from_='quad1'),
                    ...
                    # using pre-defined elements by name
                    Node(13, 'quad'),
                    Node(14, 'quad', name='quad3'),
                    Node(2, 'bend', from_='quad3', name='bend2'),
                    ....
                    # using nested sequences
                    Node(5, 'arc', name='section_1'),
                    Node(3, monitor, from_='section_1'),
                ], length = 5, elements=elements, sequences=sequences)
        """

        # flatten the sequence
        nodes = flatten_sequence(nodes, elements=elements, sequences=sequences,
            copy_elements=copy_elements, naming_scheme=naming_scheme)
        if auto_reorder:
            nodes = sorted(nodes, key=lambda node: node.s)

        # add drifts
        element_objects = []
        element_names = []
        drifts = {}
        last_s = 0
        for node in nodes:
            if node.s < last_s:
                raise ValueError(f'Negative drift space from {last_s} to {node.s}'
                    f' ({node.name}). Fix or set auto_reorder=True')
            if _is_thick(node.what):
                raise NotImplementedError(
                    f'Thick elements currently not implemented: {node.name}')

            # insert drift as needed (re-use if possible)
            if node.s > last_s:
                ds = node.s - last_s
                if ds not in drifts:
                    drifts[ds] = Drift(length=ds)
                element_objects.append(drifts[ds])
                element_names.append(_next_name('drift', element_names, naming_scheme))

            # insert element
            element_objects.append(node.what)
            element_names.append(node.name)
            last_s = node.s

        # add last drift
        if length < last_s:
            raise ValueError(f'Last element {node.name} at s={last_s} is outside length {length}')
        element_objects.append(Drift(length=length - last_s))
        element_names.append(_next_name('drift', element_names, naming_scheme))

        return cls(elements=element_objects, element_names=element_names, **kwargs)

    @classmethod
    def from_sixinput(cls, sixinput, classes=()):
        log.warning("\n"
            "WARNING: xtrack.Line.from_sixinput(sixinput) will be removed in future versions.\n"
            "Please use sixinput.generate_xtrack_line()\n")
        line = sixinput.generate_xtrack_line(classes=classes)
        return line

    @classmethod
    def from_madx_sequence(
        cls,
        sequence,
        classes=(),
        ignored_madtypes=(),
        exact_drift=False,
#        drift_threshold=1e-6, # not used anymore with expanded sequences
        deferred_expressions=False,
        install_apertures=False,
        apply_madx_errors=False,
        skip_markers=False,
        merge_drifts=False,
        merge_multipoles=False,
        expressions_for_element_types=None,
        replace_in_expr=None
    ):

        """
        Build a Line from a MAD-X sequence object.
        """

        if not (exact_drift is False):
            raise NotImplementedError("Exact drifts not implemented yet")

        class_namespace=mk_class_namespace(classes)

        loader = MadLoader(sequence,
            classes=class_namespace,
            ignore_madtypes=ignored_madtypes,
            exact_drift=False,
            enable_errors=apply_madx_errors,
            enable_apertures=install_apertures,
            enable_expressions=deferred_expressions,
            skip_markers=skip_markers,
            merge_drifts=merge_drifts,
            merge_multipoles=merge_multipoles,
            expressions_for_element_types=expressions_for_element_types,
            error_table=None,  # not implemented yet
            replace_in_expr=replace_in_expr
            )
        line=loader.make_line()
        return line

    def _init_var_management(self, dct=None):

        from collections import defaultdict
        import xdeps as xd

        _var_values = defaultdict(lambda: 0)
        _var_values.default_factory = None

        manager = xd.Manager()
        _vref = manager.ref(_var_values, 'vars')
        _fref = manager.ref(mathfunctions, 'f')
        _lref = manager.ref(self.element_dict, 'element_refs')

        self._var_management = {}
        self._var_management['data'] = {}
        self._var_management['data']['var_values'] = _var_values

        self._var_management['manager'] = manager
        self._var_management['lref'] = _lref
        self._var_management['vref'] = _vref
        self._var_management['fref'] = _fref

        if dct is not None:
            manager = self._var_management['manager']
            for kk in self._var_management['data'].keys():
                self._var_management['data'][kk].update(
                                            dct['_var_management_data'][kk])
            manager.load(dct['_var_manager'])

    @property
    def config(self):
        return self.tracker.config

    @config.setter
    def config(self, value):
        self.tracker.config = value

    @property
    def vars(self):
        if self._var_management is not None:
            return self._var_management['vref']

    @property
    def element_refs(self):
        if self._var_management is not None:
            return self._var_management['lref']

    @property
    def element_dict(self):
        return self._element_dict

    @element_dict.setter
    def element_dict(self, value):
        if self._element_dict is None:
            self._element_dict = {}
        self._element_dict.clear()
        self._element_dict.update(value)

    def __init__(self, elements=(), element_names=None, particle_ref=None):
        if isinstance(elements, dict):
            element_dict = elements
            if element_names is None:
                raise ValueError('`element_names` must be provided'
                                 ' if `elements` is a dictionary.')
        else:
            if element_names is None:
                element_names = [f"e{ii}" for ii in range(len(elements))]
            if len(element_names) > len(set(element_names)):
                log.warning("Repetition found in `element_names` -> renaming")
                old_element_names = element_names
                element_names = []
                counters = {nn: 0 for nn in old_element_names}
                for nn in old_element_names:
                    if counters[nn] > 0:
                        new_nn = nn + '_' + str(counters[nn])
                    else:
                        new_nn = nn
                    counters[nn] += 1
                    element_names.append(new_nn)

            assert len(element_names) == len(elements), (
                "`elements` and `element_names` should have the same length"
            )
            element_dict = dict(zip(element_names, elements))

        self.element_dict = element_dict.copy()  # avoid modifications if user provided
        self.element_names = list(element_names).copy()

        self.particle_ref = particle_ref

        self._var_management = None
        self._needs_rng = False
        self.tracker = None

    def build_tracker(self, **kwargs):
        "Build a tracker associated for the line."
        assert self.tracker is None, 'The line already has an associated tracker'
        import xtrack as xt # avoid circular import
        self.tracker = xt.Tracker(line=self, **kwargs)
        return self.tracker

    @property
    def elements(self):
        return tuple([self.element_dict[nn] for nn in self.element_names])

    def __getitem__(self, ii):
        if isinstance(ii, str):
            if ii not in self.element_names:
                raise IndexError(f'No installed element with name {ii}')
            return self.element_dict.__getitem__(ii)
        else:
            names = self.element_names.__getitem__(ii)
            if isinstance(names, str):
                return self.element_dict.__getitem__(names)
            else:
                return [self.element_dict[nn] for nn in names]

    def filter_elements(self, mask=None, exclude_types_starting_with=None,
                        _make_tracker=True):
        """
        Return a new line with only the elements satisfying a given condition.
        Other elements are replaced with Drifts.

        Parameters
        ----------
        mask: list of bool
            A list of booleans with the same length as the line.
            If True, the element is kept, otherwise it is replaced with a Drift.
        exclude_types_starting_with: str
            If not None, all elements whose type starts with the given string
            are replaced with Drifts.
        """

        if _make_tracker and self.tracker is not None:
            new_tracker = self.tracker.filter_elements(mask=mask,
                        exclude_types_starting_with=exclude_types_starting_with)
            new_line = new_tracker.line
            new_line.tracker = new_tracker
            return new_line

        if mask is None:
            assert exclude_types_starting_with is not None

        if exclude_types_starting_with is not None:
            assert mask is None
            mask = [not(ee.__class__.__name__.startswith(exclude_types_starting_with))
                    for ee in self.elements]

        new_elements = []
        assert len(mask) == len(self.elements)
        for ff, ee in zip(mask, self.elements):
            if ff:
                new_elements.append(ee)
            else:
                if _is_thick(ee) and not _is_drift(ee):
                    new_elements.append(Drift(length=ee.length))
                else:
                    new_elements.append(Drift(length=0))

        new_line = self.__class__(elements=new_elements,
                              element_names=self.element_names)
        if self.particle_ref is not None:
            new_line.particle_ref = self.particle_ref.copy()

        return new_line

    def cycle(self, index_first_element=None, name_first_element=None,
              _make_tracker=True):

        """
        Cycle the line to start from a given element.

        Parameters
        ----------
        index_first_element: int
            Index of the element to start from
        name_first_element: str
            Name of the element to start from
        """

        if _make_tracker and self.tracker is not None:
            new_tracker = self.tracker.cycle(
                index_first_element=index_first_element,
                name_first_element=name_first_element,
            )
            new_line = new_tracker.line
            new_line.tracker = new_tracker
            return new_line

        if ((index_first_element is not None and name_first_element is not None)
               or (index_first_element is None and name_first_element is None)):
             raise ValueError(
                "Please provide either `index_first_element` or `name_first_element`.")

        if type(index_first_element) is str:
            name_first_element = index_first_element
            index_first_element = None

        if name_first_element is not None:
            assert self.element_names.count(name_first_element) == 1, (
                f"name_first_element={name_first_element} occurs more than once!"
            )
            index_first_element = self.element_names.index(name_first_element)

        new_element_names = (list(self.element_names[index_first_element:])
                             + list(self.element_names[:index_first_element]))

        return self.__class__(
            elements=self.element_dict,
            element_names=new_element_names,
            particle_ref=self.particle_ref,
        )

    def _freeze(self):
        self.element_names = tuple(self.element_names)

    def unfreeze(self):
        """
        Unfreeze the line. This is useful if you want to modify the line
        after it has been frozen (most likely by calling `build_tracker`).
        """
        self.element_names = list(self.element_names)
        if hasattr(self, 'tracker') and self.tracker is not None:
            self.tracker._invalidate()
            self.tracker = None

    def _frozen_check(self):
        if isinstance(self.element_names, tuple):
            raise ValueError(
                'This action is not allowed as the line is frozen!')

    def __getattr__(self, attr):
        # If not in self look in self.tracker (if not None)
        if self.tracker is not None and attr in dir(self.tracker):
            return getattr(self.tracker, attr)
        elif attr in dir(xt.Tracker):
            # If in Tracker class, ask the used to build the tracker
            raise AttributeError(
                'The tracker is not built. Please call Line.build_tracker()')
        else:
            raise AttributeError(
                f'Line object has no attribute `{attr}`')

    def __dir__(self):
        return list(set(object.__dir__(self) + dir(self.tracker)))

    def __len__(self):
        return len(self.element_names)

    def copy(self, _context=None, _buffer=None):
        '''
        Return a copy of the line.

        Parameters
        ----------
        _context: xobjects.Context
            xobjects context to be used for the copy
        _buffer: xobjects.Buffer
            xobjects buffer to be used for the copy
        '''

        elements = {nn: ee.copy(_context=_context, _buffer=_buffer)
                                    for nn, ee in self.element_dict.items()}
        element_names = [nn for nn in self.element_names]

        out = self.__class__(elements=elements, element_names=element_names)

        if self.particle_ref is not None:
            out.particle_ref = self.particle_ref.copy(
                                        _context=_context, _buffer=_buffer)

        if self._var_management is not None:
            out._init_var_management(dct=self._var_management_to_dict())

        return out

    def _var_management_to_dict(self):
        out = {}
        out['_var_management_data'] = deepcopy(self._var_management['data'])
        out['_var_manager'] = self._var_management['manager'].dump()
        return out

    def to_dict(self, include_var_management=True):

        '''Return a dictionary representation of the line.

        Parameters
        ----------
        include_var_management : bool, optional
            If True (default) the dictionary will contain the information
            needed to restore the line with deferred expressions.
        '''

        out = {}
        out["elements"] = {k: el.to_dict() for k, el in self.element_dict.items()}
        out["element_names"] = self.element_names[:]
        if self.particle_ref is not None:
            out['particle_ref'] = self.particle_ref.to_dict()
        if self._var_management is not None and include_var_management:
            if hasattr(self, '_in_multiline') and self._in_multiline:
                raise ValueError('The line is part ot a MultiLine object. '
                    'To save without expressions please use '
                    '`line.to_dict(include_var_management=False)`.\n'
                    'To save also the deferred expressions please save the '
                    'entire multiline.\n ')

            out.update(self._var_management_to_dict())
        return out

    def to_json(self, file, **kwargs):
        '''Save the line to a json file.

        Parameters
        ----------
        file: str or file-like object
            The file to save to. If a string is provided, a file is opened and
            closed. If a file-like object is provided, it is used directly.
        **kwargs: dict
            Additional keyword arguments are passed to the `Line.to_dict` method.
        '''

        if isinstance(file, io.IOBase):
            json.dump(self.to_dict(**kwargs), file, cls=xo.JEncoder)
        else:
            with open(file, 'w') as fid:
                json.dump(self.to_dict(**kwargs), fid, cls=xo.JEncoder)

    def to_pandas(self):
        '''
        Return a pandas DataFrame with the elements of the line.
        '''

        elements = self.elements
        s_elements = np.array(self.get_s_elements())
        element_types = list(map(lambda e: e.__class__.__name__, elements))
        isthick = np.array(list(map(_is_thick, elements)))

        import pandas as pd

        elements_df = pd.DataFrame({
            'element_type': element_types,
            's': s_elements,
            'name': self.element_names,
            'isthick': isthick,
            'element': elements
        })
        return elements_df

    def insert_element(self, index=None, element=None, name=None, at_s=None,
                       s_tol=1e-6):

        '''Insert an element in the line.

        Parameters
        ----------
        index: int, optional
            Index of the element in the line. If `index` is provided, `at_s`
            must be None.
        element: xline.Element, optional
            Element to be inserted. If `element` is provided, `name` must be
            provided.
        name: str
            Name of the element. If `name` is provided, `element` must be
            provided.
        at_s: float, optional
            Position of the element in the line. If `at_s` is provided, `index`
            must be None.
        s_tol: float, optional
            Tolerance for the position of the element in the line.
        '''

        if isinstance(index, str):
            assert index in self.element_names
            index = self.element_names.index(index)

        assert name is not None
        if element is None:
            assert name in self.element_names
            element  = self.element_dict[name]

        self._frozen_check()

        assert ((index is not None and at_s is None) or
                (index is None and at_s is not None)), (
                    "Either `index` or `at_s` must be provided"
                )

        if at_s is not None:
            s_vect_upstream = np.array(self.get_s_position(mode='upstream'))

            if not _is_thick(element) or np.abs(element.length)==0:
                i_closest = np.argmin(np.abs(s_vect_upstream - at_s))
                if np.abs(s_vect_upstream[i_closest] - at_s) < s_tol:
                    return self.insert_element(index=i_closest,
                                            element=element, name=name)

            s_vect_downstream = np.array(self.get_s_position(mode='downstream'))

            s_start_ele = at_s
            i_first_drift_to_cut = np.where(s_vect_downstream > s_start_ele)[0][0]

            # Shortcut for thin element without drift splitting
            if (not _is_thick(element)
                and np.abs(s_vect_upstream[i_first_drift_to_cut]-at_s)<1e-10):
                    return self.insert_element(index=i_first_drift_to_cut,
                                              element=element, name=name)

            if _is_thick(element) and np.abs(element.length)>0:
                s_end_ele = at_s + element.length
            else:
                s_end_ele = s_start_ele

            i_last_drift_to_cut = np.where(s_vect_upstream < s_end_ele)[0][-1]
            if _is_thick(element) and element.length > 0:
                assert i_first_drift_to_cut <= i_last_drift_to_cut
            name_first_drift_to_cut = self.element_names[i_first_drift_to_cut]
            name_last_drift_to_cut = self.element_names[i_last_drift_to_cut]
            first_drift_to_cut = self.element_dict[name_first_drift_to_cut]
            last_drift_to_cut = self.element_dict[name_last_drift_to_cut]

            assert _is_drift(first_drift_to_cut)
            assert _is_drift(last_drift_to_cut)

            for ii in range(i_first_drift_to_cut, i_last_drift_to_cut+1):
                e_to_replace = self.element_dict[self.element_names[ii]]
                if (not _is_drift(e_to_replace) and
                    not isinstance(e_to_replace, Marker) and
                    not _is_aperture(e_to_replace)):
                    raise ValueError('Cannot replace active element '
                                        f'{self.element_names[ii]}')

            l_left_part = s_start_ele - s_vect_upstream[i_first_drift_to_cut]
            l_right_part = s_vect_downstream[i_last_drift_to_cut] - s_end_ele
            assert l_left_part >= 0
            assert l_right_part >= 0
            name_left = name_first_drift_to_cut + '_part0'
            name_right = name_last_drift_to_cut + '_part1'

            self.element_names[i_first_drift_to_cut:i_last_drift_to_cut] = []
            i_insert = i_first_drift_to_cut

            drift_base = self.element_dict[self.element_names[i_insert]]
            drift_left = drift_base.copy()
            drift_left.length = l_left_part
            drift_right = drift_base.copy()
            drift_right.length = l_right_part

            # Insert
            assert name_left not in self.element_names
            assert name_right not in self.element_names

            names_to_insert = []

            if drift_left.length > 0:
                names_to_insert.append(name_left)
                self.element_dict[name_left] = drift_left
            names_to_insert.append(name)
            self.element_dict[name] = element
            if drift_right.length > 0:
                names_to_insert.append(name_right)
                self.element_dict[name_right] = drift_right

            self.element_names[i_insert] = names_to_insert[-1]
            if len(names_to_insert) > 1:
                for nn in names_to_insert[:-1][::-1]:
                    self.element_names.insert(i_insert, nn)

        else:
            if _is_thick(element) and np.abs(element.length)>0:
                raise NotImplementedError('use `at_s` to insert thick elements')
            assert name not in self.element_dict.keys()
            self.element_dict[name] = element
            self.element_names.insert(index, name)

        return self

    def append_element(self, element, name):

        '''Append element to the end of the lattice

        Parameters
        ----------
        element : object
            Element to append
        name : str
            Name of the element to append
        '''

        self._frozen_check()
        assert name not in self.element_dict.keys()
        self.element_dict[name] = element
        self.element_names.append(name)
        return self

    def get_length(self):
        ll = 0
        for ee in self.elements:
            if _is_thick(ee):
                ll += ee.length

        return ll

    def get_s_elements(self, mode="upstream"):
        '''Get s position for all elements'''
        return self.get_s_position(mode=mode)

    def get_s_position(self, at_elements=None, mode="upstream"):

        '''Get s position for given elements

        Parameters
        ----------
        at_elements : str or list of str
            Name of the element(s) to get s position for (default: all elements)
        mode : str
            "upstream" or "downstream" (default: "upstream")

        Returns
        -------
        s : float or list of float
            s position for given element(s)
        '''

        assert mode in ["upstream", "downstream"]
        s_prev = 0
        s = []
        for ee in self.elements:
            if mode == "upstream":
                s.append(s_prev)
            if _is_thick(ee):
                s_prev += ee.length
            if mode == "downstream":
                s.append(s_prev)

        if at_elements is not None:
            if np.isscalar(at_elements):
                if isinstance(at_elements, str):
                    assert at_elements in self.element_names
                    idx = self.element_names.index(at_elements)
                else:
                    idx = at_elements
                return s[idx]
            else:
                assert all([nn in self.element_names for nn in at_elements])
                return [s[self.element_names.index(nn)] for nn in at_elements]
        else:
            return s

    def remove_markers(self, inplace=True, keep=None):
        '''
        Remove markers from the line

        Parameters
        ----------
        inplace : bool
            If True, remove markers from the line (default: True)
        keep : str or list of str
            Name of the markers to keep (default: None)
        '''

        if self._var_management is not None:
            raise NotImplementedError('`remove_markers` not'
                                      ' available when deferred expressions are'
                                      ' used')

        self._frozen_check()

        if keep is None:
            keep = []
        elif isinstance(keep, str):
            keep = [keep]

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if isinstance(ee, Marker) and nn not in keep:
                continue
            newline.append_element(ee, nn)

        if inplace:
            self.element_names = newline.element_names
            self.element_dict = newline.element_dict
            return self
        else:
            return newline

    def remove_inactive_multipoles(self, inplace=True, keep=None):
        '''
        Remove inactive multipoles from the line
        '''

        if self._var_management is not None:
            raise NotImplementedError('`remove_inactive_multipoles` not'
                                      ' available when deferred expressions are'
                                      ' used')

        self._frozen_check()

        if keep is None:
            keep = []
        elif isinstance(keep, str):
            keep = [keep]

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if isinstance(ee, Multipole) and nn not in keep:
                ctx2np = ee._context.nparray_from_context_array
                aux = ([ee.hxl, ee.hyl]
                        + list(ctx2np(ee.knl)) + list(ctx2np(ee.ksl)))
                if np.sum(np.abs(np.array(aux))) == 0.0:
                    continue
            newline.append_element(ee, nn)

        if inplace:
            self.element_names = newline.element_names
            self.element_dict = newline.element_dict
            return self
        else:
            return newline

    def remove_zero_length_drifts(self, inplace=True, keep=None):
        '''
        Remove zero-length drifts from the line
        '''

        if self._var_management is not None:
            raise NotImplementedError('`remove_zero_length_drifts` not'
                                      ' available when deferred expressions are'
                                      ' used')

        self._frozen_check()

        if keep is None:
            keep = []
        elif isinstance(keep, str):
            keep = [keep]

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if _is_drift(ee) and nn not in keep:
                if ee.length == 0.0:
                    continue
            newline.append_element(ee, nn)

        if inplace:
            self.element_names = newline.element_names
            self.element_dict = newline.element_dict
            return self
        else:
            return newline

    def merge_consecutive_drifts(self, inplace=True, keep=None):
        '''
        Merge consecutive drifts into one drift
        '''

        if self._var_management is not None:
            raise NotImplementedError('`merge_consecutive_drifts` not'
                                      ' available when deferred expressions are'
                                      ' used')

        self._frozen_check()

        if keep is None:
            keep = []
        elif isinstance(keep, str):
            keep = [keep]

        newline = Line(elements=[], element_names=[])

        for ii, (ee, nn) in enumerate(zip(self.elements, self.element_names)):
            if ii == 0:
                newline.append_element(ee.copy(), nn)
                continue

            this_ee = ee if inplace else ee.copy()
            if _is_drift(ee) and not nn in keep:
                prev_nn = newline.element_names[-1]
                prev_ee = newline.element_dict[prev_nn]
                if _is_drift(prev_ee) and not prev_nn in keep:
                    prev_ee.length += ee.length
                else:
                    newline.append_element(this_ee, nn)
            else:
                newline.append_element(this_ee, nn)

        if inplace:
            self.element_names = newline.element_names
            self.element_dict = newline.element_dict
            return self
        else:
            return newline

    # For every occurence of three or more apertures that are the same,
    # only separated by Drifts or Markers, this script removes the
    # middle apertures
    def remove_redundant_apertures(self, inplace=True, keep=None,
                                  drifts_that_need_aperture=[]):
        '''
        Merge consecutive aperture checks by deleting the middle ones
        '''

        # TODO: this probably actually works, but better be safe than sorry
        if self._var_management is not None:
            raise NotImplementedError('`remove_redundant_apertures` not'
                                      ' available when deferred expressions are'
                                      ' used')

        self._frozen_check()

        if keep is None:
            keep = []
        elif isinstance(keep, str):
            keep = [keep]

        aper_to_remove = []
        # current aperture in loop
        aper_0  = None
        # previous aperture in loop (-1)
        aper_m1 = None
        # aperture before previous in loop (-2)
        aper_m2 = None

        for ee, nn in zip(self.elements, self.element_names):
            if ee.__class__.__name__.startswith('Limit'):
            # We encountered a new aperture, shift all previous
                aper_m2 = aper_m1
                aper_m1 = aper_0
                aper_0  = nn
            elif (not isinstance(ee, (Drift, Marker)) 
            or nn in drifts_that_need_aperture):
            # We are in an active element: all previous apertures
            # should be kept in the line
                aper_0  = None
                aper_m1 = None
                aper_m2 = None
            if (aper_m2 is not None
                and _apertures_equal(self.element_dict[aper_0], self.element_dict[aper_m1])
                and _apertures_equal(self.element_dict[aper_m1], self.element_dict[aper_m2])
                ):
                # We found three consecutive apertures (with only Drifts and Markers
                # in between) that are the same, hence the middle one can be removed
                if aper_m1 not in keep:
                    aper_to_remove = [*aper_to_remove, aper_m1]
                    # Middle aperture removed, so the -2 shifts to the -1 position
                    aper_m1 = aper_m2
                    aper_m2 = None

        if inplace:
            newline = self
        else:
            newline = self.copy()

        for name in aper_to_remove:
            newline.element_names.remove(name)

        return newline

    def merge_consecutive_multipoles(self, inplace=True, keep=None):
        '''
        Merge consecutive multipoles into one multipole
        '''

        if self._var_management is not None:
            raise NotImplementedError('`merge_consecutive_multipoles` not'
                                      ' available when deferred expressions are'
                                      ' used')

        self._frozen_check()

        if keep is None:
            keep = []
        elif isinstance(keep, str):
            keep = [keep]

        newline = Line(elements=[], element_names=[])

        for ee, nn in zip(self.elements, self.element_names):
            if len(newline.element_names) == 0:
                newline.append_element(ee, nn)
                continue

            if isinstance(ee, Multipole) and nn not in keep:
                prev_nn = newline.element_names[-1]
                prev_ee = newline.element_dict[prev_nn]
                if (isinstance(prev_ee, Multipole)
                    and prev_ee.hxl==ee.hxl==0 and prev_ee.hyl==ee.hyl==0
                    and prev_nn not in keep
                    ):

                    oo=max(len(prev_ee.knl), len(prev_ee.ksl),
                           len(ee.knl), len(ee.ksl))
                    knl=np.zeros(oo,dtype=float)
                    ksl=np.zeros(oo,dtype=float)
                    for ii,kk in enumerate(prev_ee._xobject.knl):
                        knl[ii]+=kk
                    for ii,kk in enumerate(ee._xobject.knl):
                        knl[ii]+=kk
                    for ii,kk in enumerate(prev_ee._xobject.ksl):
                        ksl[ii]+=kk
                    for ii,kk in enumerate(ee._xobject.ksl):
                        ksl[ii]+=kk
                    newee = Multipole(
                            knl=knl, ksl=ksl, hxl=prev_ee.hxl, hyl=prev_ee.hyl,
                            length=prev_ee.length,
                            radiation_flag=prev_ee.radiation_flag)
                    prev_nn += ('_' + nn)
                    newline.element_dict[prev_nn] = newee
                    newline.element_names[-1] = prev_nn
                else:
                    newline.append_element(ee, nn)
            else:
                newline.append_element(ee, nn)

        if inplace:
            self.element_names = newline.element_names
            self.element_dict = newline.element_dict
            return self
        else:
            return newline

    def use_simple_quadrupoles(self):
        '''
        Replace multipoles having only the normal quadrupolar component
        with quadrupole elements. The element is not replaced when synchrotron
        radiation is active.
        '''
        self._frozen_check()

        for name, element in self.element_dict.items():
            if _is_simple_quadrupole(element):
                fast_quad = beam_elements.SimpleThinQuadrupole(
                    knl=element.knl,
                    _context=element._context,
                )
                self.element_dict[name] = fast_quad

    def use_simple_bends(self):
        '''
        Replace multipoles having only the horizontal dipolar component
        with dipole elements. The element is not replaced when synchrotron
        radiation is active.
        '''
        self._frozen_check()

        for name, element in self.element_dict.items():
            if _is_simple_dipole(element):
                fast_di = beam_elements.SimpleThinBend(
                    knl=element.knl,
                    hxl=element.hxl,
                    length=element.length,
                    _context=element._context,
                )
                self.element_dict[name] = fast_di

    def get_elements_of_type(self, types):
        if not hasattr(types, "__iter__"):
            type_list = [types]
        else:
            type_list = types

        names = []
        elements = []
        for ee, nn in zip(self.elements, self.element_names):
            for tt in type_list:
                if isinstance(ee, tt):
                    names.append(nn)
                    elements.append(ee)

        return elements, names

    def check_aperture(self, needs_aperture=[]):

        '''Check that all active elements have an associated aperture.'''

        elements_df = self.to_pandas()

        elements_df['is_aperture'] = elements_df.name.map(
                                            lambda nn: _is_aperture(self.element_dict[nn]))
        elements_df['i_aperture_upstream'] = np.nan
        elements_df['s_aperture_upstream'] = np.nan
        elements_df['i_aperture_downstream'] = np.nan
        elements_df['s_aperture_downstream'] = np.nan
        num_elements = len(self.element_names)

        # Elements that don't need aperture
        dont_need_aperture = {name: False for name in elements_df['name']}
        for name in elements_df['name']:
            ee = self.element_dict[name]
            if _allow_backtrack(ee) and not name in needs_aperture:
                dont_need_aperture[name] = True

            # Correct isthick for elements that need aperture but have zero length.
            # Use-case example: Before collimators are installed as EverestCollimator
            # (or any BaseCollimator element), they are just Markers or Drifts. We
            # want to enforce that they have an aperture when loading the line (when
            # they are still Drifts), so their names are added to 'needs_aperture'.
            # However, it is enough for them to have an upstream aperture as they are
            # at this stage just Markers (and xcoll takes care of providing the down-
            # stream aperture), so we mark them as thin.
            if name in needs_aperture and hasattr(ee, 'length') and ee.length == 0:
                elements_df.loc[elements_df['name']==name, 'isthick'] = False

        i_prev_aperture = elements_df[elements_df['is_aperture']].index[0]
        i_next_aperture = 0

        for iee in range(i_prev_aperture, num_elements):

            if iee % 100 == 0:
                print(
                    f'Checking aperture: {round(iee/num_elements*100):2d}%  ',
                    end="\r", flush=True)

            if dont_need_aperture[elements_df.loc[iee, 'name']]:
                continue

            if elements_df.loc[iee, 'is_aperture']:
                i_prev_aperture = iee
                continue

            if i_next_aperture < iee:
                for ii in range(iee, num_elements):
                    if elements_df.loc[ii, 'is_aperture']:
                        i_next_aperture = ii
                        break

            elements_df.at[iee, 'i_aperture_upstream'] = i_prev_aperture
            elements_df.at[iee, 'i_aperture_downstream'] = i_next_aperture

            elements_df.at[iee, 's_aperture_upstream'] = elements_df.loc[i_prev_aperture, 's']
            elements_df.at[iee, 's_aperture_downstream'] = elements_df.loc[i_next_aperture, 's']

        # Check for elements missing aperture upstream
        elements_df['misses_aperture_upstream'] = ((elements_df['s_aperture_upstream'] != elements_df['s'])
            & ~(np.isnan(elements_df['i_aperture_upstream'])))

        # Check for elements missing aperture downstream
        s_downstream = elements_df.s.copy()
        df_thick_to_check = elements_df[elements_df['isthick'] & ~(elements_df.i_aperture_upstream.isna())].copy()
        s_downstream.loc[df_thick_to_check.index] += np.array([ee.length for ee in df_thick_to_check.element])
        elements_df['misses_aperture_downstream'] = (
            (np.abs(elements_df['s_aperture_downstream'] - s_downstream) > 1e-6)
            & ~(np.isnan(elements_df['i_aperture_upstream'])))

        # Flag problems
        elements_df['has_aperture_problem'] = (
            elements_df['misses_aperture_upstream'] | (
                elements_df['isthick'] & elements_df['misses_aperture_downstream']))

        print('Done checking aperture.           ')

        # Identify issues with apertures associate with thin elements
        df_thin_missing_aper = elements_df[elements_df['misses_aperture_upstream'] & ~elements_df['isthick']]
        print(f'{len(df_thin_missing_aper)} thin elements miss associated aperture (upstream):')
        pp(list(df_thin_missing_aper.name))

        # Identify issues with apertures associate with thick elements
        df_thick_missing_aper = elements_df[
            (elements_df['misses_aperture_upstream'] | elements_df['misses_aperture_downstream'])
            & elements_df['isthick']]
        print(f'{len(df_thick_missing_aper)} thick elements miss associated aperture (upstream or downstream):')
        pp(list(df_thick_missing_aper.name))

        return elements_df


mathfunctions = type('math', (), {})
mathfunctions.sqrt = math.sqrt
mathfunctions.log = math.log
mathfunctions.log10 = math.log10
mathfunctions.exp = math.exp
mathfunctions.sin = math.sin
mathfunctions.cos = math.cos
mathfunctions.tan = math.tan
mathfunctions.asin = math.asin
mathfunctions.acos = math.acos
mathfunctions.atan = math.atan
mathfunctions.sinh = math.sinh
mathfunctions.cosh = math.cosh
mathfunctions.tanh = math.tanh
mathfunctions.sinc = np.sinc
mathfunctions.abs = math.fabs
mathfunctions.erf = math.erf
mathfunctions.erfc = math.erfc
mathfunctions.floor = math.floor
mathfunctions.ceil = math.ceil
mathfunctions.round = np.round
mathfunctions.frac = lambda x: (x % 1)


def _deserialize_element(el, class_dict, _buffer):
    eldct = el.copy()
    eltype = class_dict[eldct.pop('__class__')]
    if hasattr(eltype, '_XoStruct'):
        return eltype.from_dict(eldct, _buffer=_buffer)
    else:
        return eltype.from_dict(eldct)


def _is_simple_quadrupole(el):
    if not isinstance(el, Multipole):
        return False
    return (el.radiation_flag == 0 and
            el.order == 1 and
            el.knl[0] == 0 and
            el.length == 0 and
            not any(el.ksl) and
            not el.hxl and
            not el.hyl)


def _is_simple_dipole(el):
    if not isinstance(el, Multipole):
        return False
    return (el.radiation_flag == 0 and el.order == 0
            and not any(el.ksl) and not el.hyl)
