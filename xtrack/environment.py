from collections import Counter, UserDict
from collections.abc import Iterable
from functools import cmp_to_key
from typing import Literal
from weakref import WeakSet
from copy import deepcopy
import re
import importlib.util
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

import xobjects as xo
import xdeps as xd
import xtrack as xt
from xdeps.refs import is_ref
from .multiline_legacy.multiline_legacy import MultilineLegacy
from .progress_indicator import progress

ReferType = Literal['start', 'center', 'centre', 'end']

DEFAULT_REF_STRENGTH_NAME = {
    'Bend': 'k0',
    'Quadrupole': 'k1',
    'Sextupole': 'k2',
    'Octupole': 'k3',
}

def _flatten_components(components, refer: ReferType = 'center'):
    if refer not in ['start', 'center', 'centre', 'end']:
        raise ValueError(
            f'Allowed values for refer are "start", "center" and "end". Got "{refer}".'
        )

    flatt_components = []
    for nn in components:
        if isinstance(nn, Place) and isinstance(nn.name, xt.Line):

            anchor = nn.anchor
            if anchor is None:
                anchor = refer or 'center'

            line = nn.name
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
        elif isinstance(nn, xt.Line):
            flatt_components += nn.element_names
        elif isinstance(nn, Iterable) and not isinstance(nn, str):
            flatt_components += _flatten_components(nn, refer=refer)
        else:
            flatt_components.append(nn)

    return flatt_components

class Environment:
    def __init__(self, element_dict=None, particle_ref=None, _var_management=None,
                 lines=None):

        '''
        Create an environment.

        Parameters
        ----------
        element_dict : dict, optional
            Dictionary with the elements of the environment.
        particle_ref : ParticleRef, optional
            Reference particle.
        lines : dict, optional
            Dictionary with the lines of the environment.

        Short description of main attributes of the Environment class:
         - Environment[...]: accesses values of variables, elements and lines.
         - ref[...]: provides reference objects to variables and elements.
         - vars[...]: container with all variables, returns variable objects.
         - elements[...]: container with all elements.
         - lines[...]: container with all lines.

        Short description of main methods of the Environment class:
         - info(...): displays information about a variable or element.
         - get(...): returns variable value or element.
         - set(...): sets variable or element properties.
         - eval(...): evaluates an expression provided as a string (returns a value).
         - new_expr(...): creates a new expression from a string.
         - get_expr(...): returns the expression for a variable.
         - new(...): creates a new element.
         - new_line(...): creates a new line.
         - new_builder(...): creates a new builder.
         - place(...): creates a place object, which can be user in new_line(...)
           or by a Builder object.

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
                env.new('mq1_clone', 'mq1', k1='2*a'),   # Clone 'mq1' with a different k1
                env.place('mq2', at=20.0, from_='mymark'),  # Place 'mq2' at s=20
                ])

        '''
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

            # Identify common elements
            counts = Counter()
            for ll in lines.values():
                # Extract names of all elements and parents
                elems_and_parents = set(ll.element_names)
                for nn in ll.element_names:
                    if hasattr(ll.element_dict[nn], 'parent_name'):
                        elems_and_parents.add(ll.element_dict[nn].parent_name)
                # Count if it is not a marker or a drift, which will be handled by
                # `import_line`
                for nn in elems_and_parents:
                    if (not (isinstance(ll.element_dict[nn], (xt.Marker))) and
                        not bool(re.match(r'^drift_\d+$', nn))):
                        counts[nn] += 1
            common_elements = [nn for nn, cc in counts.items() if cc>1]

            for nn, ll in lines.items():
                rename_elements = {el: el+'/'+nn for el in common_elements}
                self.import_line(line=ll, suffix_for_common_elements='/'+nn,
                    line_name=nn, rename_elements=rename_elements)
                self.lines[nn]._renamed_elements = rename_elements

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
            mirror=False, force=False, import_from=None, **kwargs):

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

        if name in self.element_dict and not force:
            raise ValueError(f'Element `{name}` already exists')

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
        prototype = None
        if isinstance(parent, str):
            if parent in self.element_dict:
                # Clone an existing element
                prototype = parent
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

        if 'rbend' in kwargs:
            raise ValueError('Use the `RBend` element directly, instead of '
                             'specifying the `rbend` flag.')

        if 'rbarc' in kwargs:
            raise ValueError('Use the `RBend` element with the `length` or '
                             '`length_straight` parameter set accordingly, '
                             'instead of specifying the `rbarc` flag.')

        ref_kwargs, value_kwargs = _parse_kwargs(parent, kwargs, _eval)

        if needs_instantiation: # Parent is a class and not another element
            self.element_dict[name] = parent(**value_kwargs)

        _set_kwargs(name=name, ref_kwargs=ref_kwargs, value_kwargs=value_kwargs,
                    element_dict=self.element_dict, element_refs=self.element_refs)

        if extra is not None:
            assert isinstance(extra, dict)
            self.element_dict[name].extra = extra

        self.element_dict[name].prototype = prototype

        return name

    def _init_var_management(self, dct=None):

        self._var_management = xt.line._make_var_management(element_dict=self.element_dict,
                                               dct=dct)
        self._line_vars = xt.line.LineVars(self)


    def new_line(self, components=None, name=None, refer: ReferType = 'center', length=None):

        '''
        Create a new line.

        Parameters
        ----------
        components : list, optional
            List of components to be added to the line. It can include strings,
            place objects, and lines.
        name : str, optional
            Name of the new line.
        refer : str, optional
            Specifies which part of the component the ``at`` position will refer
            to. Allowed values are ``start``, ``center`` (default; also allowed
            is ``centre```), and ``end``.
        length : float | str, optional
            Length of the line to be built by the builder. Can be an expression.
            If not specified, the length will be the minimum length that can
            fit all the components.

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

        out = xt.Line(env=self)

        if components is None:
            components = []

        if isinstance(length, str):
            length = self.eval(length)

        components = _resolve_lines_in_components(components, self)
        flattened_components = _flatten_components(components, refer=refer)

        if np.array([isinstance(ss, str) for ss in flattened_components]).all():
            # All elements provided by name
            element_names = [str(ss) for ss in flattened_components]
        else:
            seq_all_places = _all_places(flattened_components)
            tab_unsorted = _resolve_s_positions(seq_all_places, self, refer=refer)
            tab_sorted = _sort_places(tab_unsorted)
            element_names = _generate_element_names_with_drifts(self, tab_sorted,
                                                                length=length)

        out.element_names = element_names
        out._name = name
        out.builder = Builder(env=self, components=components, length=length,
                              name=name, refer=refer)

        # Temporary solution to keep consistency in multiline
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            out._var_management = None
            out._in_multiline = self._in_multiline
            out._name_in_multiline = self._name_in_multiline

        self._lines_weakrefs.add(out) # Weak references
        if name is not None:
            self.lines[name] = out

        return out

    def place(self, name, obj=None, at=None, from_=None, anchor=None, from_anchor=None):
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

        if obj is not None:
            assert not isinstance(name, xt.Line)
            assert name not in self.element_dict
            self.element_dict[name] = obj

        # if list of strings, create a line
        if (isinstance(name, Iterable) and not isinstance(name, str)
            and all(isinstance(item, str) for item in name)):
            name = self.new_line(components=name)

        return Place(name, at=at, from_=from_, anchor=anchor, from_anchor=from_anchor)

    def new_builder(self, components=None, name=None, refer: ReferType = 'center',
                    length=None):
        '''
        Create a new builder.

        Parameters
        ----------
        components : list, optional
            List of components to be added to the builder. It can include strings,
            place objects, and lines.
        name : str, optional
            Name of the line that will be built by the builder.
        refer : str, optional
            Specifies which part of the component the ``at`` position will refer
            to. Allowed values are ``start``, ``center`` (default; also allowed
            is ``centre```), and ``end``.
        length : float | str, optional
            Length of the line to be built by the builder. Can be an expression.
            If not specified, the length will be the minimum length that can
            fit all the components.

        Returns
        -------
        Builder
            The new builder.
        '''

        return Builder(env=self, components=components, name=name, refer=refer,
                       length=length)

    def call(self, filename):
        '''
        Call a file with xtrack commands.

        Parameters
        ----------
        filename : str
            Name of the file to be called.
        '''
        import xtrack
        xtrack._passed_env = self
        try:
            load_module_from_path(Path(filename))
        except Exception as ee:
            xtrack._passed_env = None
            raise ee
        xtrack._passed_env = None

    def copy_element_from(self, name, source, new_name=None):
        return xt.Line.copy_element_from(self, name, source, new_name)


    def _import_element(self, line, name, rename_elements, suffix_for_common_elements,
                        already_imported):
        new_name = name
        if name in rename_elements:
            new_name = rename_elements[name]
        elif (bool(re.match(r'^drift_\d+$', name))
            and line.ref[name].length._expr is None):
            new_name = self._get_a_drift_name()
        elif (name in self.element_dict and
                not (isinstance(line[name], xt.Marker) and
                    isinstance(self.element_dict.get(name), xt.Marker))):
            new_name += suffix_for_common_elements

        self.copy_element_from(name, line, new_name=new_name)
        already_imported[name] = new_name
        if hasattr(line.element_dict[name], 'parent_name'):
            parent_name = line.element_dict[name].parent_name
            if parent_name not in already_imported:
                self._import_element(line, parent_name, rename_elements,
                                     suffix_for_common_elements, already_imported)
            self.element_dict[new_name].parent_name = already_imported[parent_name]

        return new_name

    def import_line(
            self,
            line,
            suffix_for_common_elements=None,
            rename_elements={},
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
        rename_elements : dict, optional
            Dictionary with the elements to be renamed. The keys are the names
            of the elements in `line`, and the values are the new names.
        line_name : str, optional
            Name of the new line. If None, the name of the imported line is used.
        overwrite_vars : bool, optional
            If True, the variables in the imported line will overwrite the
            variables with the same name in this environment. Default is False.
        """
        line_name = line_name or line.name
        if suffix_for_common_elements is None:
            suffix_for_common_elements = f'/{line_name}'

        new_var_values = line.ref_manager.containers['vars']._owner
        if not overwrite_vars:
            new_var_values = new_var_values.copy()
            new_var_values.update(self.ref_manager.containers['vars']._owner)
        self.ref_manager.containers['vars']._owner.update(new_var_values)

        self.ref_manager.copy_expr_from(line.ref_manager, 'vars', overwrite=overwrite_vars)
        old_default_to_zero = self.vars.default_to_zero # Not sure why this is needed
        self.vars.default_to_zero = True
        self.ref_manager.run_tasks()
        self.vars.default_to_zero = old_default_to_zero

        components = []
        already_imported = {}
        for name in line.element_names:
            new_name = self._import_element(
                line, name, rename_elements, suffix_for_common_elements,
                already_imported)

            components.append(new_name)

        out = self.new_line(components=components, name=line_name)

        if line.particle_ref is not None:
            out.particle_ref = line.particle_ref.copy()

        out.config.clear()
        out.config.update(line.config.copy())
        out._extra_config.update(line._extra_config.copy())
        out.metadata.clear()
        out.metadata.update(line.metadata)

        if out.energy_program is not None:
            out.energy_program.line = out


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
        return xt.multiline_legacy._multiline_from_madx(cls, filename=filename, madx=madx, stdout=stdout,
                             return_lines=return_lines, **kwargs)

    @property
    def elements(self):
        return self.element_dict

    @property
    def line_names(self):
        return list(self.lines.keys())

    @property
    def functions(self):
        return self._xdeps_fref

    def _remove_element(self, name):

        pars_with_expr = list(
            self._xdeps_manager.tartasks[self.element_refs[name]].keys())

        # Kill all references
        for rr in pars_with_expr:
            if isinstance(rr, xd.refs.AttrRef):
                setattr(self[name], rr._key, 99)
            elif isinstance(rr, xd.refs.ItemRef):
                getattr(self[name], rr._owner._key)[rr._key] = 99
            else:
                raise ValueError('Only AttrRef and ItemRef are supported for now')

        self.element_dict.pop(name)

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

    def set_multipolar_errors(env, errors):

        '''
        Set multipolar errors for specified elements of the environment.

        Parameters
        ----------

        errors : dict
            Dictionary with the errors to be set. The keys are the names of the
            elements, and the values are dictionaries with the following keys:
             - rel_knl: list of relative errors for the normal multipolar strengths.
             - rel_ksl: list of relative errors for the skew multipolar strengths.
             - refer: name of the strength to be used as reference, which is
               multiplied by the length. If None, the default reference strength
               is used (k0 for bends, k1 for quadrupoles, k2 for sextupoles,
               and k3 for octupoles).

        Examples
        --------

        .. code-block:: python

            env = xt.Environment()
            env.vars.default_to_zero = True
            line = env.new_line(components=[
                env.new('mq', 'Quadrupole', length=0.5, k1='kq'),
                env.new('mqs', 'Quadrupole', length=0.5, k1s='kqs'),
                env.new('mb', 'Bend', length=0.5, angle='ang', k0_from_h=True),
            ])

            env.set_multipolar_errors({
                'mq': {'rel_knl': [1e-6, 1e-5, 1e-4],
                       'rel_ksl': [-1e-6, -1e-5, -1e-4]},
                'mqs': {'rel_knl': [2e-6, 2e-5, 2e-4],
                        'rel_ksl': [3e-6, 3e-5, 3e-4],
                        'refer': 'k1s'},
                'mb': {'rel_knl': [2e-6, 3e-5, 4e-4],
                       'rel_ksl': [5e-6, 6e-5, 7e-4]},
                })

        '''

        for ele_name in progress(errors.keys(), desc='Setting multipolar errors'):

            err = errors[ele_name]
            rel_knl = err.get('rel_knl', [])
            rel_ksl = err.get('rel_ksl', [])
            refer = err.get('refer', None)
            ele_class = env[ele_name].__class__.__name__

            if 'Replica' in ele_class or 'Slice' in ele_class:
                raise ValueError(f'Cannot set multipolar errors for element `{ele_name}`'
                                 f' of type `{ele_class}`')

            if refer is not None:
                reference_strength_name = refer
            else:
                reference_strength_name = DEFAULT_REF_STRENGTH_NAME.get(ele_class, None)

            if reference_strength_name is None:
                raise ValueError(f'Cannot find reference strength for element `{ele_name}`')

            ref_str_ref = getattr(env.ref[ele_name], reference_strength_name)
            length_ref = env.ref[ele_name].length

            for ii, kk in enumerate(rel_knl):
                err_vname = f'err_{ele_name}_knl{ii}'
                env[err_vname] = kk
                if (env.ref[ele_name].knl[ii]._expr is None or env.ref[err_vname] in
                        env.ref[ele_name].knl[ii]._expr._get_dependencies()):
                    env[ele_name].knl[ii] += env.ref[err_vname] * ref_str_ref * length_ref

            for ii, kk in enumerate(rel_ksl):
                err_vname = f'err_{ele_name}_ksl{ii}'
                env[err_vname] = kk
                if (env.ref[ele_name].ksl[ii]._expr is None or env.ref[err_vname] in
                        env.ref[ele_name].ksl[ii]._expr._get_dependencies()):
                    env[ele_name].ksl[ii] += env.ref[err_vname] * ref_str_ref * length_ref

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

def _compute_one_s(at, anchor, from_anchor, self_length, from_length, s_start_from,
                   default_anchor):

    if is_ref(at):
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

def _resolve_s_positions(seq_all_places, env, refer: ReferType = 'center',
                         allow_duplicate_places=True, s_tol=1e-10):

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
                    ss.at = 0
                    ss.from_ = ss_prev.name
                    ss.from_anchor = 'end'
                    n_resolved += 1
            else:
                if isinstance(ss.at, str):
                    at = aux_line._xdeps_eval.eval(ss.at)
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

    tt_out['from_'] = np.array([ss.from_ for ss in seq_all_places])
    tt_out['from_anchor'] = np.array([ss.from_anchor for ss in seq_all_places])

    return tt_out

# @profile
def _sort_places(tt_unsorted, s_tol=1e-10, allow_non_existent_from=False):

    tt_unsorted['i_place'] = np.arange(len(tt_unsorted))

    # Sort by s_center
    iii = _argsort_s(tt_unsorted.s_center, tol=s_tol)
    tt_s_sorted = tt_unsorted.rows[iii]

    group_id = np.zeros(len(tt_s_sorted), dtype=int)
    group_id[0] = 0
    for ii in range(1, len(tt_s_sorted)):
        if abs(tt_s_sorted.s_center[ii] - tt_s_sorted.s_center[ii-1]) > s_tol:
            group_id[ii] = group_id[ii-1] + 1
        elif tt_s_sorted.isthick[ii]: # Needed in Line.insert (on the first sorting pass there can be overlapping elements)
            group_id[ii] = group_id[ii-1] + 1
        else:
            group_id[ii] = group_id[ii-1]

    tt_s_sorted['group_id'] = group_id
    # tt_s_sorted.show(cols=['group_id', 's_center', 'name', 'from_', 'from_anchor', 'i_place'])

    # cache indices (indices will change but only within groups, so no need to update in the loop)
    # This trick gives me x40 speedup compared to using tt_s_sorted.rows.indices
    # at each iteration.
    ind_name = {nn: ii for ii, nn in enumerate(tt_s_sorted.name)}

    n_places = len(tt_s_sorted)
    i_start_group = 0
    i_place_sorted = []
    while i_start_group < n_places:
        i_group = tt_s_sorted['group_id', i_start_group]
        i_end_group = i_start_group + 1
        while i_end_group < n_places and tt_s_sorted['group_id', i_end_group] == i_group:
            i_end_group += 1
        # print(f'Group {i_group}: {tt_s_sorted.name[i_start_group:i_end_group]}')

        n_group = i_end_group - i_start_group
        if n_group == 1: # Single element
            i_place_sorted.append(tt_s_sorted.i_place[i_start_group])
            i_start_group = i_end_group
            continue

        if np.all(tt_s_sorted.from_anchor[i_start_group:i_end_group] == None): # Nothing to do
            i_place_sorted.extend(list(tt_s_sorted.i_place[i_start_group:i_end_group]))
            i_start_group = i_end_group

        tt_group = tt_s_sorted.rows[i_start_group:i_end_group]
        # tt_group.show(cols=['s_center', 'name', 'from_', 'from_anchor'])

        for ff in tt_group.from_:
            if ff is None:
                continue
            if ff not in ind_name:
                if allow_non_existent_from:
                    continue
                else:
                    raise ValueError(f'Element {ff} not found in the line')
            i_from_global = ind_name[ff] - i_start_group
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

            if np.all(np.diff(key_sort) >=0):
                continue # already sorted
            tt_group = tt_group.rows[np.argsort(key_sort, kind='stable')]

        i_place_sorted.extend(list(tt_group.i_place))
        i_start_group = i_end_group

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
            drift_name = env._get_a_drift_name()
            env.new(drift_name, xt.Drift, length=ds_upstream)
            names_with_drifts.append(drift_name)
        names_with_drifts.append(nn)

    if length is not None:
        length_line = tt_sorted['s_end'][-1]
        if length_line > length + s_tol:
            raise ValueError(f'Line length {length_line} is greater than the requested length {length}')
        if length_line < length - s_tol:
            drift_name = env._get_a_drift_name()
            env.new(drift_name, xt.Drift, length=length - length_line)
            names_with_drifts.append(drift_name)

    return list(map(str, names_with_drifts))

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
            and (not hasattr(cls, '_noexpr_fields') or kk not in cls._noexpr_fields)):
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


class Builder:
    def __init__(self, env, components=None, name=None, length=None, refer: ReferType = 'center'):
        self.env = env
        self.components = components or []
        self.name = name
        self.refer = refer
        self.length = length

    def __repr__(self):
        parts = [f'name={self.name!r}']
        if self.length is not None:
            parts.append(f'length={self.length!r}')
        if self.refer not in {'center', 'centre'}:
            parts.append(f'refer={self.refer!r}')
        parts.append(f'components={self.components!r}')
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

    def build(self, name=None):
        if name is None:
            name = self.name
        out =  self.env.new_line(components=self.components, name=name, refer=self.refer,
                                 length=self.length)
        out.builder = self
        return out

    def set(self, *args, **kwargs):
        self.components.append(self.env.set(*args, **kwargs))

    def get(self, *args, **kwargs):
        return self.env.get(*args, **kwargs)


    def resolve_s_positions(self):
        components = self.components
        if components is None:
            components = []

        components = _resolve_lines_in_components(components, self.env)
        flattened_components = _flatten_components(components, refer=self.refer)

        seq_all_places = _all_places(flattened_components)
        tab_unsorted = _resolve_s_positions(seq_all_places, self.env, refer=self.refer)
        tab_sorted = _sort_places(tab_unsorted)
        return tab_sorted

    def flatten(self, inplace=False):

        assert not inplace, 'Inplace not yet implemented'

        out = self.__class__(self.env)
        out.__dict__.update(self.__dict__)

        components = _resolve_lines_in_components(self.components, self.env)
        out.components = _flatten_components(components, refer=self.refer)
        out.components = _all_places(out.components)
        return out

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


def load_module_from_path(file_path):
    """
    Load a module from the given file path, always generating a unique module name.

    Parameters:
        file_path (str): The full path to the module file.

    Returns:
        module: The newly loaded module.
    """
    # Generate a unique module name using uuid4.
    module_name = f"module_{uuid.uuid4().hex}"

    # Create a module spec from the file location.
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Cannot load module from path: {file_path}")

    # Create a new module based on the spec.
    module = importlib.util.module_from_spec(spec)

    # Execute the module in its own namespace.
    spec.loader.exec_module(module)

    return module

def _reverse_element(env, name):
    """Return a reversed element without modifying the original."""

    SUPPORTED = {'RBend', 'Bend', 'Quadrupole', 'Sextupole', 'Octupole',
                'Multipole', 'Cavity', 'Solenoid', 'RFMultipole',
                'Marker', 'Drift', 'LimitRect', 'LimitEllipse', 'LimitPolygon',
                'LimitRectEllipse'}

    ee = env.get(name)
    ee_ref = env.ref[name]

    if ee.__class__.__name__ not in SUPPORTED:
        raise NotImplementedError(
            f'Cannot reverse the element `{name}`, as reversing elements '
            f'of type `{ee.__class__.__name__}` is not supported!'
        )

    def _reverse_field(key):
        if hasattr(ee, key):
            key_ref = getattr(ee_ref, key)
            if key_ref._expr is not None:
                setattr(ee_ref, key,  -(key_ref._expr))
            else:
                setattr(ee_ref, key,  -(key_ref._value))

    def _exchange_fields(key1, key2):
        value1 = None
        if hasattr(ee, key1):
            key1_ref = getattr(ee_ref, key1)
            value1 = key1_ref._expr or key1_ref._value

        value2 = None
        if hasattr(ee, key2):
            key2_ref = getattr(ee_ref, key2)
            value2 = key2_ref._expr or key2_ref._value


        if value1 is not None:
            setattr(ee_ref, key2, value1)

        if value2 is not None:
            setattr(ee_ref, key1, value2)

    _reverse_field('k0s')
    _reverse_field('k1')
    _reverse_field('k2s')
    _reverse_field('k3')
    _reverse_field('ks')
    _reverse_field('ksi')
    _reverse_field('rot_s_rad')

    if hasattr(ee, 'lag'):
        ee_ref.lag = 180 - (ee_ref.lag._expr or ee_ref.lag._value)

    if hasattr(ee, 'knl'):
        for i in range(1, len(ee.knl), 2):
            ee_ref.knl[i] = -(ee_ref.knl[i]._expr or ee_ref.knl[i]._value)

    if hasattr(ee, 'ksl'):
        for i in range(0, len(ee.ksl), 2):
            ee_ref.ksl[i] = -(ee_ref.ksl[i]._expr or ee_ref.ksl[i]._value)

    _exchange_fields('edge_entry_model', 'edge_exit_model')
    _exchange_fields('edge_entry_angle', 'edge_exit_angle')
    _exchange_fields('edge_entry_angle_fdown', 'edge_exit_angle_fdown')
    _exchange_fields('edge_entry_fint', 'edge_exit_fint')
    _exchange_fields('edge_entry_hgap', 'edge_exit_hgap')

def _resolve_lines_in_components(components, env):

    components = list(components) # Make a copy

    for ii, nn in enumerate(components):
        if (isinstance(nn, Place) and isinstance(nn.name, str)
                and nn.name in env.lines):
            nn.name = env.lines[nn.name]
        if isinstance(nn, str) and nn in env.lines:
            components[ii] = env.lines[nn]

    return components