import importlib.util
import json
import re
import uuid
from collections import Counter, UserDict
from collections.abc import Iterable
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Literal
from warnings import warn
from weakref import WeakKeyDictionary, WeakSet

import numpy as np
import pandas as pd
import xdeps as xd
import xobjects as xo

import xtrack as xt

from .functions import Functions
from .match import Action
from .multiline_legacy.multiline_legacy import MultilineLegacy
from .progress_indicator import progress
from .view import View

ReferType = Literal['start', 'center', 'centre', 'end']

DEFAULT_REF_STRENGTH_NAME = {
    'Bend': '_k0', # using underscored to get the value also when k0_from_h is True
    'RBend': '_k0',
    'Quadrupole': 'k1',
    'Sextupole': 'k2',
    'Octupole': 'k3',
}



class Environment:

    def __init__(self, element_dict=None, particle_ref=None, lines=None,
                 _var_management_dct=None, particles=None):

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
         - place(...): creates a place object, which can be user in new_line(...).

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
        self._particles = {}
        self.particle_ref = particle_ref

        self._init_var_management(dct=_var_management_dct)
        self._line_vars = EnvVars(self)

        self.lines = EnvLines(self)
        self._lines_weakrefs = WeakSet()
        self._line_builders = WeakKeyDictionary()
        self._drift_counter = 0
        self.ref = EnvRef(self)
        self._elements = EnvElements(self)
        self._particles_container = EnvParticles(self)
        self._enable_name_clash_check = True
        self._last_context = None
        self._drift_cache = {}

        for nn, ee in self._element_dict.items():
            if nn.startswith('||drift_') and isinstance(ee, xt.Drift):
                self._drift_cache[ee.length] = nn

        if lines is not None:

            # Identify common elements
            counts = Counter()
            for ll in lines.values():
                # Extract names of all elements and parents
                elems_and_parents = set(ll.element_names)
                for nn in ll.element_names:
                    if hasattr(ll._element_dict[nn], 'parent_name'):
                        elems_and_parents.add(ll._element_dict[nn].parent_name)
                # Count if it is not a marker or a drift, which will be handled by
                # `import_line`
                for nn in elems_and_parents:
                    if (not (isinstance(ll._element_dict[nn], (xt.Marker))) and
                        not bool(re.match(r'^\|\|drift_\d+$', nn))):
                        counts[nn] += 1
            common_elements = [nn for nn, cc in counts.items() if cc>1]

            for nn, ll in lines.items():
                rename_elements = {el: el+'/'+nn for el in common_elements}
                self.import_line(line=ll, suffix_for_common_elements='/'+nn,
                    line_name=nn, rename_elements=rename_elements)
                self.lines[nn]._renamed_elements = rename_elements

        if particles is not None:
            self._particles.update(particles)

        self.metadata = {}

    def _init_var_management(self, dct=None):
        self._var_management = _make_var_management(
            element_dict=self._element_dict,
            particles=self._particles,
            dct=dct)

    def __repr__(self):
        line_names = list(self.lines.keys())
        n_lines = len(line_names)
        n_elements = len(self.elements)
        n_vars = (len(self.vars) if self.ref_manager is not None else 0)
        n_particles = len(self.particles)
        preview_tokens = []
        for ii, nn in enumerate(line_names):
            preview_tokens.append(nn)
            if ii >= 2:
                preview_tokens.append('...')
                break
        preview_lines = ', '.join(preview_tokens)
        return (f"Environment({n_lines} lines: {{{preview_lines}}}, "
                f"{n_elements} elements, {n_vars} vars, {n_particles} particles)")

    def __getstate__(self):
        out = self.__dict__.copy()
        out.pop('_lines_weakrefs')
        out.pop('_line_builders', None)
        out.pop('_xdeps_eval_obj', None)
        return out

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lines_weakrefs = WeakSet()
        self._line_builders = WeakKeyDictionary()

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

        if name in self.elements and not force:
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
            return xt.Place(self.new(**all_kwargs), at=at, from_=from_,
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
                self.lines[name] = parent.replicate(suffix=name, mirror=mirror)
                return name
            else:
                assert mode in [None, 'clone'], f'Unknown mode {mode}'
                assert name is not None, 'Name must be provided when cloning a line'
                self.lines[name] = parent.clone(suffix=name, mirror=mirror)
                return name

        assert mirror is False, 'mirror=True only allowed when cloning lines.'

        if parent is xt.Line or (parent=='Line' and (
            'Line' not in self.lines and 'Line' not in self.elements)):
            assert mode is None, 'Mode not allowed when cls is Line'
            return self.new_line(name=name, **kwargs)

        if mode == 'replica':
            assert parent in self.elements, f'Element {parent} not found, cannot replicate'
            kwargs['parent_name'] = xo.String(parent)
            parent = xt.Replica
        elif mode == 'clone':
            assert parent in self.elements, f'Element {parent} not found, cannot clone'
        else:
            assert mode is None, f'Unknown mode {mode}'

        _eval = self._xdeps_eval.eval

        if not (isinstance(parent, str) or parent in _ALLOWED_ELEMENT_TYPES_IN_NEW):
            raise ValueError(
            'Only '
            + _STR_ALLOWED_ELEMENT_TYPES_IN_NEW
            + ' elements are allowed in `new` for now. In this case it is possible '
            + 'to create a new element using the class and add it to the environment '
            + ' using `env.elements`. For example:\n\n'
            + '`env.elements["myname"] = MyClass(...)`\n'
            )

        needs_instantiation = True
        parent_element = None
        prototype = None
        if isinstance(parent, str):
            if parent in self.elements:
                # Clone an existing element
                prototype = parent
                self.elements[name] = xt.Replica(parent_name=parent)
                self.replace_replica(name)

                parent_element = self._element_dict[name]
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
            self.elements[name] = parent(**value_kwargs)

        self._set_kwargs(name=name, ref_kwargs=ref_kwargs, value_kwargs=value_kwargs,
                    container=self._element_dict, container_refs=self._xdeps_eref)

        if extra is not None:
            assert isinstance(extra, dict)
            if not hasattr(self[name], 'extra'):
                self[name].extra = {}
            for kk in extra:
                self.ref[name].extra[kk] = extra[kk]

        self._element_dict[name].prototype = prototype

        return name

    def new_particle(self, name, parent=None, force=False, **kwargs):

        '''
        Create a new particle type.

        Parameters
        ----------
        name : str
            Name of the new particle type
        parent : str or class
            Parent class or name of the parent particle type


        '''

        if name in self.particles and not force:
            raise ValueError(f'Particle `{name}` already exists')

        if parent is None:
            parent = xt.Particles

        _eval = self._xdeps_eval.eval

        needs_instantiation = True
        prototype = None
        if isinstance(parent, str):
            if parent in self.particles:
                # Clone an existing particle
                raise NotImplementedError # To be sorted out
                prototype = parent
                self.particles[name] = xt.Replica(parent_name=parent)
                self.replace_replica(self, name)

                parent_element = self._element_dict[name]
                parent = type(parent_element)
                needs_instantiation = False
            elif parent == 'Particles':
                parent = xt.Particles
                needs_instantiation = True
            else:
                 self.particles[name] = xt.particles.reference_from_pdg_id(parent)
                 parent = xt.Particles
                 needs_instantiation = False

        # Make lists where needed
        for kk in kwargs:
            if not np.isscalar(kwargs[kk]):
                continue
            if kk in xt.Particles._xofields and 'Arr' in xt.Particles._xofields[kk].__name__:
                kwargs[kk] = [kwargs[kk]]

        ref_kwargs, value_kwargs = _parse_kwargs(parent, kwargs, _eval)

        if needs_instantiation: # Parent is a class and not another particle
            self.particles[name] = parent(**value_kwargs)

        self._set_kwargs(name=name, ref_kwargs=ref_kwargs, value_kwargs=value_kwargs,
                    container=self._particles, container_refs=self._xdeps_pref)

        self.particles[name].prototype = prototype

        return name


    def new_line(self, components=None, name=None, refer: ReferType = 'center',
                 length=None, mirror=False, s_tol=1e-6, compose=False) -> xt.Line:
        """
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
        """

        if isinstance(components, str):
            raise ValueError('components must be a list or tuple, not a string')

        out = xt.Line(env=self, compose=True, length=length, refer=refer,
                      s_tol=s_tol, mirror=mirror)

        if components is not None:
            out.composer.components += list(components)

        if not compose:
            out.end_compose()

        self._lines_weakrefs.add(out) # Weak references

        out._name = name

        # Temporary solution to keep consistency in multiline
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            out._var_management = None
            out._in_multiline = self._in_multiline
            out._name_in_multiline = self._name_in_multiline

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
            assert name not in self.elements
            self.elements[name] = obj

        # if list of strings, create a line
        if (isinstance(name, Iterable) and not isinstance(name, str)
            and all(isinstance(item, str) for item in name)):
            name = self.new_line(components=name)

        if isinstance(name, xt.Line):
            if hasattr(name, 'name') and name.name is not None:
                assert name.name in self.lines
                name = name.name
            else:
                name = name.copy(shallow=True)

        return xt.Place(name, at=at, from_=from_, anchor=anchor, from_anchor=from_anchor)

    def new_builder(self, components=None, name=None, refer: ReferType = 'center',
                    length=None, s_tol=1e-6):
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

        out = xt.Builder(env=self, components=components, name=name, refer=refer,
                       length=length, s_tol=s_tol)

        return out

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

    def copy(self):
        return self.__class__.from_dict(self.to_dict())

    def copy_element_from(self, name, source, new_name=None):
        """Copy an element from another environment.

        Parameters
        ----------
        name: str
            Name of the element to copy.
        source: Environment | Line
            Environment or line where the element is located.
        new_name: str, optional
            Rename the element in the new line/environment. If not provided, the
            element is copied with the same name.
        """
        new_name = new_name or name
        cls = type(source._element_dict[name])

        if (cls not in xt.line._ALLOWED_ELEMENT_TYPES_IN_NEW + [xt.DipoleEdge] # No issue in copying DipoleEdge while creating it requires handling properties which are strings.
            and 'ThickSlice' not in cls.__name__ and 'ThinSlice' not in cls.__name__
            and 'DriftSlice' not in cls.__name__):
            raise ValueError(
                f'Only {xt.line._STR_ALLOWED_ELEMENT_TYPES_IN_NEW} elements are '
                f'allowed in `copy_from_env` for now.'
            )

        self._element_dict[new_name] = source._element_dict[name].copy()

        if self.ref_manager is not None:
            pars_with_expr = list(
                source._xdeps_manager.tartasks[source._xdeps_eref[name]].keys())
        else:
            pars_with_expr = []

        formatter = xd.refs.CompactFormatter(scope=None)

        for rr in pars_with_expr:
            # Assign expressions by string to avoid having to deal with the
            # fact that they come from a different manager!
            expr_string = rr._expr._formatted(formatter)
            new_expr = self.new_expr(expr_string)

            if isinstance(rr, xd.refs.AttrRef):
                setattr(self.ref[new_name], rr._key, new_expr)
            elif isinstance(rr, xd.refs.ItemRef):
                getattr(self.ref[new_name], rr._owner._key)[rr._key] = new_expr
            else:
                raise ValueError('Only AttrRef and ItemRef are supported for now')

        return new_name

    def replace_replica(self, name):
        name_parent = self._element_dict[name].resolve(self, get_name=True)
        self.copy_element_from(name_parent, self, new_name=name)

    def _import_element(self, line, name, rename_elements, suffix_for_common_elements,
                        already_imported):
        new_name = name
        if name in rename_elements:
            new_name = rename_elements[name]
        elif (bool(re.match(r'^\|\|drift_\d+$', name))
            and line.ref[name].length._expr is None):
            new_name = self._get_a_drift_name()
        elif (name in self.elements and
                not (isinstance(line[name], xt.Marker) and
                    isinstance(self._element_dict.get(name), xt.Marker))):
            new_name += suffix_for_common_elements

        self.copy_element_from(name, line, new_name=new_name)
        already_imported[name] = new_name
        if hasattr(line._element_dict[name], 'parent_name'):
            parent_name = line._element_dict[name].parent_name
            if parent_name not in already_imported:
                self._import_element(line, parent_name, rename_elements,
                                     suffix_for_common_elements, already_imported)
            self._element_dict[new_name].parent_name = already_imported[parent_name]

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

    def discard_trackers(self):
        '''Discard all trackers in all lines of the environment.'''
        for ln in self._lines_weakrefs:
            if ln._has_valid_tracker():
                ln.discard_tracker()

    def _get_a_drift_name(self):
        self._drift_counter += 1
        while nn := f'||drift_{self._drift_counter}':
            if nn not in self.elements:
                return nn
            self._drift_counter += 1

    def _get_drift(self, length):
        if length in self._drift_cache:
            return self._drift_cache[length]
        nn = self._get_a_drift_name()
        self.elements[nn] = xt.Drift(length=length)
        self._drift_cache[length] = nn
        return nn

    def __setitem__(self, key, value):

        if isinstance(value, xt.Line):
            assert value.env is self, 'Line must be in the same environment'
            if key in self.lines:
                raise ValueError(f'There is already a line with name {key}')
            if key in self.elements:
                raise ValueError(f'There is already an element with name {key}')
            self.lines[key] = value
        elif np.isscalar(value) or xd.refs.is_ref(value):
            self.vars[key] = value
        else:
            raise ValueError('Only lines, scalars or references are allowed')

    def to_dict(self, include_var_management=True, include_version=True):

        out = {}
        out['__class__'] = self.__class__.__name__

        if include_version:
            out["xtrack_version"] = xt.__version__

        out["elements"] = {k: el.to_dict() for k, el in self._element_dict.items()}

        if self._particle_ref is not None:
            if isinstance(self._particle_ref, str):
                out['particle_ref'] = self._particle_ref
            else:
                out['particle_ref'] = self._particle_ref.to_dict()
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
            if isinstance(ll, xt.Line):
                out['lines'][nn] = ll.to_dict(include_element_dict=False,
                                            include_var_management=False)
            else:
                raise ValueError(f'Unknown line type {type(ll)}')

        out['particles'] = {}
        for nn, pp in self.particles.items():
            out['particles'][nn] = pp.to_dict()

        return out

    @classmethod
    def from_dict(cls, dct, _context=None, _buffer=None, classes=()):
        cls = xt.Environment

        if "xtrack_version" in dct:
            version = dct["xtrack_version"]
            if xt.general._compare_versions(version, xt.__version__) > 0:
                print(f'Warning: The environment you are loading was created '
                      f'with xtrack version {version}, which is more recent '
                      f'than the current version {xt.__version__}. '
                      'Some features may not be available or '
                      f'may not work correctly. Please update your xsuite '
                      f'package to the latest version.')

        elements = _deserialize_elements(dct=dct, classes=classes,
                                         _buffer=_buffer, _context=_context)
        particles = {}
        if 'particles' in dct:
            for nn, ppd in dct['particles'].items():
               particles[nn] = xt.Particles.from_dict(ppd)

        particle_ref = None
        if 'particle_ref' in dct.keys():
            particle_ref = dct['particle_ref']
            if not isinstance(particle_ref, str):
                particle_ref = xt.Particles.from_dict(particle_ref,
                                    _context=_context, _buffer=_buffer)

        if '_var_manager' in dct.keys():
            _var_management_dct = dct
        else:
            _var_management_dct = None

        out = cls(element_dict=elements, particle_ref=particle_ref,
                _var_management_dct=_var_management_dct, particles=particles)

        dct_lines = dct.copy()
        dct_lines.pop('elements', None)
        for nn in dct['lines'].keys():
            ddll = dct_lines['lines'][nn]
            llcls = ddll.get('__class__', 'Line') # For backward compatibility
            if llcls == 'Line':
                ll = xt.Line.from_dict(ddll, _env=out, verbose=False)
            else:
                raise ValueError(f'Unknown line type {type(ll)}')

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
        """Constructs an environment from a JSON file.

        Parameters
        ----------
        file : str or file-like object
            Path to the JSON file or file-like object.
            If filename ends with '.gz' file is decompressed.
        **kwargs : dict
            Additional keyword arguments passed to `Environment.from_dict`.

        Returns
        -------
        environment : Environment
            Environment object.

        """
        warn('`Environment.from_json` is deprecated, use `xt.load` instead', FutureWarning)
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

        if 'include_version' not in kwargs:
            kwargs['include_version'] = True

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
        return self._elements

    @property
    def particles(self):
        return self._particles_container

    def set_particle_ref(self, *args, lines=True, **kwargs):

        if lines is True:
            lines = self.lines.keys()
        elif lines is False or lines is None:
            lines = []
        elif isinstance(lines, str):
            lines = [lines]
        elif isinstance(lines, Iterable):
            lines = list(lines)
        else:
            raise ValueError('lines must be True, False, None, a string or an iterable of strings')

        if len(args)==1 and isinstance(args[0], xt.Particles):
            self.particle_ref = args[0].copy()
            for ln in lines:
                self.lines[ln].particle_ref = self.particle_ref.copy()
        elif len(args)==1 and isinstance(args[0], str):
            name = args[0]
            if name in self.particles:
                self.particle_ref = name
                for ln in lines:
                    self.lines[ln].particle_ref = name
            else:
                self.particle_ref = xt.Particles(*args, **kwargs)
                for ln in lines:
                    self.lines[ln].particle_ref = self.particle_ref.copy()
        else:
            self.particle_ref = xt.Particles(*args, **kwargs)
            for ln in lines:
                self.lines[ln].particle_ref = self.particle_ref.copy()

    @property
    def particle_ref(self):
        if self._particle_ref is None:
            return None
        return EnvParticleRef(self)

    @particle_ref.setter
    def particle_ref(self, particle_ref):
        self._particle_ref = particle_ref

    @property
    def line_names(self):
        return list(self.lines.keys())

    @property
    def functions(self):
        return self._xdeps_fref

    def _remove_element(self, name):

        pars_with_expr = list(
            self._xdeps_manager.tartasks[self._xdeps_eref[name]].keys())

        # Kill all references
        for rr in pars_with_expr:
            if isinstance(rr, xd.refs.AttrRef):
                setattr(self[name], rr._key, 99)
            elif isinstance(rr, xd.refs.ItemRef):
                getattr(self[name], rr._owner._key)[rr._key] = 99
            else:
                raise ValueError('Only AttrRef and ItemRef are supported for now')

        self._element_dict.pop(name)

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

    @property
    def element_dict(self):
        return self._element_dict

    @element_dict.setter
    def element_dict(self, value):
        if self._element_dict is None:
            self._element_dict = {}
        self._element_dict.clear()
        self._element_dict.update(value)

    @property
    def _xdeps_vref(self):
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            return self._in_multiline._xdeps_vref
        if self._var_management is not None:
            return self._var_management['vref']

    @property
    def _xdeps_eref(self):
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            return self._in_multiline.element_refs
        if self._var_management is not None:
            return self._var_management['lref']

    @property
    def _xdeps_fref(self):
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            return self._in_multiline._xdeps_fref
        if self._var_management is not None:
            return self._var_management['fref']

    @property
    def _xdeps_manager(self):
        if hasattr(self, '_in_multiline') and self._in_multiline is not None:
            return self._in_multiline._xdeps_manager
        if self._var_management is not None:
            return self._var_management['manager']

    @property
    def _xdeps_eval(self):
        try:
            eva_obj = self._xdeps_eval_obj
        except AttributeError:
            eva_obj = None

        # in case the var manager has been replaced, invalidate the cached eval obj
        if eva_obj is not None and eva_obj.variables is not self._xdeps_vref:
            eva_obj = None

        if eva_obj is None:
            eva_obj = xd.madxutils.MadxEval(variables=self._xdeps_vref,
                                            functions=self._xdeps_fref,
                                            elements=self._element_dict,
                                            get='attr')
            self._xdeps_eval_obj = eva_obj

        return eva_obj

    @property
    def vars(self):
        return self._line_vars

    @property
    def varval(self):
        return self.vars.val

    @property
    def vv(self): # Shorter alias
        return self.vars.val

    def eval(self, expr):
        '''
        Get the value of an expression

        Parameters
        ----------
        expr : str
            Expression to evaluate.

        Returns
        -------
        value : float
            Value of the expression.
        '''

        return self.vars.eval(expr)


    @property
    def element_refs(self):
        if self._var_management is not None:
            return self._var_management['lref']

    @property
    def _xdeps_pref(self):
        if self._var_management is not None:
            return self._var_management['pref']

    def __getitem__(self, key):
        assert isinstance(key, str)
        if key in self._element_dict:
            if self.ref_manager is None:
                return self._element_dict[key]
            return View(self._element_dict[key], self._xdeps_eref[key],
                        evaluator=self._xdeps_eval.eval)
        elif key in self.particles:
            if self._xdeps_pref is None:
                return self.particles[key]
            return View(self.particles[key], self._xdeps_pref[key],
                        evaluator=self._xdeps_eval.eval)
        elif self.ref_manager is not None and key in self.vars:
            return self.vv[key]
        elif key in self.lines: # Want to reuse the method for the env
            return self.lines[key]
        else:
            raise KeyError(f'Name `{key}` not found')

    def __contains__(self, key):
        return (key in self._element_dict or
                key in self.particles or
                key in self.lines or
                (self.ref_manager is not None and key in self.vars)
                )

    def remove(self, key):

        if key in self._element_dict:
            self.elements.remove(key)
        elif key in self.particles:
            self.particles.remove(key)
        elif key in self.lines:
            self.lines.remove(key)
        elif self.ref_manager is not None and key in self.vars:
            self.vars.remove(key)
        else:
            raise KeyError(f'Name {key} not found')

    def __delitem__(self, key):
        self.remove(key)

    def set(self, name, *args, **kwargs):
        '''
        Set the values or expressions of variables or element properties.

        Parameters
        ----------
        name : str
            Name(s) of the variable or element.
        value: float or str
            Value or expression of the variable to set. Can be provided only
            if the name is associated to a variable.
        **kwargs, float or str
            Attributes to set. Can be provided only if the name is associated
            to an element.

        Examples
        --------
        >>> line.set('a', 0.1)
        >>> line.set('k1', '3*a')
        >>> line.set('quad', k1=0.1, k2='3*a')
        >>> line.set(['quad1', 'quad2'], k1=0.1, k2='3*a')
        >>> line.set(['c', 'd'], 0.1)
        >>> line.set(['e', 'f'], '3*a')

        '''
        if hasattr(name, 'env_name'):
            name = name.env_name
        elif hasattr(name, 'name'):
            name = name.name

        if isinstance(name, Iterable) and not isinstance(name, str):
            for nn in name:
                self.set(nn, *args, **kwargs)
            return

        _eval = self._xdeps_eval.eval

        if hasattr(self, 'lines') and name in self.lines:
            raise ValueError('Cannot set a line')

        if name in self.elements:
            if len(args) > 0:
                raise ValueError(f'Only kwargs are allowed when setting element attributes')

            extra = kwargs.pop('extra', None)

            ref_kwargs, value_kwargs = xt.environment._parse_kwargs(
                type(self._element_dict[name]), kwargs, _eval)
            self._set_kwargs(
                name=name, ref_kwargs=ref_kwargs, value_kwargs=value_kwargs,
                container=self._element_dict, container_refs=self._xdeps_eref)
            if extra is not None:
                assert isinstance(extra, dict), (
                    'Description must be a dictionary')
                if (not hasattr(self._element_dict[name], 'extra')
                    or not isinstance(self._element_dict[name].extra, dict)):
                    self._element_dict[name].extra = {}
                self._element_dict[name].extra.update(extra)
        else:
            if len(kwargs) > 0:
                raise ValueError(f'Only a single value is allowed when setting variable')
            if len(args) != 1:
                raise ValueError(f'A value must be provided when setting a variable')
            value = args[0]
            if 'extra' in kwargs and kwargs['extra'] is not None:
                raise ValueError(f'Extra is only allowed for elements')
            if isinstance(value, str):
                self.vars[name] = _eval(value)
            else:
                self.vars[name] = value

    def get(self, key):
        '''
        Get an element or the value of a variable.

        Parameters
        ----------
        key : str
            Name of the element or variable.

        Returns
        -------
        element : Element or float
            Element or value of the variable.

        '''

        if key in self._element_dict:
            return self._element_dict[key]
        elif key in self.particles:
            return self._particles[key]
        elif self._xdeps_vref and key in self._xdeps_vref._owner:
            return self._xdeps_vref._owner[key]
        else:
            raise KeyError(f'Element or variable {key} not found')

    def info(self, key, limit=30):
        """
            Get information about an element or a variable.
        """

        if key in self.elements:
            return self[key].get_info()
        elif key in self.vars:
            return self.vars.info(key, limit=limit)
        else:
            raise KeyError(f'Element or variable {key} not found')


    def get_expr(self, var):
        '''
        Get expression associated to a variable

        Parameters
        ----------
        var: str
            Name of the variable

        Returns
        -------
        expr : Expression
            Expression associated to the variable
        '''

        return self.vars.get_expr(var)

    def new_expr(self, expr):
        '''
        Create a new expression

        Parameters
        ----------
        expr : str
            Expression to create.

        Returns
        -------
        expr : Expression
            New expression.
        '''
        return self.vars.new_expr(expr)

    def extend_knl_ksl(self, order, element_names=None):

        """
        Extend the order of the knl and ksl attributes of the elements.

        Parameters
        ----------
        order: int
            New order of the knl and ksl attributes.
        element_names: list of str
            Names of the elements to extend. If None, all elements having `knl`
            and `ksl` attributes are extended.

        """

        if element_names is None:
            raise NotImplementedError(
                'Extending knl and ksl for all elements is not implemented yet.')

        if isinstance(element_names, str):
            element_names = [element_names]

        self.discard_trackers()

        for nn in element_names:
            if self.get(nn).order > order:
                raise ValueError(f'Order of element {nn} is smaller than {order}')

        for nn in element_names:
            ee = self.get(nn)

            if ee.order == order:
                continue

            new_knl = [vv for vv in ee.knl] + [0] * (order - len(ee.knl) + 1)
            new_ksl = [vv for vv in ee.ksl] + [0] * (order - len(ee.ksl) + 1)

            dct = ee.to_dict()
            dct.pop('order', None)
            dct['knl'] = new_knl
            dct['ksl'] = new_ksl

            new_ee = ee.__class__.from_dict(dct, _buffer=ee._buffer)
            # Need to bypass the check on element redefinition
            self._xdeps_eref._owner[nn] = new_ee

    @property
    def ref_manager(self):
        return self._xdeps_manager

    def _var_management_to_dict(self):
        out = {}
        out['_var_management_data'] = deepcopy(self._var_management['data'])
        for kk in out['_var_management_data'].keys():
            if hasattr(out['_var_management_data'][kk], 'to_dict'):
                out['_var_management_data'][kk] = (
                    out['_var_management_data'][kk].to_dict())
        out['_var_manager'] = self._var_management['manager'].dump()
        return out

    def _check_name_clashes(self, name, check_vars=True):
        if not self._enable_name_clash_check:
            return
        if name in self._element_dict:
            raise ValueError(f'There is already an element with name {name}')
        if name in self.lines:
            raise ValueError(f'There is already a line with name {name}')
        if name in self._particles:
            raise ValueError(f'There is already a particle with name {name}')
        if (check_vars and self._xdeps_vref is not None
            and name in self._xdeps_vref._owner):
            raise ValueError(f'There is already a variable with name {name}')

    def _unregister_object(self,name):

        rr = self.ref[name]

        revdeps = self.ref_manager.find_deps([rr])
        if len(revdeps) > 1:
            raise RuntimeError(f'Cannot remove object {name} because it is used '
                               f'to control: {revdeps[1:]}')

        for task in list(self.ref_manager.tasks):
            deps = task._get_dependencies()
            if rr in deps:
                self.ref_manager.unregister(task)

    def _set_kwargs(self, name, ref_kwargs, value_kwargs, container, container_refs):
        for kk in value_kwargs:
            if hasattr(value_kwargs[kk], '__iter__') and not isinstance(value_kwargs[kk], str):
                len_value = len(value_kwargs[kk])
                target = getattr(container[name], kk)
                if len(target) < len_value:
                    if kk=='knl' or kk=='ksl' and name in self._element_dict:
                        self.extend_knl_ksl(len_value-1, element_names=[name])
                        target = getattr(container[name], kk)
                    else:
                        raise ValueError(
                            f'Cannot set attribute {kk} of element {name}: '
                            f'length mismatch ({len(target)} vs {len_value})')
                target[:len_value] = value_kwargs[kk]
                if kk in ref_kwargs:
                    for ii, vvv in enumerate(value_kwargs[kk]):
                        if ref_kwargs[kk][ii] is not None:
                            getattr(container_refs[name], kk)[ii] = ref_kwargs[kk][ii]
            elif kk in ref_kwargs:
                setattr(container_refs[name], kk, ref_kwargs[kk])
            else:
                setattr(container[name], kk, value_kwargs[kk])

    twiss = MultilineLegacy.twiss
    build_trackers = MultilineLegacy.build_trackers
    match = MultilineLegacy.match
    match_knob = MultilineLegacy.match_knob
    install_beambeam_interactions = MultilineLegacy.install_beambeam_interactions
    configure_beambeam_interactions =  MultilineLegacy.configure_beambeam_interactions
    apply_filling_pattern = MultilineLegacy.apply_filling_pattern





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



class EnvElements:
    def __init__(self, env):
        self.env = env

    def __getitem__(self, key):

        if key in self.env._element_dict:
            if self.env.ref_manager is None:
                return self.env._element_dict[key]
            return View(self.env._element_dict[key], self.env._xdeps_eref[key],
                        evaluator=self.env._xdeps_eval.eval)
        else:
            raise KeyError(f'Element `{key}` not found.')

    def __setitem__(self, key, value):
        self.env._check_name_clashes(key)
        self.env._element_dict[key] = value

    def __contains__(self, key):
        return key in self.env._element_dict

    def __getattr__(self, name):
        env = object.__getattribute__(self, 'env')
        return getattr(env._element_dict, name)

    def __repr__(self):
        names = list(self.env._element_dict.keys())
        n = len(names)
        preview = ', '.join(names[:5]) + (', ...' if n > 5 else '')
        return f'EnvElements({n} elements: {{{preview}}})'

    def __len__(self):
        return len(self.env._element_dict)

    def get_table(self, attr=False):
        names = sorted(list(self.env._element_dict.keys()))
        dumline = self.env.new_line(components=names)
        tt = dumline.get_table(attr=attr)
        assert tt.name[-1] == '_end_point'
        tt = tt.rows[:-1] # Remove endpoint
        for cc in ['s', 's_start', 's_center', 's_end', 'env_name']:
            if cc in tt._col_names:
                tt._col_names.remove(cc)
                del tt._data[cc]
        if 'length' not in tt._col_names:
            tt['length'] = np.array(
                [getattr(self.env._element_dict[nn], 'length', 0) for nn in tt.name])
        return tt

    def remove(self, name):

        if name not in self.env._element_dict:
            raise KeyError(f'Element `{name}` not found.')

        if (self.env.ref_manager is not None
            and not isinstance(self.env._element_dict[name], xt.Marker)):

            self.env._unregister_object(name)

        self.env.discard_trackers()
        del self.env._element_dict[name]

    def __delitem__(self, name):
        self.remove(name)


class EnvParticles:
    def __init__(self, env):
        self.env = env

    def __getitem__(self, key):

        if key in self.env._particles:
            if self.env.ref_manager is None:
                return self.env._particles[key]
            return View(self.env._particles[key], self.env._xdeps_pref[key],
                        evaluator=self.env._xdeps_eval.eval)
        else:
            raise KeyError(f'Particle `{key}` not found.')

    def __setitem__(self, key, value):
        self.env._check_name_clashes(key)
        self.env._particles[key] = value

    def __contains__(self, key):
        return key in self.env._particles

    def __getattr__(self, name):
        env = object.__getattribute__(self, 'env')
        return getattr(env._particles, name)

    def __repr__(self):
        names = list(self.env._particles.keys())
        n = len(names)
        preview = ', '.join(names[:5]) + (', ...' if n > 5 else '')
        return f'EnvParticles({n} particles: {{{preview}}})'

    def __len__(self):
        return len(self.env._particles)

    def get_table(self):
        names = np.array(sorted(list(self.env._particles.keys())))
        mass0 = np.array(
            [self.env._particles[nn].mass0 for nn in names])
        charge0 = np.array(
            [self.env._particles[nn].q0 for nn in names])
        energy0 = np.array(
            [self.env._particles[nn].energy0[0] for nn in names])
        p0c = np.array(
            [self.env._particles[nn].p0c[0] for nn in names])
        gamma0 = np.array(
            [self.env._particles[nn].gamma0[0] for nn in names])
        beta0 = np.array(
            [self.env._particles[nn].beta0[0] for nn in names])
        tt = xt.Table({
            'name': names,
            'mass0': mass0,
            'charge0': charge0,
            'energy0': energy0,
            'p0c': p0c,
            'gamma0': gamma0,
            'beta0': beta0
        })
        return tt

    def remove(self, name):

        if name not in self.env._particles:
            raise KeyError(f'Particle `{name}` not found.')

        if self.env.ref_manager is not None:
            self.env._unregister_object(name)

        del self.env._particles[name]

    def __delitem__(self, name):
        self.remove(name)


class EnvRef:
    def __init__(self, env):
        self.env = env

    def __getitem__(self, name):
        if hasattr(self.env, 'lines') and name in self.env.lines:
            return self.env.lines[name].ref
        elif name in self.env._element_dict:
            return self.env._xdeps_eref[name]
        elif name in self.env.vars:
            return self.env._xdeps_vref[name]
        elif name in self.env.particles:
            return self.env._xdeps_pref[name]
        else:
            raise KeyError(f'Name {name} not found.')

    def __setitem__(self, key, value):
        if isinstance(value, xt.Line):
            assert value.env is self.env, 'Line must be in the same environment'
            if key in self.env.lines:
                raise ValueError(f'There is already a line with name {key}')
            if key in self.env._element_dict:
                raise ValueError(f'There is already an element with name {key}')
            self.env.lines[key] = value

        if hasattr(value, '_value'):
            val_ref = value
            val_value = value._value
        else:
            val_ref = value
            val_value = value

        if np.isscalar(val_value):
            if key in self.env._element_dict:
                raise ValueError(f'There is already an element with name {key}')
            self.env.vars[key] = val_ref
        else:
            if key in self.env.vars:
                raise ValueError(f'There is already a variable with name {key}')
            self.env._xdeps_eref[key] = val_ref

    @property
    def elements(self):
        return self.env._xdeps_eref

    @property
    def particles(self):
        return self.env._xdeps_pref

    @property
    def vars(self):
        return self.env._xdeps_vref


class EnvLines(UserDict):

    def __init__(self, env):
        self.data = {}
        self.env = env

    def __setitem__(self, key, value):
        self.env._check_name_clashes(key)
        self.env._lines_weakrefs.add(value)
        UserDict.__setitem__(self, key, value)

    def get_table(self):
        names = np.array(list(self.keys()))
        num_elements = np.array([len(self.env.lines[nn]) for nn in names])
        mode = np.array([self.env.lines[nn].mode for nn in names])
        tt = xt.Table({'name': names, 'num_elements': num_elements, 'mode':mode})
        return tt

    def __repr__(self):
        names = list(self.keys())
        n = len(names)
        preview = ', '.join(names[:5]) + (', ...' if n > 5 else '')
        return f'EnvLines({n} lines: {{{preview}}})'

    def remove(self, name):

        del self.env.lines[name]

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
                'Multipole', 'Cavity', 'UniformSolenoid',
                'Marker', 'Drift', 'LimitRect', 'LimitEllipse', 'LimitPolygon',
                'LimitRectEllipse', 'CrabCavity'}

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

    if ee.__class__.__name__ == 'CrabCavity':
        ee_ref.crab_voltage = -(ee_ref.crab_voltage._expr or ee_ref.crab_voltage._value)

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



def _deserialize_elements(dct, classes, _buffer, _context):
    class_dict = xt.line.mk_class_namespace(classes)

    _buffer = xo.get_a_buffer(context=_context, buffer=_buffer,size=8)

    if isinstance(dct['elements'], dict):
        elements = {}
        for (kk, ee) in progress(dct['elements'].items(), desc='Loading line from dict'):
            elements[kk] = xt.line._deserialize_element(ee, class_dict, _buffer)
    elif isinstance(dct['elements'], list):
        elements = []
        for ii, ee in enumerate(
                progress(dct['elements'], desc='Loading line from dict')):
            elements.append(xt.line._deserialize_element(ee, class_dict, _buffer))
    else:
        raise ValueError('Field `elements` must be a dict or a list')

    return elements

def _make_var_management(element_dict, particles, dct=None):

    from collections import defaultdict

    _var_values = defaultdict(lambda: 0)
    _var_values.default_factory = None

    functions = Functions()

    manager = xd.Manager()
    _vref = manager.ref(_var_values, 'vars')
    _fref = manager.ref(functions, 'f')
    _lref = manager.ref(element_dict, 'element_refs')
    _pref = manager.ref(particles, 'particles')

    _var_management = {}
    _var_management['data'] = {}
    _var_management['data']['var_values'] = _var_values
    _var_management['data']['functions'] = functions

    _var_management['manager'] = manager
    _var_management['lref'] = _lref
    _var_management['vref'] = _vref
    _var_management['fref'] = _fref
    _var_management['pref'] = _pref

    _vref['t_turn_s'] = 0.0

    if dct is not None:
        manager = _var_management['manager']
        for kk in dct['_var_management_data'].keys():
            data_item = dct['_var_management_data'][kk]
            if kk == 'functions':
                data_item = Functions.from_dict(data_item)
            _var_management['data'][kk].update(data_item)
        manager.load(dct['_var_manager'])

    return _var_management

class EnvParticleRef:

    def __init__(self, env):
        self.env = env

    @property
    def _resolved(self):
        _particle_ref = self.env._particle_ref
        if isinstance(_particle_ref, str):
            return self.env[_particle_ref]
        else:
            return _particle_ref

    def __getattr__(self, key):
        return getattr(self._resolved, key)

    def __setattr__(self, key, value):
        if key == 'env':
            object.__setattr__(self, key, value)
        else:
            setattr(self._resolved, key, value)

    def copy(self, **kwargs):
        return self._resolved.copy(**kwargs)

class EnvVars:

    def __init__(self, env):
        self.env = env
        if '__vary_default' not in self.env._xdeps_vref._owner.keys():
            self.env._xdeps_vref._owner['__vary_default'] = {}
        self.val = VarValues(self)
        self.vars_to_update = WeakSet()

    def __repr__(self):
        if self.env._xdeps_vref is None:
            return 'EnvVars(inactive, no xdeps manager)'
        n = len(self.env._xdeps_vref._owner) - 1
        names_preview = []
        for ii, kk in enumerate(self.env._xdeps_vref._owner.keys()):
            if kk != '__vary_default':
                names_preview.append(str(kk))
            if ii == 5:
                names_preview.append('...')
                break
        preview = ', '.join(names_preview)
        return f'EnvVars({n} vars: {{{preview}}})'

    def keys(self):
        if self.env._xdeps_vref is None:
            raise RuntimeError(
                f'Cannot access variables as the environment has no xdeps manager')
        out = list(self.env._xdeps_vref._owner.keys()).copy()
        return out

    def __iter__(self):
        raise NotImplementedError('Use keys() method') # Untested
        return self.env._xdeps_vref._owner.__iter__()

    def __len__(self):
        if self.env._xdeps_vref is None:
            raise RuntimeError(
                f'Cannot access variables as the environment has no xdeps manager')
        return len(self.env._xdeps_vref._owner) - 1

    def update(self, *args, **kwargs):
        default_to_zero = kwargs.pop('default_to_zero', None)
        old_default_to_zero = self.default_to_zero
        if default_to_zero is not None:
            self.default_to_zero = default_to_zero
        try:
            if self.env._xdeps_vref is None:
                raise RuntimeError(
                    f'Cannot access variables as the environment has no xdeps manager')
            if len(args) > 0:
                assert len(args) == 1, 'update expected at most 1 positional argument'
                other = args[0]
                for kk in other.keys():
                    self[kk] = other[kk]
            for kk, vv in kwargs.items():
                self[kk] = vv
        except Exception as ee:
            if default_to_zero is not None:
                self.default_to_zero = old_default_to_zero
            raise ee
        if default_to_zero is not None:
            self.default_to_zero = old_default_to_zero

    def load(
            self,
            file=None,
            string=None,
            format: Literal['json', 'madx', 'python'] = None,
            timeout=5.,
        ):

        if isinstance(file, Path):
            file = str(file)

        if (file is None) == (string is None):
            raise ValueError('Must specify either file or string, but not both')

        FORMATS = {'json', 'madx', 'python'}
        if string and format not in FORMATS:
            raise ValueError(f'Format must be specified to be one of {FORMATS} when '
                            f'using string input')

        if format is None and file is not None:
            if file.endswith('.json') or file.endswith('.json.gz'):
                format = 'json'
            elif file.endswith('.str') or file.endswith('.madx'):
                format = 'madx'
            elif file.endswith('.py'):
                format = 'python'

        if file and (file.startswith('http://') or file.startswith('https://')):
            string = xt.general.read_url(file, timeout=timeout)
            file = None

        if format == 'json':
            ddd = xt.json.load(file=file, string=string)
            self.update(ddd, default_to_zero=True)
        elif format == 'madx':
            return self.load_madx(file, string)
        elif format == 'python':
            if string is not None:
                raise NotImplementedError('Loading from string not implemented for python format')
            env = xt.Environment()
            env.call(file)
            self.update(env.vars.get_table().to_dict(), default_to_zero=True)
            return env

    @property
    def vary_default(self):
        if self.env._xdeps_vref is None:
            raise RuntimeError(
                f'Cannot access variables as the environment has no xdeps manager')
        return self.env._xdeps_vref._owner['__vary_default']

    def get_table(self, compact=True):
        if self.env._xdeps_vref is None:
            raise RuntimeError(
                f'Cannot access variables as the environment has no xdeps manager')
        name = np.array([kk for kk in list(self.keys()) if kk != '__vary_default'], dtype=object)
        value = np.array([self.env._xdeps_vref[kk]._value for kk in name])

        if compact:
            formatter = xd.refs.CompactFormatter(scope=None)
            expr = []
            for kk in name:
                ee = self.env._xdeps_vref[kk]._expr
                if ee is None:
                    expr.append(None)
                else:
                    expr.append(ee._formatted(formatter))
        else:
            expr  = [self.env._xdeps_vref[str(kk)]._expr for kk in name]
            for ii, ee in enumerate(expr):
                if ee is not None:
                    expr[ii] = str(ee)

        expr = np.array(expr)

        return VarsTable({'name': name, 'value': value, 'expr': expr})

    def new_expr(self, expr):
        return self.env._xdeps_eval.eval(expr)

    def eval(self, expr):
        expr_or_value = self.new_expr(expr)
        if xd.refs.is_ref(expr_or_value):
            return expr_or_value._get_value()
        return expr_or_value

    def info(self, var, limit=10):
        return self[var]._info(limit=limit)

    def get_expr(self, var):
        return self[var]._expr

    def rename(self, old, new, verbose=False):

        env = self.env
        mgr = env.ref_manager
        env.vars[new] = env.vv[old]
        r_old = env.ref[old]
        r_new = env.ref[new]
        t_old = mgr.tasks.get(r_old)
        if t_old is not None:
            if verbose:
                print(f"replacing target {t_old} with {r_new}={t_old.expr}")
            mgr.set_value(r_new, t_old.expr)
        for rt in list(env.ref_manager.rdeps[r_old]):
            if rt in mgr.tasks:
                tt = mgr.tasks[rt]
                old_expr = str(tt.expr)
                new_expr = old_expr.replace(str(r_old), str(r_new))
                if verbose:
                    print(f"replancing {old_expr} with {new_expr}")
                mgr.set_value(rt, eval(new_expr, mgr.containers))

        if verbose:
            env.info(old, limit=None)
            env.info(new, limit=None)

        env.vars.remove(old)

    def __contains__(self, key):
        if self.env._xdeps_vref is None:
            raise RuntimeError(
                f'Cannot access variables as the environment has no xdeps manager')
        return key in self.env._xdeps_vref._owner

    def get_independent_vars(self):

        """
        Returns the list of independent variables in the environment.
        """

        out = []
        for kk in self.keys():
            if self[kk]._expr is None:
                out.append(kk)
        return out

    def __getitem__(self, key):
        if key not in self: # uses __contains__ method
            raise KeyError(f'Variable `{key}` not found')
        return self.env._xdeps_vref[key]

    def __setitem__(self, key, value):
        self.env._check_name_clashes(key, check_vars=False)
        if isinstance(value, str):
            value = self.env._xdeps_eval.eval(value)
        self.env._xdeps_vref[key] = value
        for cc in self.vars_to_update:
            cc[key] = value

    def __getstate__(self):
        out = self.__dict__.copy()
        out['vars_to_update'] = None
        return out

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.vars_to_update = WeakSet()

    def set_from_madx_file(self, filename=None, string=None):
        """Deprecated: see `load_madx` instead."""
        warn('EnvVars.set_from_madx_file is deprecated, use `load_madx` instead.', FutureWarning)
        self.load_madx(filename=filename, string=string)

    def load_madx_optics_file(self, filename=None, string=None):
        """Deprecated: see `load_madx` instead."""
        warn('EnvVars.load_madx_optics_file is deprecated, use `load_madx` instead.', FutureWarning)
        self.load_madx(filename=filename, string=string)

    def load_madx(self, filename=None, string=None):
        """
        Set variables values of expression from a MAD-X file.

        Parameters
        ----------
        filename: str or list of str
            Path to the MAD-X file(s) to load.
        string: str
            MAD-X source string to load.
        """
        old_default_to_zero = self.default_to_zero
        loader = xt.mad_parser.MadxLoader(env=self.env)
        if filename is not None:
            assert string is None, 'Cannot specify both filename and string'
            loader.load_file(filename)
        elif string is not None:
            assert filename is None, 'Cannot specify both filename and string'
            loader.load_string(string)
        self.default_to_zero = old_default_to_zero  # restore (in case changed by loader)

    def load_json(self, filename):
        warn(
            '`EnvVars.load_json` is deprecated, use `load`, optionally with `format="json"` instead.',
            FutureWarning
        )
        with open(filename, 'r') as fid:
            data = json.load(fid)

        _old_default_to_zero = self.default_to_zero
        self.default_to_zero = True
        self.update(data)
        self.default_to_zero = _old_default_to_zero

    def target(self, tar, value, **kwargs):
        action = ActionVars(self.env)
        return xt.Target(action=action, tar=tar, value=value, **kwargs)

    def __call__(self, *args, **kwargs):
        _eval = self.env._xdeps_eval.eval
        if len(args) > 0:
            assert len(kwargs) == 0
            assert len(args) == 1
            if isinstance(args[0], str):
                return self[args[0]]
            elif isinstance(args[0], dict):
                kwargs.update(args[0])
            else:
                raise ValueError('Invalid argument')
        for kk in kwargs:
            if isinstance(kwargs[kk], str):
                self[kk] = _eval(kwargs[kk])
            else:
                self[kk] = kwargs[kk]

    def set(self, name, value):
        if isinstance(value, str):
            self[name] = self.env._xdeps_eval.eval(value)
        else:
            self[name] = value

    def get(self, name):
        return self[name]._value

    @property
    def default_to_zero(self):
        default_factory = self.env._xdeps_vref._owner.default_factory
        if default_factory is None:
            return False
        return default_factory.default == 0

    @default_to_zero.setter
    def default_to_zero(self, value):
        assert value in (True, False)
        if value:
            self.env._xdeps_vref._owner.default_factory = _DefaultFactory(0.)
        else:
            self.env._xdeps_vref._owner.default_factory = None

    def remove(self, name):

        if name not in self:
            raise KeyError(f'Variable `{name}` not found')

        if self.env.ref_manager is not None:
            self.env._unregister_object(name)

        del self.env._xdeps_vref._owner[name]

    def __delitem__(self, name):
        self.remove(name)

class VarsTable(xd.Table):

    def to_dict(self):
        out = {}
        for nn, ee, vv in zip(self['name'], self['expr'], self['value']):
            if ee is not None:
                out[nn] = ee
            else:
                out[nn] = vv
        return out

class ActionVars(Action):

    def __init__(self, line):
        self.line = line

    def run(self, **kwargs):
        return self.line._xdeps_vref._owner

class VarValues:

    def __init__(self, vars):
        self.vars = vars

    def __getitem__(self, key):
        return self.vars[key]._value

    def __setitem__(self, key, value):
        self.vars[key] = value

    def get(self,key, default=0):
        if key in self.vars:
            return self.vars[key]._value
        else:
            return default

class _DefaultFactory:
    def __init__(self, default):
        self.default = default

    def __call__(self):
        return self.default

@contextmanager
def _disable_name_clash_checks(env):
    old_value = env._enable_name_clash_check
    env._enable_name_clash_check = False
    try:
        yield
    finally:
        env._enable_name_clash_check = old_value
