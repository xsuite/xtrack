"""Public API for building lines from longitudinal component placements.

The composer accepts a flexible component language: element names, ``Place``
objects, lines, composers, and nested iterables. Materialization follows an
explicit pipeline implemented in :mod:`xtrack._composer`: expand components,
normalize placements, resolve coordinates, order coincident elements, and
materialize positive gaps as drifts.
"""

import copy
from warnings import warn

import xdeps as xd
import xtrack as xt

from .components import (
    _all_places,
    _evaluate_length,
    _expand_components,
    _validate_placement_geometry,
)
from .ordering import _sort_places
from .pipeline import _build_element_names
from .positions import _ALLOWED_ANCHORS, _resolve_s_positions
from ..general import DEPRECATION_INFO_PREP_1_0, parse_anchor_spec


class Composer:
    """Mutable specification used to assemble an :class:`xtrack.Line`.

    Parameters are retained rather than immediately materialized, allowing the
    line to be regenerated after referenced variables or element lengths change.
    """

    def __init__(
        self,
        env,
        components=None,
        name=None,
        length=None,
        refer='center',
        s_tol=1e-6,
        mirror=False,
    ):
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
        return f'Composer({", ".join(parts)})'

    def new(
        self,
        name,
        prototype=None,
        at=None,
        from_=None,
        extra=None,
        force=False,
        cls=None,
        parent=None,
        **kwargs,
    ):
        """Create an element in the environment and add its placement."""
        if cls is not None:
            if prototype is not None:
                raise TypeError(
                    'Only one of `prototype` and deprecated `cls` can be provided.'
                )
            warn(
                'The `cls` argument of `Line.new(...)` is deprecated. Use '
                '`prototype` instead.' + DEPRECATION_INFO_PREP_1_0,
                FutureWarning,
                stacklevel=3,
            )
            prototype = cls

        if parent is not None:
            if prototype is not None:
                raise TypeError(
                    'Only one of `prototype` and deprecated `parent` can be provided.'
                )
            warn(
                'The `parent` argument of `Line.new(...)` is deprecated. Use '
                '`prototype` instead.' + DEPRECATION_INFO_PREP_1_0,
                FutureWarning,
                stacklevel=3,
            )
            prototype = parent

        if prototype is None:
            raise TypeError("Line.new() missing required argument: 'prototype'")

        out = self.env.new(
            name,
            prototype,
            at=at,
            from_=from_,
            extra=extra,
            force=force,
            **kwargs,
        )
        self.components.append(out)
        return out

    def place(self, name, obj=None, at=None, from_=None, anchor=None, from_anchor=None):
        """Create and append a placement using the associated environment."""
        out = self.env.place(
            name=name,
            obj=obj,
            at=at,
            from_=from_,
            anchor=anchor,
            from_anchor=from_anchor,
        )
        self.components.append(out)
        return out

    def build(
        self,
        name=None,
        inplace=None,
        s_tol=None,
        line=None,
        diagnostics=False,
    ):
        """Materialize the current component specification as a line.

        If ``line`` is supplied it is updated in place. Otherwise a new line is
        returned. A named composer builds back into its environment by default.
        """
        if inplace is None and name is None and self.name is not None:
            inplace = True
        if inplace and self.name is None:
            raise ValueError('Inplace build requires the Composer to have a name')
        if inplace:
            name = self.name
        if s_tol is None:
            s_tol = self.s_tol
        if line is not None and line.env is not self.env:
            raise ValueError('Line must belong to the same environment as the Composer')

        length = _evaluate_length(self.env, self.length)
        expanded_components = _expand_components(
            self.env, self.components, refer=self.refer
        )
        element_names = _build_element_names(
            self.env,
            expanded_components,
            refer=self.refer,
            length=length,
            s_tol=s_tol,
            diagnostics=diagnostics,
        )

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

    def resolve_s_positions(self, sort=True, diagnostics=False):
        """Return a table containing the resolved component coordinates."""
        expanded_components = _expand_components(
            self.env, self.components or [], refer=self.refer
        )
        places = _all_places(expanded_components)
        table = _resolve_s_positions(
            places,
            self.env,
            refer=self.refer,
            diagnostics=diagnostics,
        )
        return _sort_places(table) if sort else table

    def validate(self, s_tol=None):
        """Validate the current placement specification without building a line."""
        if s_tol is None:
            s_tol = self.s_tol
        length = _evaluate_length(self.env, self.length)
        table = self.resolve_s_positions(sort=True, diagnostics=True)
        _validate_placement_geometry(table, length=length, s_tol=s_tol)

    def flatten(self, inplace=False):
        """Return a shallow copy whose nested components have been expanded."""
        if inplace:
            raise NotImplementedError('Inplace flattening is not yet implemented')

        out = self.__class__(self.env)
        out.__dict__.update(self.__dict__)
        out.components = _expand_components(self.env, self.components, refer=self.refer)
        out.components = _all_places(out.components)
        return out

    def to_dict(self):
        """Serialize this specification without environment element data."""
        data = {
            '__class__': self.__class__.__name__,
            'components': [],
        }
        formatter = xd.refs.CompactFormatter(scope=None)

        for component in self.components:
            if isinstance(component, str):
                data['components'].append(component)
                continue
            if not isinstance(component, xt.Place):
                raise NotImplementedError(
                    'Only Place components are implemented for now'
                )

            name = component.name
            if hasattr(name, 'to_dict'):
                name = name.to_dict(
                    include_element_dict=False,
                    include_var_management=False,
                )
            component_data = {'name': name}
            if component.at is not None:
                if xd.refs.is_ref(component.at):
                    component_data['at'] = component.at._formatted(formatter)
                else:
                    component_data['at'] = component.at
            if component.from_ is not None:
                component_data['from_'] = component.from_
            if component.anchor is not None:
                component_data['anchor'] = component.anchor
            if component.from_anchor is not None:
                component_data['from_anchor'] = component.from_anchor
            data['components'].append(component_data)

        if self.name is not None:
            data['name'] = self.name
        if self.refer is not None:
            data['refer'] = self.refer
        if self.length is not None:
            if xd.refs.is_ref(self.length):
                data['length'] = self.length._formatted(formatter)
            else:
                data['l'] = self.length
        if self.s_tol is not None:
            data['s_tol'] = self.s_tol
        if self.mirror:
            data['mirror'] = self.mirror
        return data

    @classmethod
    def from_dict(cls, dct, env):
        """Restore a composer while accepting legacy ``l`` length data."""
        data = dct.copy()
        data.pop('__class__', None)
        if 'l' in data:
            if 'length' in data:
                raise ValueError(
                    'Composer dictionary cannot contain both `l` and `length`.'
                )
            data['length'] = data.pop('l')

        out = cls(env=env)
        components = data.pop('components')
        for component_data in components:
            if isinstance(component_data, str):
                out.components.append(component_data)
                continue

            component_data = component_data.copy()
            if isinstance(component_data['name'], dict):
                if component_data['name'].get('__class__') != 'Line':
                    raise ValueError(
                        'Only serialized Line objects can be used as '
                        'Composer component names.'
                    )
                component_data['name'] = xt.Line.from_dict(
                    component_data['name'], _env=env
                )
            out.place(**component_data)

        for key, value in data.items():
            setattr(out, key, value)
        return out


class Place:
    """Placement of a component relative to an element or line coordinate."""

    def __init__(
        self, name, at=None, from_=None, anchor=None, from_anchor=None, env=None
    ):
        if isinstance(at, str):
            if '@' in at:
                if from_ is not None or from_anchor is not None:
                    raise ValueError(
                        'An anchor specification in `at` cannot be combined '
                        'with `from_` or `from_anchor`.'
                    )
                from_, from_anchor = parse_anchor_spec(at)
                at = 0
            elif env is not None and at in env._element_dict:
                from_ = at
                at = 0

        if from_ is not None:
            if not isinstance(from_, str):
                raise TypeError('`from_` must be a string or None.')
            if '@' in from_:
                from_, from_anchor = parse_anchor_spec(from_)

        if anchor not in _ALLOWED_ANCHORS:
            raise ValueError(f'Unknown anchor {anchor!r}.')
        if from_anchor not in _ALLOWED_ANCHORS:
            raise ValueError(f'Unknown from_anchor {from_anchor!r}.')

        self.name = name
        self.at = at
        self.from_ = from_
        self.anchor = anchor
        self.from_anchor = from_anchor
        self.env = env

    def __repr__(self):
        out = f'Place({self.name}'
        if self.at is not None:
            out += f', at={self.at}'
        if self.from_ is not None:
            out += f', from_={self.from_}'
        if self.anchor is not None:
            out += f', anchor={self.anchor}'
        if self.from_anchor is not None:
            out += f', from_anchor={self.from_anchor}'
        return out + ')'

    def copy(self):
        return copy.copy(self)
