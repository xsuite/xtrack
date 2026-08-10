"""Public API for building lines from longitudinal component placements.

The composer accepts a flexible component language: element names, ``Place``
objects, lines, composers, and nested iterables. Materialization expands
components, normalizes placements, resolves coordinates, orders coincident
elements, and fills positive gaps with drifts.
"""

import copy
from warnings import warn

import xdeps as xd
import xtrack as xt

from .components import (
    _all_places,
    _evaluate_length,
    _flatten_components,
    _generate_element_names_with_drifts,
    _validate_placement_geometry,
)
from .ordering import _sort_places
from .positions import _ALLOWED_ANCHORS, _resolve_s_positions
from ..general import DEPRECATION_INFO_PREP_1_0, parse_anchor_spec


class Composer:
    """Define a line by arranging components along its length.

    A composer is created automatically for a line in compose mode. Users should
    normally create such a line with ``env.new_line(compose=True)`` and add
    components through the line, rather than instantiate a composer directly. The
    underlying composer remains available as ``line.composer`` for inspection and
    validation.

    Components can be element names, placements, lines, other composers, or nested
    sequences of these objects. They can be placed at an absolute longitudinal
    position, relative to another component, or sequentially after the previous
    component.

    Parameters
    ----------
    env : xtrack.Environment
        Environment containing the elements, variables, and lines used by the
        composer.
    components : list, optional
        Components defining the line. Entries can be element names,
        :class:`xtrack.Place` objects, lines, composers, or nested sequences of
        these objects.
    length : float, str, or xdeps reference, optional
        Requested total length of the line. Strings and references are evaluated
        using ``env`` when the line is built. If omitted, the line is made just
        long enough to contain all its components.
    refer : {'start', 'center', 'centre', 'end'}, optional
        Default anchor used when a placement does not specify one. The default is
        ``'center'``.
    s_tol : float, optional
        Longitudinal tolerance used when filling gaps and checking overlaps and
        line-length constraints. The default is ``1e-6``.
    mirror : bool, optional
        If true, reverse the component sequence after assembling the line.
        The default is false.

    Examples
    --------
    Create a line in compose mode. Its composer is available as
    ``line.composer`` and can resolve component positions before the line is
    assembled:

    .. code-block:: python

        import xtrack as xt

        env = xt.Environment()
        env.new('q1', xt.Quadrupole, length=1)
        env.new('q2', xt.Quadrupole, length=1)
        env.new('ip', xt.Marker)

        line = env.new_line(compose=True)
        line.place('q1', at=1, anchor='start')
        line.place('ip', at=2, from_='q1', from_anchor='end')
        line.place('q2')

        positions = line.composer.resolve_s_positions()
        positions.cols['name s_start s_center s_end'].show()
        # name       s_start      s_center         s_end
        # q1               1           1.5             2
        # ip               4             4             4
        # q2               4           4.5             5

        line.end_compose()
    """

    def __init__(
        self,
        env,
        components=None,
        length=None,
        refer='center',
        s_tol=1e-6,
        mirror=False,
    ):
        if refer is None:
            refer = 'center'
        self.env = env
        self.components = components or []
        self.refer = refer
        self.length = length
        self.s_tol = s_tol
        self.mirror = mirror

    def copy(self):
        """Return a shallow copy of the composer.

        Returns
        -------
        xtrack.Composer
            A new composer with a separate top-level component list. The
            environment and the individual component objects are shared with the
            original composer.
        """
        out = self.__class__(self.env)
        out.__dict__.update(self.__dict__)
        out.components = self.components.copy()
        return out

    def __repr__(self):
        parts = []
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
        """Create an element and add it to the composer.

        Parameters
        ----------
        name : str
            Name of the new element.
        prototype : str or type
            Element type or existing element used as the prototype.
        at : float, str, or xdeps reference, optional
            Longitudinal position of the element. If ``from_`` is omitted, the
            position is measured from the beginning of the line.
        from_ : str, optional
            Component relative to which ``at`` is measured.
        extra : dict, optional
            Additional metadata associated with the new element.
        force : bool, optional
            If true, replace an existing element with the same name. The default
            is false.
        cls : str or type, optional
            Deprecated alias for ``prototype``.
        parent : str or type, optional
            Deprecated alias for ``prototype``.
        **kwargs
            Attributes used to initialize or customize the element.

        Returns
        -------
        str or xtrack.Place
            The name of the created element when neither ``at`` nor ``from_`` is
            provided; otherwise, an ``xtrack.Place`` object describing where the
            element is positioned. The returned value is also appended to
            ``composer.components``.
        """
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
        """Add a component to the composer.

        When neither ``at`` nor ``from_`` is provided, the component is placed
        sequentially after the preceding component.

        Parameters
        ----------
        name : str, xtrack.Line, or sequence of str
            Element or line to place. A sequence of element names is first combined
            into a line.
        obj : object, optional
            Object to register in the environment under ``name`` before placing it.
        at : float, str, or xdeps reference, optional
            Position of the selected component anchor. If ``from_`` is omitted, the
            position is measured from the beginning of the line.
        from_ : str, optional
            Component relative to which ``at`` is measured.
        anchor : {'start', 'center', 'centre', 'end'}, optional
            Anchor of the placed component positioned at ``at``. If omitted, the
            composer's default reference anchor is used.
        from_anchor : {'start', 'center', 'centre', 'end'}, optional
            Anchor of the reference component from which ``at`` is measured. If
            omitted, the composer's default reference anchor is used.

        Returns
        -------
        xtrack.Place
            The ``xtrack.Place`` object describing the component placement. It is
            also appended to ``composer.components``.

        Examples
        --------
        Place ``q1`` at an absolute position, place ``q2`` relative to the end of
        ``q1``, and place ``ip`` sequentially after ``q2``:

        .. code-block:: python

            import xtrack as xt

            env = xt.Environment()
            env.new('q1', xt.Quadrupole, length=1)
            env.new('q2', xt.Quadrupole, length=1)
            env.new('ip', xt.Marker)

            line = env.new_line(compose=True)
            composer = line.composer

            composer.place('q1', at=1, anchor='start')
            composer.place(
                'q2',
                at=2,
                from_='q1',
                anchor='start',
                from_anchor='end',
            )
            composer.place('ip')
        """
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
        s_tol=None,
        line=None,
        diagnostics=False,
    ):
        """Build a line from the current component definitions.

        Component positions are resolved and positive gaps are filled with drifts.
        A new line is created unless an existing line is supplied through ``line``.

        Parameters
        ----------
        name : str, optional
            Name under which the resulting line is registered in the environment.
        s_tol : float, optional
            Longitudinal tolerance used when filling gaps and checking overlaps and
            line-length constraints. If omitted, the composer's configured
            tolerance is used.
        line : xtrack.Line, optional
            Existing line whose element sequence is replaced with the assembled
            sequence. The line must belong to the composer's environment. If
            omitted, a new line is created.
        diagnostics : bool, optional
            If true, analyze unresolved placement dependencies and distinguish
            missing references from dependency cycles. The default is false.

        Returns
        -------
        xtrack.Line
            The assembled line. If ``line`` was provided, the same line object is
            returned.
        """
        if s_tol is None:
            s_tol = self.s_tol
        if line is not None and line.env is not self.env:
            raise ValueError('Line must belong to the same environment as the Composer')

        # evaluate line length if it is an expression
        length = _evaluate_length(self.env, self.length)

        expanded_components = _flatten_components(
            self.env,
            self.components,
            refer=self.refer,
        )

        if all(isinstance(component, str) for component in expanded_components):
            # Skip placement resolution for purely sequential components. This avoids
            # building a positions table and is significantly faster for large lines.
            element_names = list(map(str, expanded_components))
            if length is not None:
                components_length = self.env.new_line(
                    components=element_names
                ).get_length()
                if components_length > length + s_tol:
                    raise ValueError(
                        f'Line length {components_length} is greater than the '
                        f'requested length {length}'
                    )
                if components_length < length - s_tol:
                    drift = self.env.new(
                        self.env._get_a_drift_name(),
                        xt.Drift,
                        length=length - components_length,
                    )
                    element_names.append(drift)
        else:
            places = _all_places(expanded_components)
            positions = _resolve_s_positions(
                places,
                self.env,
                refer=self.refer,
                diagnostics=diagnostics,
            )
            positions = _sort_places(positions)
            element_names = _generate_element_names_with_drifts(
                self.env,
                positions,
                length=length,
                s_tol=s_tol,
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
        """Resolve the longitudinal positions of the composer components.

        The components are expanded and their start, center, and end positions are
        computed. This method neither assembles the final line nor fills gaps with
        drifts. Overlaps are not checked at this stage: overlapping components are
        included in the returned table with their resolved positions and do not
        raise an error.

        Parameters
        ----------
        sort : bool, optional
            If true, sort the returned rows by longitudinal position and order
            components sharing the same position according to their placement
            dependencies. If false, preserve the input component order. The default
            is true.
        diagnostics : bool, optional
            If true, analyze unresolved placement dependencies and distinguish
            missing references from dependency cycles. The default is false.

        Returns
        -------
        xtrack.line.LineTable
            Table containing one row per expanded component, including its
            ``s_start``, ``s_center``, and ``s_end`` positions.
        """
        expanded_components = _flatten_components(
            self.env,
            self.components or [],
            refer=self.refer,
        )
        places = _all_places(expanded_components)
        table = _resolve_s_positions(
            places,
            self.env,
            refer=self.refer,
            diagnostics=diagnostics,
        )
        return _sort_places(table) if sort else table

    def validate(self, s_tol=None, check_overlaps=True):
        """Validate the current component definitions.

        Component positions and dependencies are resolved with detailed
        diagnostics. The resolved components are then checked for overlaps and
        against the requested line length. Gap-filling drifts are not created.

        Parameters
        ----------
        s_tol : float, optional
            Longitudinal tolerance used when checking overlaps and the requested
            line length. If omitted, the composer's configured tolerance is used.
        check_overlaps : bool, optional
            If true, raise an error when components overlap. If false, skip the
            overlap check. The default is true.

        Returns
        -------
        None
            Returns normally when the component definitions are valid.

        Raises
        ------
        ValueError
            If a placement reference is missing, the placement dependencies contain
            a cycle, enabled overlap checking finds overlapping components, or the
            components exceed the requested line length.
        """
        if s_tol is None:
            s_tol = self.s_tol
        length = _evaluate_length(self.env, self.length)
        table = self.resolve_s_positions(sort=True, diagnostics=True)
        _validate_placement_geometry(
            table,
            length=length,
            s_tol=s_tol,
            check_overlaps=check_overlaps,
        )

    def flatten(self, inplace=False):
        """Return a composer with all nested components expanded.

        Lines, composers, and nested component sequences are expanded into a flat
        list of ``xtrack.Place`` objects. The original composer is not modified.

        Returns
        -------
        xtrack.Composer
            A new composer with the same configuration and environment, and a flat
            component list.
        """
        if inplace:
            raise NotImplementedError('Inplace flattening is not yet implemented')

        out = self.__class__(self.env)
        out.__dict__.update(self.__dict__)
        out.components = _flatten_components(
            self.env,
            self.components,
            refer=self.refer,
        )
        out.components = _all_places(out.components)
        return out

    def to_dict(self):
        """Serialize the composer definition to a dictionary.

        The dictionary contains the component placements and composer configuration,
        but not the elements, variables, or lines stored in the environment. It can
        be restored with :meth:`from_dict` using a compatible environment.

        Returns
        -------
        dict
            Dictionary representation of the composer.

        Raises
        ------
        NotImplementedError
            If a component is neither an element name nor an ``xtrack.Place``
            object.
        """
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
        """Create a composer from a dictionary representation.

        The environment must contain the elements, variables, and named lines
        referenced by the serialized component definitions. The input dictionary is
        not modified.

        Parameters
        ----------
        dct : dict
            Dictionary produced by :meth:`to_dict`.
        env : xtrack.Environment
            Environment in which the restored composer will resolve its components
            and expressions.

        Returns
        -------
        xtrack.Composer
            Composer restored from the dictionary.
        """
        data = dct.copy()
        data.pop('__class__', None)
        # ``name`` was serialized by the deprecated Builder API. It no longer
        # affects composers, but accepting it keeps older files loadable.
        data.pop('name', None)
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
    """Define how a component is positioned within a line.

    A ``Place`` object can position a component at an absolute longitudinal
    coordinate, relative to another component, or sequentially after the preceding
    component. Place objects are normally created with ``env.place(...)`` and passed
    to ``env.new_line(...)`` or added to a line in compose mode.

    Reference anchors can be written compactly by appending ``@anchor`` to the
    component name. For example, ``from_='q1@end'`` is equivalent to
    ``from_='q1', from_anchor='end'``. Similarly, ``at='q1@end'`` places a component
    with zero offset from the end of ``q1``.

    Parameters
    ----------
    name : str or xtrack.Line
        Element or line to place.
    at : float, str, or xdeps reference, optional
        Position of the selected component anchor. If ``from_`` is provided, this is
        an offset from the selected anchor of the reference component. A string of
        the form ``'name@anchor'`` places the component directly at that anchor. If
        omitted together with ``from_``, the component is placed sequentially.
    from_ : str, optional
        Component relative to which ``at`` is measured. The reference anchor can be
        included using the ``'name@anchor'`` form.
    anchor : {'start', 'center', 'centre', 'end'}, optional
        Anchor of the placed component positioned at ``at``. If omitted, the
        composer's default reference anchor is used.
    from_anchor : {'start', 'center', 'centre', 'end'}, optional
        Anchor of the reference component from which ``at`` is measured. If omitted,
        the composer's default reference anchor is used.
    env : xtrack.Environment, optional
        Associated environment. This is normally supplied automatically by
        ``env.place(...)``.

    Examples
    --------
    Place ``q1`` at an absolute position, place ``q2`` relative to the end of
    ``q1``, and place ``ip`` sequentially after ``q2``:

    .. code-block:: python

        import xtrack as xt

        env = xt.Environment()
        env.new('q1', xt.Quadrupole, length=1)
        env.new('q2', xt.Quadrupole, length=1)
        env.new('ip', xt.Marker)

        line = env.new_line(components=[
            env.place('q1', at=1, anchor='start'),
            env.place(
                'q2',
                at=2,
                from_='q1@end',
                anchor='start',
            ),
            env.place('ip'),
        ])
    """

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
