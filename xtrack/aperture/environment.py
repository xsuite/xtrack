from __future__ import annotations

from typing import get_args
from weakref import ref

import xobjects as xo
from xtrack.api_categorization import GroupedAPICollector, doc_group, property_with_doc_group
from xtrack.json import dump as json_dump
from xtrack.json import load as json_load

from xtrack.aperture.structures import (
    ApertureEnvStruct,
    Pipe,
    Profile,
    ProfilePosition,
    ShapeTypes,
)
from xtrack.aperture.views import PipesView, ProfilesView


SHAPE_CLASSES = get_args(ShapeTypes)
SHAPE_CLASSES_BY_NAME = {shape_cls.__name__: shape_cls for shape_cls in SHAPE_CLASSES}

_env_aperture_doc_groups = GroupedAPICollector([
    "Construction",
    "Model Lifecycle",
    "Introspection",
    "Loading and Serialization",
])


def _shape_from_input(shape, **shape_params):
    """Create an aperture shape from a name, class, or existing shape."""
    if isinstance(shape, str):
        shape_cls = SHAPE_CLASSES_BY_NAME.get(shape)
    elif shape in SHAPE_CLASSES:
        shape_cls = shape
    elif isinstance(shape, SHAPE_CLASSES):
        if shape_params:
            raise ValueError(
                "Shape parameters cannot be provided with a shape instance.")
        return type(shape)(**shape._to_dict())
    else:
        shape_cls = None

    if shape_cls is None:
        raise ValueError(f"Unsupported aperture profile shape `{shape}`.")

    return shape_cls(**shape_params)


class EnvAperture:
    """Profiles and pipes shared by the aperture models in an environment.

    Before :meth:`build_model`, profiles and pipes are standalone Xobjects
    stored in Python lists. Building replaces those lists with one packed
    :class:`ApertureEnvStruct`. Only one representation is authoritative at a
    time; the name-based views resolve against whichever representation exists.

    Access this container through ``env.aperture``. Profiles and pipes can be
    shared by multiple lines in the environment. Individual definitions cannot
    be removed; :meth:`clear` removes all aperture data and dependent models.
    New definitions can only be added while the environment aperture is unbuilt.

    Examples
    --------
    .. code-block:: python

        import xtrack as xt

        env = xt.Environment()
        profile = env.aperture.new_profile('round', 'Circle', radius=0.02)
        pipe = env.aperture.new_pipe('tube', positions=[
            profile.place(at=0.0),
            profile.place(at=1.0),
        ])

        line = env.new_line(components=[env.new('drift', 'Drift', length=1.0)])
        line.aperture.place_pipe('installed', pipe, at='drift')
        line.aperture.build_model()
    """

    def __init__(self, env):
        self._env_ref = ref(env)
        self._profile_name_to_index = {}
        self._pipe_name_to_index = {}
        self._profiles_unbuilt = []
        self._pipes_unbuilt = []
        self._struct = None
        self._profiles = ProfilesView(self)
        self._pipes = PipesView(self)

    def __repr__(self):
        return f'<EnvAperture: {len(self.pipes)} pipes, {len(self.profiles)} profiles>'

    @classmethod
    def _generate_doc_rst(cls, *, include_properties=True, include_summary_table=True):
        """Generate grouped API documentation in RST format."""
        from xtrack.api_docs import generate_grouped_class_rst

        return generate_grouped_class_rst(
            cls,
            include_properties=include_properties,
            include_summary_table=include_summary_table,
        )

    @property_with_doc_group("Introspection")
    def env(self):
        """Environment owning this aperture container."""
        return self._env_ref()

    @property_with_doc_group("Introspection")
    def profiles(self) -> ProfilesView:
        """Named profile collection exposing persistent ``ProfileView`` objects.

        Use ``profiles[name]`` to inspect, edit, or plot a profile, and
        :meth:`new_profile` to create one. Views remain valid across model
        build and discard operations. Raw Xobject references do not provide
        that guarantee.
        """
        return self._profiles

    @property_with_doc_group("Introspection")
    def pipes(self) -> PipesView:
        """Named pipe collection exposing persistent ``PipeView`` objects.

        Use ``pipes[name]`` to inspect, edit, or plot a pipe, and
        :meth:`new_pipe` to create one. Views resolve the current authoritative
        Xobject, whether the environment aperture is built or unbuilt.
        """
        return self._pipes

    def __getstate__(self):
        """Leave the owner reference to be restored by the environment."""
        state = self.__dict__.copy()
        state.pop('_env_ref')
        return state

    @property_with_doc_group("Introspection")
    def is_built(self):
        """Whether profiles and pipes are stored in the packed structure."""
        return self._struct is not None

    def _resolve_profile(self, name):
        index = self._profile_name_to_index[name]
        if self._struct is not None:
            return self._struct.profiles[index]
        return self._profiles_unbuilt[index]

    def _resolve_pipe(self, name):
        index = self._pipe_name_to_index[name]
        if self._struct is not None:
            return self._struct.pipes[index]
        return self._pipes_unbuilt[index]

    @doc_group("Introspection")
    def profile_index(self, profile):
        """Return the stable index associated with a profile name or view.

        Parameters
        ----------
        profile : str or ProfileView
            Profile registered in this environment aperture.

        Returns
        -------
        int
            Profile index, preserved across build and discard operations.
        """
        if not isinstance(profile, str) and hasattr(profile, 'name'):
            profile = profile.name

        if isinstance(profile, str):
            return self._profile_name_to_index[profile]

        raise TypeError("A profile must be identified by name or ProfileView.")

    @doc_group("Construction")
    def place_profile(self, profile, at=0.0, shift_s=None, **kwargs):
        """Create a profile position for use in a pipe definition.

        This is also available as ``profile.place(...)`` on a ``ProfileView``.

        Parameters
        ----------
        profile : str or ProfileView
            Profile registered in this environment aperture.
        at : float, optional
            Longitudinal position along the pipe axis [m]. Defaults to zero.
        shift_s : float, optional
            Alternative to ``at``; do not specify both.
        **kwargs
            Additional ``ProfilePosition`` fields: transverse shifts,
            rotations, and aperture tolerances.

        Returns
        -------
        ProfilePosition
            Position to include in ``new_pipe(..., positions=[...])``.
        """
        if shift_s is not None:
            if at != 0.0:
                raise ValueError("Provide either `at` or `shift_s`, not both.")

            at = shift_s

        return ProfilePosition(
            profile_index=self.profile_index(profile),
            shift_s=at,
            **kwargs,
        )

    @doc_group("Introspection")
    def pipe_index(self, pipe):
        """Return the stable index associated with a pipe name or view.

        Parameters
        ----------
        pipe : str or PipeView
            Pipe registered in this environment aperture.

        Returns
        -------
        int
            Pipe index, preserved across build and discard operations.
        """
        if not isinstance(pipe, str) and hasattr(pipe, 'name'):
            pipe = pipe.name

        if isinstance(pipe, str):
            return self._pipe_name_to_index[pipe]

        raise TypeError("A pipe must be identified by name or PipeView.")

    @doc_group("Construction")
    def new_profile(self, name, shape, **shape_params):
        """Create a named profile while the environment aperture is unbuilt.

        Parameters
        ----------
        name : str
            Unique profile name within this environment aperture.
        shape : str, shape class, or shape instance
            Aperture shape, for example ``'Circle'``. Existing instances are
            copied; shape parameters cannot be supplied with an instance.
        **shape_params
            Constructor arguments for the selected shape, such as ``radius``.

        Returns
        -------
        ProfileView
            Persistent named view of the new profile.

        Notes
        -----
        If the environment aperture is built, call :meth:`discard_model`
        before adding profiles. This also discards dependent line models.
        """
        if self.is_built:
            raise RuntimeError(
                "Cannot add profiles while the aperture environment is built. "
                "Call `env.aperture.discard_model()` first.")

        if name in self._profile_name_to_index:
            raise ValueError(f"Profile `{name}` already exists.")

        profile = Profile(
            shape=_shape_from_input(shape, **shape_params),
        )
        index = len(self._profile_name_to_index)
        self._profiles_unbuilt.append(profile)
        self._profile_name_to_index[name] = index

        return self.profiles[name]

    @doc_group("Construction")
    def new_pipe(self, name, curvature=0.0, positions=None):
        """Create a named pipe while the environment aperture is unbuilt.

        Parameters
        ----------
        name : str
            Unique pipe name within this environment aperture.
        curvature : float, optional
            Horizontal curvature of the pipe axis [1/m]. Defaults to zero.
        positions : sequence of ProfilePosition, optional
            Profile positions created with :meth:`place_profile` or
            ``ProfileView.place()``. They are sorted longitudinally on build.

        Returns
        -------
        PipeView
            Persistent named view of the new pipe. Install it on a line with
            ``line.aperture.place_pipe(...)``.

        Notes
        -----
        If the environment aperture is built, call :meth:`discard_model`
        before adding pipes. This also discards dependent line models.
        """
        if self.is_built:
            raise RuntimeError(
                "Cannot add pipes while the aperture environment is built. "
                "Call `env.aperture.discard_model()` first.")

        if name in self._pipe_name_to_index:
            raise ValueError(f"Pipe `{name}` already exists.")

        pipe = Pipe(curvature=curvature, positions=positions or [])
        index = len(self._pipe_name_to_index)
        self._pipes_unbuilt.append(pipe)
        self._pipe_name_to_index[name] = index

        return self.pipes[name]

    @doc_group("Model Lifecycle")
    def build_model(self, context=None):
        """Freeze profiles and pipes into one shared packed representation.

        Building prevents adding profiles or pipes. Existing views still
        allow scalar edits. Such edits are not tracked automatically; call
        ``line.aperture.build_model()`` to refresh the line's derived polygons
        and bounds when needed.

        Parameters
        ----------
        context : xobjects.ContextCpu, optional
            Context used for the initial packed structure. Defaults to the
            Xobjects default context. If already built, the existing structure
            is returned without changing its context.

        Returns
        -------
        ApertureEnvStruct
            Shared kernel representation, also built implicitly by
            ``line.aperture.build_model()``. Profile positions within each
            pipe are sorted by longitudinal position; profile and pipe indices
            retain their insertion order.
        """
        if self._struct is None:
            # Profile positions need to be longitudinally ordered for the
            # interpolation kernels. Preserve profile and pipe insertion order.
            pipes = []
            for pipe in self._pipes_unbuilt:
                data = pipe._to_dict()
                data['positions'] = sorted(
                    data['positions'], key=lambda position: position['shift_s'])
                pipes.append(data)

            self._struct = ApertureEnvStruct(
                profiles=[item._to_dict() for item in self._profiles_unbuilt],
                pipes=pipes,
                profile_names=list(self._profile_name_to_index),
                pipe_names=list(self._pipe_name_to_index),
                _context=context or xo.context_default,
            )
            self._profiles_unbuilt = None
            self._pipes_unbuilt = None

        return self._struct

    @doc_group("Model Lifecycle")
    def discard_model(self):
        """Restore standalone profiles and pipes and discard dependent line models.

        Current values are preserved, and new definitions can be added again.
        Named views remain valid. Dependent lines lose their survey snapshots
        and derived geometry; their next model build captures a new survey.
        Calling this method while unbuilt has no effect.

        Returns
        -------
        EnvAperture
            This container.
        """
        if self._struct is None:
            return self

        # A built line model cannot be used without the packed environment.
        for line in list(self.env._lines_weakrefs):
            line.aperture.discard_model()

        self._profiles_unbuilt = [
            Profile(**profile._to_dict())
            for profile in self._struct.profiles
        ]
        self._pipes_unbuilt = [
            Pipe(**pipe._to_dict())
            for pipe in self._struct.pipes
        ]
        self._struct = None

        return self

    @doc_group("Model Lifecycle")
    def clear(self):
        """Remove all aperture definitions, placements, and built line models.

        Also clear unnamed lines belonging to this environment. Lattice elements
        and line aperture computation settings are retained. Return this container.
        """
        for line in list(self.env._lines_weakrefs):
            line.aperture._clear_model()
        self._struct = None
        self._profile_name_to_index = {}
        self._pipe_name_to_index = {}
        self._profiles_unbuilt = []
        self._pipes_unbuilt = []
        return self

    @doc_group("Loading and Serialization")
    def to_dict(self, include_line_models=True):
        """Serialize shared aperture definitions, optionally including line models.

        Parameters
        ----------
        include_line_models : bool, optional
            If True (default), include all named lines' aperture data. If
            False, return only the shared profile and pipe definitions.

        Returns
        -------
        dict
            Shared definitions under ``environment`` and per-line aperture
            data under ``lines`` when ``include_line_models=True``. Otherwise,
            return the shared definitions directly, without those wrapper keys.

        Notes
        -----
        Ordinary ``Environment.to_dict()`` omits aperture data unless
        ``include_aperture=True`` is supplied.
        """
        if self._struct is not None:
            definitions = self._struct.to_dict()
        else:
            definitions = {
                'profiles': [item._to_dict() for item in self._profiles_unbuilt],
                'pipes': [item._to_dict() for item in self._pipes_unbuilt],
                'profile_names': list(self._profile_name_to_index),
                'pipe_names': list(self._pipe_name_to_index),
            }

        if not include_line_models:
            return definitions

        return {
            'environment': definitions,
            'lines': {
                name: line.aperture.to_dict()
                for name, line in self.env.lines.items()
            },
        }

    @doc_group("Loading and Serialization")
    def to_json(self, filename, **kwargs):
        """Write aperture definitions and optional line models to JSON.

        Parameters
        ----------
        filename : str, pathlib.Path, or writable file object
            Destination for the data returned by :meth:`to_dict`.
        **kwargs
            Arguments passed to :meth:`to_dict`, including
            ``include_line_models``.
        """
        json_dump(self.to_dict(**kwargs), filename)

    @doc_group("Loading and Serialization")
    def from_dict(self, data, context=None):
        """Load definitions and optional named line apertures into a clean container.

        Accept either form returned by :meth:`to_dict`. Raise ValueError if any
        aperture definitions, placements, or built line models already exist;
        call :meth:`clear` first to replace them. Named lines must already exist.
        Line models are left unbuilt. Return this container.
        """
        if (self._profile_name_to_index or self._pipe_name_to_index
                or any(not line.aperture.empty or line.aperture._model is not None
                       for line in self.env._lines_weakrefs)):
            raise ValueError(
                "Cannot load into a non-empty aperture environment. "
                "Call `env.aperture.clear()` first.")

        lines = data.get('lines', {})
        for name in lines:
            if name not in self.env.lines:
                raise ValueError(f"Cannot load aperture for unknown line `{name}`.")

        definitions = data.get('environment', data)
        for kind in ('profile', 'pipe'):
            names = definitions[f'{kind}_names']
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate {kind} names in aperture data.")
        struct = ApertureEnvStruct(**definitions, _context=context)
        self._struct = struct
        self._profile_name_to_index = {
            name: index for index, name in enumerate(struct.profile_names)
        }
        self._pipe_name_to_index = {
            name: index for index, name in enumerate(struct.pipe_names)
        }
        self._profiles_unbuilt = None
        self._pipes_unbuilt = None
        for name, aperture in lines.items():
            self.env.lines[name].aperture.from_dict(aperture, context=context)
        return self

    @doc_group("Loading and Serialization")
    def from_json(self, filename, context=None):
        """Load an environment aperture export into a clean aperture container.

        Call ``env.aperture.clear()`` first if it already contains aperture data.
        Lattice lines must already exist. Models are left unbuilt; return self.
        """
        return self.from_dict(json_load(filename), context=context)


EnvAperture.__doc_groups__ = _env_aperture_doc_groups.collect(EnvAperture)
