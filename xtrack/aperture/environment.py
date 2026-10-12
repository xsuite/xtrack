from __future__ import annotations

from typing import get_args
from weakref import ref

import xobjects as xo
from xtrack.json import dump as json_dump

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


class ProfileCollection(ProfilesView):
    """Named profile collection with a profile-position factory."""

    def place(self, profile, at=0.0, shift_s=None, **kwargs):
        """Create a profile position to be passed to ``new_pipe``."""
        if shift_s is not None:
            if at != 0.0:
                raise ValueError("Provide either `at` or `shift_s`, not both.")

            at = shift_s

        index = self._owner.profile_index(profile)
        return ProfilePosition(profile_index=index, shift_s=at, **kwargs)


class EnvAperture:
    """Profiles and pipes shared by the aperture models in an environment.

    Before :meth:`build_model`, profiles and pipes are standalone Xobjects
    stored in Python lists. Building replaces those lists with one packed
    :class:`ApertureEnvStruct`. Only one representation is authoritative at a
    time; the name-based views resolve against whichever representation exists.
    """

    def __init__(self, env):
        self._env_ref = ref(env)
        self._profile_name_to_index = {}
        self._pipe_name_to_index = {}
        self._profiles_unbuilt = []
        self._pipes_unbuilt = []
        self._struct = None
        self.profiles = ProfileCollection(self)
        self.pipes = PipesView(self)

    @property
    def env(self):
        """Environment owning this aperture container."""
        return self._env_ref()

    @property
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

    def profile_index(self, profile):
        """Return the stable index associated with a profile name or view."""
        if not isinstance(profile, str) and hasattr(profile, 'name'):
            profile = profile.name

        if isinstance(profile, str):
            return self._profile_name_to_index[profile]

        raise TypeError("A profile must be identified by name or ProfileView.")

    def pipe_index(self, pipe):
        """Return the stable index associated with a pipe name or view."""
        if not isinstance(pipe, str) and hasattr(pipe, 'name'):
            pipe = pipe.name

        if isinstance(pipe, str):
            return self._pipe_name_to_index[pipe]

        raise TypeError("A pipe must be identified by name or PipeView.")

    def new_profile(self, name, shape, **shape_params):
        """Create a named profile while the environment is unbuilt."""
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

    def new_pipe(self, name, curvature=0.0, positions=None):
        """Create a named pipe while the environment is unbuilt."""
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

    def build_model(self, context=None):
        """Freeze profiles and pipes into one packed kernel representation."""
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

    def discard_model(self):
        """Restore standalone profiles and pipes on all dependent lines."""
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

    def _to_dict(self):
        """Serialize the environment-owned aperture data without changing state."""
        if self._struct is not None:
            return self._struct.to_dict()

        return {
            'profiles': [item._to_dict() for item in self._profiles_unbuilt],
            'pipes': [item._to_dict() for item in self._pipes_unbuilt],
            'profile_names': list(self._profile_name_to_index),
            'pipe_names': list(self._pipe_name_to_index),
        }

    def to_dict(self):
        """Serialize aperture data for this environment and all its lines."""
        return {
            'environment': self._to_dict(),
            'lines': {
                name: line.aperture._to_dict()
                for name, line in self.env.lines.items()
            },
        }

    def to_json(self, filename):
        """Write aperture data for this environment and all its lines to JSON."""
        json_dump(self.to_dict(), filename)

    def load_dict(self, data, context=None):
        """Replace the current data with a packed representation."""
        for line in list(self.env._lines_weakrefs):
            line.aperture.discard_model()

        self._struct = ApertureEnvStruct(**data, _context=context)
        self._profile_name_to_index = {
            name: index
            for index, name in enumerate(self._struct.profile_names)
        }
        self._pipe_name_to_index = {
            name: index
            for index, name in enumerate(self._struct.pipe_names)
        }
        self._profiles_unbuilt = None
        self._pipes_unbuilt = None

        return self
