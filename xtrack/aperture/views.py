from __future__ import annotations

import hashlib
import re
from typing import Literal

import numpy as np

from xtrack.aperture.structures import (
    ApertureModel, FloatType, Pipe, PipePosition, Profile, ProfilePosition, Racetrack, ShapeTypes,
)
from xtrack.aperture.transform import (
    Frame, arc_matrix, Transform, matrix_to_transform, transform_matrix, poly2d_to_homogeneous,
)

DTypeFloat = np.dtype[FloatType._dtype]
NDArrayNx2 = np.ndarray[tuple[int, Literal[2]], DTypeFloat]
NDArrayNxMx2 = np.ndarray[tuple[int, int, Literal[2]], DTypeFloat]
HomogenousMatrix = np.ndarray[tuple[Literal[4], Literal[4]], DTypeFloat]
HomogenousMatrices = np.ndarray[tuple[int, Literal[4], Literal[4]], DTypeFloat]


def _hashed_color(name: str, palette: list[str]) -> str:
    if not palette:
        return 'C0'
    digest = hashlib.sha1(name.encode('utf-8')).digest()
    return palette[int.from_bytes(digest[:4], 'little') % len(palette)]


def _deduplicate_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label and label not in unique:
            unique[label] = handle
    if unique:
        ax.legend(unique.values(), unique.keys())


class ProfileView:
    __slots__ = ('_model', '_index')

    def __init__(self, model: ApertureModel, index: int):
        self._model = model
        self._index = index

    def __repr__(self):
        return f'<ProfileView {self.name!r}: {self.raw!r}>'

    @property
    def raw(self) -> Profile:
        return self._model.profiles[self._index]

    @property
    def name(self) -> str:
        return self._model.profile_names[self._index]

    @property
    def shape(self) -> ShapeTypes:
        return self.raw.shape

    @shape.setter
    def shape(self, shape: ShapeTypes):
        self.raw.shape = shape

    @property
    def tol_r(self) -> float:
        return self.raw.tol_r

    @tol_r.setter
    def tol_r(self, tol_r: float):
        self.raw.tol_r = tol_r

    @property
    def tol_x(self) -> float:
        return self.raw.tol_x

    @tol_x.setter
    def tol_x(self, tol_x: float):
        self.raw.tol_x = tol_x

    @property
    def tol_y(self) -> float:
        return self.raw.tol_y

    @tol_y.setter
    def tol_y(self, tol_y: float):
        self.raw.tol_y = tol_y

    def plot(self, len_points=128, ax=None):
        ax = self.raw.plot(len_points=len_points, ax=ax, c='black', label=type(self.shape).__name__)

        if self.tol_x or self.tol_y or self.tol_r:
            tol_rt = Racetrack(
                half_width=self.tol_x + self.tol_r,
                half_height=self.tol_y + self.tol_r,
                half_major=self.tol_r,
                half_minor=self.tol_r,
            )
            Profile(shape=tol_rt).plot(len_points=len_points, ax=ax, c='black', linestyle='--', label='Tolerances')

        ax.set_title(f'Profile {self.name}')
        ax.legend()
        return ax


class ProfilesView:
    __slots__ = ('_model',)

    def __init__(self, model: ApertureModel):
        self._model = model

    def __repr__(self):
        count = len(self)
        profiles_str = 'profile' if count == 1 else 'profiles'
        return f'<ProfilesView: {count} {profiles_str}>'

    def __getitem__(self, item: str | int):
        if isinstance(item, str):
            item = self._model.profile_names.index(item)

        return ProfileView(self._model, item)

    def __len__(self) -> int:
        return len(self._model.profiles)

    def __iter__(self):
        for ii in range(len(self)):
            yield self[ii]

    def keys(self):
        return self._model.profile_names

    def values(self):
        return list(self)

    def items(self):
        return zip(self.keys(), self.values())

    def search(self, pattern: str):
        regex = re.compile(pattern)
        matches = [name for name in self.keys() if regex.match(name)]
        return matches


class PipePositionView:
    __slots__ = ('_model', '_index')

    def __init__(self, model: ApertureModel, index: int):
        self._model = model
        self._index = index

    def __repr__(self):
        shift_and_rot = self.get_transform()
        non_zero_transform = {
            field: value
            for field in shift_and_rot._fields
            if (value := getattr(shift_and_rot, field))
        }
        transform = ''.join(f', {k} = {v}' for k, v in non_zero_transform.items())

        return (f'<PipePositionView {self.name!r}: {self.pipe.name!r}, '
                f'survey_ref = {self.survey_reference_name!r}{transform}>')

    @property
    def raw(self) -> PipePosition:
        return self._model.pipe_positions[self._index]

    @property
    def name(self) -> str:
        return self._model.pipe_position_names[self._index]

    @property
    def pipe_index(self) -> int:
        return self.raw.pipe_index

    @pipe_index.setter
    def pipe_index(self, pipe_index: int):
        self.raw.pipe_index = pipe_index

    @property
    def pipe(self) -> PipeView:
        return PipeView(self._model, self.pipe_index)

    @property
    def survey_reference_name(self) -> str:
        return self.raw.survey_reference_name

    @survey_reference_name.setter
    def survey_reference_name(self, survey_reference_name: str):
        self.raw.survey_reference_name = survey_reference_name

    @property
    def survey_index(self) -> int:
        return self.raw.survey_index

    @survey_index.setter
    def survey_index(self, survey_index: int):
        self.raw.survey_index = survey_index

    @property
    def transformation(self) -> np.ndarray:
        return self.raw.transformation.to_nplike()

    @transformation.setter
    def transformation(self, transformation):
        self.raw.transformation = transformation

    def get_transform(self) -> Transform:
        return matrix_to_transform(self.transformation)

    def set_transform(self, transform: Transform):
        matrix = transform_matrix(**transform._asdict())
        self.raw.transformation.to_nplike()[:] = matrix

    @property
    def shift_x(self) -> float:
        return self.get_transform().shift_x

    @shift_x.setter
    def shift_x(self, value: float):
        as_dict = self.get_transform()._asdict()
        as_dict['shift_x'] = value
        self.set_transform(Transform(**as_dict))

    @property
    def shift_y(self) -> float:
        return self.get_transform().shift_y

    @shift_y.setter
    def shift_y(self, value: float):
        as_dict = self.get_transform()._asdict()
        as_dict['shift_y'] = value
        self.set_transform(Transform(**as_dict))

    @property
    def shift_z(self) -> float:
        return self.get_transform().shift_z

    @shift_z.setter
    def shift_z(self, value: float):
        as_dict = self.get_transform()._asdict()
        as_dict['shift_z'] = value
        self.set_transform(Transform(**as_dict))

    @property
    def rot_y_rad(self) -> float:
        return self.get_transform().rot_y_rad

    @rot_y_rad.setter
    def rot_y_rad(self, value: float):
        as_dict = self.get_transform()._asdict()
        as_dict['rot_y_rad'] = value
        self.set_transform(Transform(**as_dict))

    @property
    def rot_x_rad(self) -> float:
        return self.get_transform().rot_x_rad

    @rot_x_rad.setter
    def rot_x_rad(self, value: float):
        as_dict = self.get_transform()._asdict()
        as_dict['rot_x_rad'] = value
        self.set_transform(Transform(**as_dict))

    @property
    def rot_z_rad(self) -> float:
        return self.get_transform().rot_z_rad

    @rot_z_rad.setter
    def rot_z_rad(self, value: float):
        as_dict = self.get_transform()._asdict()
        as_dict['rot_z_rad'] = value
        self.set_transform(Transform(**as_dict))


class PipePositionsView:
    __slots__ = ('_model',)

    def __init__(self, model: ApertureModel):
        self._model = model

    def __repr__(self):
        count = len(self)
        positions_str = 'pipe position' if count == 1 else 'pipe positions'
        return f'<PipePositionsView: {count} {positions_str}>'

    def __getitem__(self, item: str | int):
        if isinstance(item, str):
            item = self._model.pipe_position_names.index(item)

        return PipePositionView(self._model, item)

    def __len__(self) -> int:
        return len(self._model.pipe_positions)

    def __iter__(self):
        for ii in range(len(self)):
            yield self[ii]

    def keys(self):
        return self._model.pipe_position_names

    def values(self):
        return list(self)

    def items(self):
        return zip(self.keys(), self.values())

    def search(self, pattern: str):
        regex = re.compile(pattern)
        matches = [name for name in self.keys() if regex.match(name)]
        return matches


class ProfilePositionView:
    __slots__ = ('_model', '_pipe_index', '_position_index')

    def __init__(self, model: ApertureModel, pipe_index: int, position_index: int):
        self._model = model
        self._pipe_index = pipe_index
        self._position_index = position_index

    def __repr__(self):
        return f'<ProfilePositionView profile={self.profile.name!r}, shift_s={self.shift_s}>'

    @property
    def raw(self) -> ProfilePosition:
        return self._model.pipes[self._pipe_index].positions[self._position_index]

    @property
    def profile_index(self) -> int:
        return self.raw.profile_index

    @profile_index.setter
    def profile_index(self, profile_index: int):
        self.raw.profile_index = profile_index

    @property
    def profile(self) -> ProfileView:
        return ProfileView(self._model, self.profile_index)

    @property
    def shift_s(self) -> float:
        return self.raw.shift_s

    @shift_s.setter
    def shift_s(self, shift_s: float):
        self.raw.shift_s = shift_s

    @property
    def shift_x(self) -> float:
        return self.raw.shift_x

    @shift_x.setter
    def shift_x(self, shift_x: float):
        self.raw.shift_x = shift_x

    @property
    def shift_y(self) -> float:
        return self.raw.shift_y

    @shift_y.setter
    def shift_y(self, shift_y: float):
        self.raw.shift_y = shift_y

    @property
    def rot_x_rad(self) -> float:
        return self.raw.rot_x_rad

    @rot_x_rad.setter
    def rot_x_rad(self, rot_x_rad: float):
        self.raw.rot_x_rad = rot_x_rad

    @property
    def rot_y_rad(self) -> float:
        return self.raw.rot_y_rad

    @rot_y_rad.setter
    def rot_y_rad(self, rot_y_rad: float):
        self.raw.rot_y_rad = rot_y_rad

    @property
    def rot_s_rad(self) -> float:
        return self.raw.rot_s_rad

    @rot_s_rad.setter
    def rot_s_rad(self, rot_s_rad: float):
        self.raw.rot_s_rad = rot_s_rad

    def get_transform(self, frame: Frame = 'curved'):
        if frame not in ('curved', 'straight'):
            return ValueError('Frame must be "curved" or "straight"')

        t_x, t_y, t_s = self.shift_x, self.shift_y, self.shift_s
        rot_y, rot_x, rot_s = self.rot_y_rad, self.rot_x_rad, self.rot_s_rad

        if frame == 'straight':
            return transform_matrix(t_x, t_y, t_s, rot_y, rot_x, rot_s)

        curvature = self._model.pipes[self._pipe_index].curvature
        local = transform_matrix(t_x, t_y, 0, rot_y, rot_x, rot_s)
        arc = arc_matrix(length=t_s, angle=curvature * t_s, tilt=0)
        return arc @ local


class PipeView:
    __slots__ = ('_model', '_index')

    def __init__(self, model: ApertureModel, index: int):
        self._model = model
        self._index = index

    def __repr__(self):
        curved_str = f', curvature = {self.curvature}' if self.curvature else ''
        len_positions = len(self)
        return f'<PipeView {self.name!r}: {len_positions} profiles{curved_str}>'

    def __getitem__(self, item: int):
        if isinstance(item, slice):
            indices = range(*item.indices(len(self)))
            return [ProfilePositionView(self._model, self._index, ii) for ii in indices]

        return ProfilePositionView(self._model, self._index, item)

    @property
    def raw(self) -> Pipe:
        return self._model.pipes[self._index]

    @property
    def name(self) -> str:
        return self._model.pipe_names[self._index]

    @property
    def curvature(self) -> float:
        return self.raw.curvature

    def __len__(self) -> int:
        return len(self.raw.positions)

    def __iter__(self):
        for ii in range(len(self)):
            yield self[ii]

    @property
    def length(self):
        s_end = self.raw.positions[len(self) - 1].shift_s
        s_start = self.raw.positions[0].shift_s
        return s_end - s_start

    @property
    def angle(self):
        return self.length * self.curvature

    @curvature.setter
    def curvature(self, curvature: float):
        self.raw.curvature = curvature

    def values(self):
        return list(self)

    def build_polygons_3d(self, frame: Frame = 'curved', len_points=128):
        def _poly_in_pipe(profile_pos_view):
            poly_2d = profile_pos_view.profile.raw.build_polygon(len_points)
            poly_hom = poly2d_to_homogeneous(poly_2d)
            profile_position_matrix = profile_pos_view.get_transform(frame=frame)
            return profile_position_matrix @ poly_hom

        polygons = np.array([_poly_in_pipe(prof_pos_view) for prof_pos_view in self])
        return polygons

    def plot_projection(
        self,
        plane: Literal['zx', 'zy', 'sx', 'sy'] = 'zx',
        len_points=32,
        transform: np.ndarray = np.identity(4),
        ax=None,
        colour: Literal['profile', 'pipe'] = 'profile',
        legend: bool = True,
    ):
        if colour not in ('profile', 'pipe'):
            raise ValueError("colour must be either 'profile' or 'pipe'")

        frame = {'z': 'curved', 's': 'straight'}[plane[0]]

        # Plot setup
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        ax.set_aspect('equal')
        ax.set_title(f'{self.name}')
        palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])

        # Plot the projected polygons
        polys = self.build_polygons_3d(frame=frame, len_points=len_points)  # noqa
        for poly, prof_view in zip(polys, self):
            label = prof_view.profile.name if colour == 'profile' else ''
            line_colour = _hashed_color(prof_view.profile.name if colour == 'profile' else self.name, palette)
            poly_trans = transform @ poly
            xs, ys, zs = poly_trans[:3]
            ax.plot(zs, {'x': xs, 'y': ys}[plane[1]], label=label, color=line_colour)

        # Plot the pipe axis
        min_s, max_s = self[0].shift_s, self[len(self) - 1].shift_s
        ss = np.linspace(min_s, max_s, len_points)
        h = self.curvature if frame == 'curved' else 0
        points = np.array([transform @ arc_matrix(length=s, angle=h * s, tilt=0) for s in ss])
        coords_z = points[:, 2, 3]
        coords_xy = points[:, 'xy'.index(plane[1]), 3]
        axis_colour = _hashed_color(self.name, palette)
        ax.plot(coords_z, coords_xy, color=axis_colour, linestyle='--', label=self.name)

        # Plot labels
        ax.set_xlabel(f'{plane[0]} [m]')
        ax.set_ylabel(f'{plane[1]} [m]')
        if legend:
            _deduplicate_legend(ax)

    def plot_3d(self, frame: Frame = 'curved', len_points=32, ax=None):
        if frame not in ('curved', 'straight'):
            return ValueError('Frame must be "curved" or "straight"')

        # Plot setup
        import matplotlib.pyplot as plt
        ax = ax or plt.figure(figsize=(10, 8)).add_subplot(111, projection='3d')
        ax.set_title(f'{self.name}')
        palette = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0'])

        # Plot the profile polygons
        polys = self.build_polygons_3d(frame=frame, len_points=len_points)
        all_points = []
        for poly, prof_view in zip(polys, self):
            points = np.asarray(poly[:3], dtype=float).T
            points = np.vstack([points, points[0]])
            all_points.append(points)
            colour = _hashed_color(prof_view.profile.name, palette)
            ax.plot(points[:, 2], points[:, 0], points[:, 1], alpha=0.85, label=prof_view.profile.name, color=colour)

        # Plot the pipe axis
        min_s, max_s = self[0].shift_s, self[len(self) - 1].shift_s
        ss = np.linspace(min_s, max_s, len_points)
        h = self.curvature if frame == 'curved' else 0
        axis_points = np.array([arc_matrix(length=s, angle=h * s, tilt=0)[:3, 3] for s in ss])
        axis_colour = _hashed_color(self.name, palette)
        ax.plot(axis_points[:, 2], axis_points[:, 0], axis_points[:, 1], color=axis_colour, linestyle='--', label=self.name)

        # Plot labels and set axes scaling
        ax.set_xlabel(f"{'z' if frame == 'curved' else 's'} [m]")
        ax.set_ylabel('x [m]')
        ax.set_zlabel('y [m]')
        if all_points:
            stacked = np.vstack(all_points)
            mins = stacked.min(axis=0)
            maxs = stacked.max(axis=0)
            centres = 0.5 * (mins + maxs)
            half_span = 0.5 * np.max(maxs - mins)
            ax.set_xlim(centres[2] - half_span, centres[2] + half_span)
            ax.set_ylim(centres[0] - half_span, centres[0] + half_span)
            ax.set_zlim(centres[1] - half_span, centres[1] + half_span)
        _deduplicate_legend(ax)
        ax.set_box_aspect((1, 1, 1))
        return ax


class PipesView:
    __slots__ = ('_model',)

    def __init__(self, model: ApertureModel):
        self._model = model

    def __repr__(self):
        count = len(self)
        pipes_str = 'pipe' if count == 1 else 'pipes'
        return f'<PipesView: {count} {pipes_str}>'

    def __getitem__(self, item: str | int) -> PipeView:
        if isinstance(item, str):
            item = self._model.pipe_names.index(item)

        return PipeView(self._model, item)

    def __len__(self) -> int:
        return len(self._model.pipes)

    def __iter__(self):
        for ii in range(len(self)):
            yield self[ii]

    def keys(self):
        return self._model.pipe_names

    def values(self):
        return list(self)

    def items(self):
        return zip(self.keys(), self.values())

    def search(self, pattern: str):
        regex = re.compile(pattern)
        matches = [name for name in self.keys() if regex.match(name)]
        return matches
