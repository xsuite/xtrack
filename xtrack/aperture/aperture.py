from __future__ import annotations

import re
import hashlib
from collections.abc import Collection
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, cast

import numpy as np

import xobjects as xo
from xdeps.table import Table
from xobjects.context import XContext
from xtrack import TwissInit, TwissTable
from xtrack.aperture.profile_converters import (
    LimitTypes, profile_from_limit_element, profile_from_madx_aperture
)
from xtrack.aperture.structures import (
    ApertureBounds, ApertureModel, BeamData, Circle, FloatType, Pipe,
    PipePosition, Profile, ProfilePolygons, ProfilePosition, Racetrack,
    Rectangle, RectEllipse, ShapeTypes, SurveyData, TwissData,
)
from xtrack.aperture.transform import (
    Frame, arc_matrix, Transform, matrix_to_transform, transform_matrix, poly2d_to_homogeneous
)
from xtrack.json import dump as json_dump
from xtrack.json import load as json_load
from xtrack.line import Line
from xtrack.progress_indicator import progress
from xtrack.survey import survey_relative_transform

DTypeFloat = np.dtype[FloatType._dtype]
NDArrayNx2 = np.ndarray[Tuple[int, Literal[2]], DTypeFloat]
NDArrayNxMx2 = np.ndarray[Tuple[int, int, Literal[2]], DTypeFloat]
HomogenousMatrix = np.ndarray[Tuple[Literal[4], Literal[4]], DTypeFloat]
HomogenousMatrices = np.ndarray[Tuple[int, Literal[4], Literal[4]], DTypeFloat]


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
        return self.raw.shape  # noqa: xobjects

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
        return self.raw.survey_reference_name  # noqa: xobjects

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


TypePositionView = PipePositionView
TypePositionsView = PipePositionsView
TypeView = PipeView
TypesView = PipesView


class Aperture:
    halo_params = {
        "emitx_norm": 3.5e-6,  # normalized emittance x
        "emity_norm": 3.5e-6,  # normalized emittance y
        "delta_rms": 0.0,  # rms energy spread
        "tol_co": 0.0,  # tolerance for closed orbit
        "tol_disp": 0.0,  # tolerance for normalized dispersion
        "tol_disp_ref": 1.8,  # tolerance for reference dispersion derivative
        "tol_disp_ref_beta": 170,  # tolerance for reference dispersion beta
        "tol_beta_beating": 1.0,  # tolerance for beta beating in sigma
        "halo_x": 6.0,  # n sigma of horizontal halo
        "halo_y": 6.0,  # n sigma of vertical halo
        "halo_r": 6.0,  # n sigma of 45 degree halo
        "halo_primary": 6.0,  # n sigma of primary halo
    }

    def __init__(
        self,
        line: Line,
        model: ApertureModel,
        num_profile_points: int = 128,
        halo_params: Optional[dict] = None,
        context: Optional[XContext] = None,
        s_tol=1e-6,
        _skip_validity_check=False,
    ):
        self.line = line
        self._model = model  # positioning of pipes in line frame
        self.halo_params = self.halo_params.copy()
        self.context = context or xo.ContextCpu()
        self.s_tol = s_tol

        self.survey = line.survey()

        # Add angle and rot_s_rad
        self.survey['angle'] = np.zeros_like(self.survey.s)
        self.survey['rot_s_rad'] = np.zeros_like(self.survey.s)
        self.survey['angle'][:-1] = line.attr['angle'] # shorter by one because survey has '_end_point'
        self.survey['rot_s_rad'][:-1] = line.attr['rot_s_rad'] # shorter by one because survey has '_end_point'

        if not _skip_validity_check:
            self._check_model_validity()

        self._survey_data = SurveyData.from_survey_table(self.survey, context=self.context)
        self.num_profile_points = num_profile_points

        self._aperture_bounds: Optional[ApertureBounds] = None
        self._profile_polygons: Optional[ProfilePolygons] = None
        self._build_aperture_bounds(check_validity=not _skip_validity_check)

        if halo_params is not None:
            self.halo_params.update(halo_params)

    @property
    def profiles(self) -> ProfilesView:
        return ProfilesView(self._model)

    @property
    def pipe_positions(self) -> PipePositionsView:
        return PipePositionsView(self._model)

    @property
    def pipes(self) -> PipesView:
        return PipesView(self._model)

    def to_json(self, filename):
        json = {
            'model': self._model.to_dict(),
            'halo_params': self.halo_params,
        }
        json_dump(json, filename)

    @classmethod
    def from_json(cls, filename, line, **kwargs):
        context = kwargs.pop('context', None)
        if context is None:
            context = getattr(line, '_context', None)
        json = json_load(filename)
        model = ApertureModel(**json['model'], _context=context)
        halo_params = json['halo_params']
        return cls(
            line=line,
            model=model,
            halo_params=halo_params,
            context=context,
            **kwargs,
        )

    @classmethod
    def from_line_with_madx_metadata(cls, line, include_offsets=True, context=None, **kwargs):
        survey = line.survey()
        survey_names = survey.name[:-1]  # _end_point is not an element
        name_to_sv_index = dict(zip(survey.name, range(len(survey))))
        layout_data = line.metadata['layout_data']

        if include_offsets:
            aperture_offsets = cls._get_per_pipe_madx_offsets(line.metadata.get('aperture_offsets', {}))
        else:
            aperture_offsets = {}

        name_iter_with_progress = progress(
            survey_names,
            desc="Building apertures",
            total=len(survey_names),
        )

        profiles = []
        pipes = []
        aperture_indices = {}
        pipe_positions_list = []
        pipe_position_names = []

        for element_name in name_iter_with_progress:
            element = line.element_dict[element_name]

            # Discard line name suffix to get the aperture name
            aper_name = cls._guess_original_mad_name(element_name)
            if aper_name not in layout_data:
                continue

            offset_data = aperture_offsets.get(aper_name, {})

            if offset_data:
                offsets_reversed = offset_data['reversed']
                x_dir = -1 if offsets_reversed else 1
                z_dir = -1 if offsets_reversed else 1
                survey_reference_name = offset_data['survey_ref']

                assert not line[survey_reference_name].isthick  # sanity check, not strictly needed for the maths

                # Transformation from the reference point to the element in the correct frame
                rel_survey_mat = survey_relative_transform(survey, survey_reference_name, element_name, reversed=offsets_reversed)
                survey_elem_index = survey.rows.get_index(element_name)

                # Compute the survey transformation of the element itself
                if not offsets_reversed:
                    elem_mat = survey_relative_transform(survey, survey_elem_index, survey_elem_index + 1, reversed=False)
                else:
                    elem_mat = survey_relative_transform(survey, survey_elem_index - 1, survey_elem_index, reversed=True)

                # Compute the length of the element projected onto the offset reference Z
                z_ref = rel_survey_mat[2, 3]
                z_next = (elem_mat @ rel_survey_mat)[2, 3]
                offset_frame_length = z_next - z_ref

                pipe_transform = transform_matrix(
                    shift_x=offset_data['x'] * x_dir,
                    shift_y=offset_data['y'],
                    shift_z=z_ref,
                )
            else:
                pipe_transform = np.identity(4)
                survey_reference_name = element_name

            if aper_name not in aperture_indices:
                element_metadata = layout_data[aper_name]

                if 'aperture' not in element_metadata:
                    continue

                shape_name, params, tols = element_metadata['aperture']
                shape = profile_from_madx_aperture(shape_name, params)

                if not shape:
                    # There is not really an aperture here, continue
                    continue

                if cls._is_broken_madx_aperture(shape):
                    continue

                tol_r, tol_x, tol_y = tols
                profile = Profile(shape=shape, tol_r=tol_r, tol_x=tol_x, tol_y=tol_y)

                assert len(pipes) == len(profiles)  # in MAD-X we will have just one pipe per profile

                aper_idx = len(pipes)

                if element.isthick and not offset_data:
                    # Place two profiles on either side of the element
                    position_entry = ProfilePosition(profile_index=aper_idx)
                    position_exit = ProfilePosition(profile_index=aper_idx, shift_s=element.length)
                    positions = [position_entry, position_exit]
                    # If no MAD-X offset data is present, the curvature follows
                    # the element
                    curvature = getattr(element, 'h', 0)
                elif offset_data and offset_frame_length > 1e-6:
                    # If MAD-X offset data is given, place profiles
                    # on the described parabola with 10cm resolution
                    positions = []

                    for s in np.linspace(0, offset_frame_length, max(2, int(offset_frame_length / 0.1))):
                        s_local = s
                        position = ProfilePosition(profile_index=aper_idx)
                        position.shift_s = s * z_dir
                        position.shift_x = (s_local * offset_data['dx'] + s_local**2 * offset_data['ddx']) * x_dir
                        position.shift_y = s_local * offset_data['dy'] + s_local**2 * offset_data['ddy']
                        positions.append(position)

                    positions = sorted(positions, key=lambda p: p.shift_s)

                    # If we have offset data, assume the pipe is straight
                    curvature = 0
                else:
                    # Place a single profile for a thin element
                    positions = [ProfilePosition(profile_index=aper_idx)]
                    curvature = 0

                aperture_indices[aper_name] = aper_idx

                pipe = Pipe(curvature=curvature, positions=positions)
                pipes.append(pipe)
                profiles.append(profile)

            pipe_position = PipePosition(
                pipe_index=aperture_indices[aper_name],
                survey_reference_name=survey_reference_name,
                survey_index=name_to_sv_index[survey_reference_name],
                transformation=pipe_transform,
            )
            pipe_positions_list.append(pipe_position)
            pipe_position_names.append(aper_name)

        aperture = cls._build_aperture_model(
            line=line,
            pipe_indices=aperture_indices,
            pipe_list=pipes,
            pipe_position_list=pipe_positions_list,
            pipe_position_names=pipe_position_names,
            profile_indices=aperture_indices,
            profile_list=profiles,
            context=context,
            **kwargs,
        )
        return aperture

    @classmethod
    def from_line_with_associated_apertures(cls, line, context=None, **kwargs):
        survey = line.survey()
        survey_names = survey.name[:-1]  # _end_point is not an element
        name_to_sv_index = dict(zip(survey.name, range(len(survey_names))))

        profiles = []
        pipes = []
        aperture_indices = {}
        pipe_positions_list = []
        pipe_position_names = []

        for survey_name in progress(survey_names, desc="Building aperture data", total=len(survey_names)):
            # Discard line name suffix to get the aperture name
            element = line[survey_name]
            aper_name = getattr(element, 'name_associated_aperture', None)

            if not aper_name:
                continue

            aper_element = line.element_dict[aper_name]

            if aper_name not in aperture_indices:
                profile, offset_x, offset_y = profile_from_limit_element(aper_element)

                assert len(pipes) == len(profiles)  # in Xsuite with associated apertures we will have just one pipe per profile

                aper_idx = len(pipes)
                aperture_indices[aper_name] = aper_idx

                profile_position = ProfilePosition(profile_index=aper_idx)
                profile_position.shift_x = offset_x
                profile_position.shift_y = offset_y

                if aper_element.transformations_active:
                    # Apply associated-aperture transforms in the local profile frame so
                    # they follow the curved pipe geometry when transported along it.
                    profile_position.shift_s = aper_element.shift_s
                    profile_position.shift_x += aper_element.shift_x
                    profile_position.shift_y += aper_element.shift_y
                    profile_position.rot_x_rad = aper_element.rot_x_rad
                    profile_position.rot_y_rad = aper_element.rot_y_rad
                    profile_position.rot_s_rad = aper_element.rot_s_rad_no_frame

                if element.isthick:
                    # Place two profiles on either side of the element
                    profile_position_start = profile_position
                    profile_position_end = profile_position.copy()
                    profile_position_end.shift_s += element.length
                    positions = [profile_position_start, profile_position_end]
                    curvature = getattr(element, 'h', 0)
                else:
                    # Place single profile at center of element
                    positions = [profile_position]
                    curvature = 0

                pipe = Pipe(curvature=curvature, positions=positions)
                pipes.append(pipe)

                profiles.append(Profile(shape=profile))

            # Apply element transformations to pipe position
            if element.transformations_active:
                # TODO: Need to correctly handle the situation where both the element and the aperture are misaligned.
                #  The matrix then needs to combine the two in a correct way. Curvature will probably complicate this
                #  even more.
                raise NotImplementedError('Aperture model not yet supported with element transformations.')

            pipe_position = PipePosition(
                pipe_index=aperture_indices[aper_name],
                survey_reference_name=survey_name,
                survey_index=name_to_sv_index[survey_name],
                transformation=np.identity(4),
            )
            pipe_positions_list.append(pipe_position)
            pipe_position_names.append(aper_name)

        aperture = cls._build_aperture_model(
            line=line,
            pipe_indices=aperture_indices,
            pipe_list=pipes,
            pipe_position_list=pipe_positions_list,
            pipe_position_names=pipe_position_names,
            profile_indices=aperture_indices,
            profile_list=profiles,
            context=context,
            **kwargs,
        )
        return aperture

    @classmethod
    def from_line_with_limits(cls, line, context=None, **kwargs):
        survey = line.survey()
        survey_names = survey.name[:-1]  # _end_point is not a limit
        name_to_sv_index = dict(zip(survey.name, range(len(survey_names))))

        profiles = []
        pipe_list = []
        indices = {}
        pipe_positions_list = []
        pipe_position_names = []

        aper_idx = 0

        for name in progress(survey_names, desc="Building aperture data", total=len(survey_names)):
            element = line[name]
            if not isinstance(element, LimitTypes):
                continue

            indices[name] = aper_idx
            profile, center_x, center_y = profile_from_limit_element(element)
            profiles.append(Profile(shape=profile))

            profile_position = ProfilePosition(profile_index=aper_idx)
            if element.transformations_active:
                profile_position.shift_s = element.shift_s
                profile_position.shift_x = element.shift_x
                profile_position.shift_y = element.shift_y
                # TODO: Is this really how it should be??
                profile_position.rot_s_rad = element.rot_s_rad_no_frame
                profile_position.rot_x_rad = element.rot_x_rad
                profile_position.rot_y_rad = element.rot_y_rad

            pipe = Pipe(curvature=0, positions=[profile_position])
            pipe_list.append(pipe)

            pipe_position = PipePosition(
                pipe_index=aper_idx,
                survey_reference_name=name,
                survey_index=name_to_sv_index[name],
                transformation=np.identity(4),
            )
            pipe_positions_list.append(pipe_position)
            pipe_position_names.append(name)

            aper_idx += 1

        aperture = cls._build_aperture_model(
            line=line,
            pipe_indices=indices,
            pipe_list=pipe_list,
            pipe_position_list=pipe_positions_list,
            pipe_position_names=pipe_position_names,
            profile_indices=indices,
            profile_list=profiles,
            context=context,
            **kwargs,
        )
        return aperture

    def polygon_for_profile(self, profile: Profile, num_points: int) -> NDArrayNx2:
        return profile.build_polygon(len_points=num_points)

    @classmethod
    def _build_aperture_model(
            cls,
            line: Line,
            pipe_indices: Dict[str, int],
            pipe_list: List[Pipe],
            pipe_position_list: List[PipePosition],
            pipe_position_names: List[str],
            profile_indices: Dict[str, int],
            profile_list: List[ShapeTypes],
            context: XContext,
            **kwargs,
    ) -> 'Aperture':
        """Build the Aperture class and its comprising xobjects.

        Parameters
        ----------
        line
            The line for which the aperture model is built.
        pipe_indices
            A mapping between the name of an aperture pipe and its index in ``pipe_list``.
        pipe_list
            List of aperture pipes featured in the model.
        pipe_position_list
            List of aperture pipe positions that define the model.
        pipe_position_names
            Names of all aperture pipe positions in ``pipe_position_list`` order.
        profile_indices
            A mapping between the name of an aperture pipe and its index in ``profile_list``.
        profile_list
            List of all profiles featured in the model. The order must be consistent with the indices used inside
            each of the pipe definitions in ``pipe_list``.
        kwargs
            Further parameters to be passed to the initialiser of `Aperture`.
        """
        if list(pipe_indices.values()) != list(range(len(pipe_list))):
            raise ValueError('Expected pipe_indices to be ordered by index')

        if list(profile_indices.values()) != list(range(len(profile_indices))):
            raise ValueError('Expected profile_indices to be ordered by index')

        context = context or xo.ContextCpu()

        model = ApertureModel(
            pipe_positions=pipe_position_list,
            pipes=pipe_list,
            profiles=profile_list,
            pipe_names=list(pipe_indices.keys()),
            pipe_position_names=pipe_position_names,
            profile_names=list(profile_indices.keys()),
            _context=context,
        )

        aperture = cls(
            line=line,
            model=model,
            context=context,
            **kwargs,
        )

        return aperture

    def get_aperture_sigmas_at_element(
            self,
            element_name: str,
            resolution: Optional[float] = None,
            twiss: Optional[TwissTable] = None,
            **kwargs,
    ) -> Tuple[Table, TwissTable]:
        """Compute the maximum number of sigmas at which the beam fits in the aperture at element ``element_name``.

        Parameters
        ----------
        elment_name
            The name of the element at which the sigmas should be computed.
        resolution
            The desired resolution, in meters along s, at which the sigmas should be computed. If not provided only the
            values at the entry and exit will be output.
        twiss
            Optionally provided twiss table from which to derive the initial beam parameters at the element.
        **kwargs
            Other parameters to be forwarded to :meth:`get_aperture_sigmas_at_s`.

        Returns
        -------
        See :meth:`get_aperture_sigmas_at_s`.
        """
        s_positions = self._get_cuts_at_element(element_name, resolution)
        twiss_init = twiss.get_twiss_init(at_element=element_name) if twiss else None
        return self.get_aperture_sigmas_at_s(s_positions, twiss_init, **kwargs)

    def get_aperture_sigmas_at_s(
            self,
            s_positions: Iterable[float],
            twiss_init: Optional[TwissInit] = None,
            method: Literal['bisection', 'rays', 'exact'] = 'rays',
            envelopes_num_points: int = 36,
            num_rays: int = 32,
            output_max_envelopes: bool = False,
            output_cross_sections: bool = False,
    ) -> Tuple[Table, TwissTable]:
        """Compute the maximum number of sigmas at which the beam fits in the aperture at element ``element_name``.

        Parameters
        ----------
        s_positions
            List of s positions at which to calculate the sigmas.
        twiss_init
            Optionally provided initial twiss conditions.
        method
            A method to use for the computation:
            - 'rays' - the aperture sigma is estimated from sampled rays and the minimum over the sampled directions
              is returned (faster method, O(R) where R is the number of rays)
            - 'exact' - the aperture sigma is estimated from sampled points on the halo racetrack, at which new sample
              rays are emitted to compare the local directional sigma to the aperture (O(R^2), where R is the number
              of rays).
            - 'bisection' - the smallest number of sigmas for the beam to fit in the aperture is computed by bisecting
              on a polygon-inside-polygon problem (slower method, O(EAK), where E is the number of envelope points,
              A is the number of aperture points, and K is the number of bisection steps; currently K <= 25, this
              depends on the tolerance and search space set in ``beam_aperture.h``).
        envelopes_num_points:
            Number of points to use when discretising the beam cross-section.
        num_rays:
            Only for methods `rays` and `exact`: number of evenly-spaced ray directions to sample in [0, 2 * pi).
        output_max_envelopes:
            If true, output beam-envelope polygons at the computed `n1`.
        output_cross_sections:
            If true, output interpolated aperture cross-sections.

        Returns
        -------
        A two-tuple ``(table, sliced_twiss)``, where:
        - ``table`` is an :class:`xdeps.table.Table` with columns ``s`` and ``n1``.
        - if ``output_cross_sections`` is true, ``table`` also contains ``cross_section``.
        - if ``output_max_envelopes`` is true, ``table`` also contains ``envelope``.
        - ``sliced_twiss`` is the twiss table computed as part of the calculation.
        """
        sliced_twiss = self._sliced_twiss_at_s(s_positions=s_positions, twiss_init=twiss_init)
        num_slices = len(sliced_twiss.s)
        twiss_at_s = TwissData.from_twiss_table(self.line.particle_ref, sliced_twiss)
        survey_at_s = self._survey_data.resample(twiss_at_s.s)
        beam_data = BeamData(**self.halo_params)

        if output_cross_sections:
            interpolated_points = np.zeros(shape=(num_slices, self.num_profile_points, 2), dtype=FloatType._dtype)
        else:
            interpolated_points = None

        if output_max_envelopes:
            envelope_at_max_sigma = np.zeros(shape=(num_slices, envelopes_num_points, 2), dtype=FloatType._dtype)
        else:
            envelope_at_max_sigma = None

        if method == 'bisection':
            sigmas = np.zeros(num_slices, dtype=FloatType._dtype)

            self._model.get_max_aperture_sigma_bisection(
                survey=self._survey_data,
                profile_polygons=self._profile_polygons,
                aperture_bounds=self._aperture_bounds,
                twiss_at_s=twiss_at_s,
                survey_at_s=survey_at_s,
                beam_data=beam_data,
                out_interpolated_apertures=interpolated_points,
                envelope_num_points=envelopes_num_points,
                out_envelope_at_max_sigma=envelope_at_max_sigma,
                sigmas=sigmas,
            )
            n1s = sigmas
        elif method == 'rays':
            ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False, dtype=FloatType._dtype)
            ray_sigmas = np.zeros((num_slices, num_rays), dtype=FloatType._dtype)

            self._model.get_max_aperture_sigma_rays(
                survey=self._survey_data,
                profile_polygons=self._profile_polygons,
                aperture_bounds=self._aperture_bounds,
                twiss_at_s=twiss_at_s,
                survey_at_s=survey_at_s,
                beam_data=beam_data,
                out_interpolated_apertures=interpolated_points,
                envelope_num_points=envelopes_num_points,
                out_envelope_at_max_sigma=envelope_at_max_sigma,
                ray_angles=ray_angles,
                num_ray_angles=num_rays,
                sigmas=ray_sigmas,

            )
            n1s = np.min(ray_sigmas, axis=1)
        elif method == 'exact':
            ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False, dtype=FloatType._dtype)
            sigmas = np.zeros(num_slices, dtype=FloatType._dtype)

            self._model.get_max_aperture_sigma_exact(
                survey=self._survey_data,
                profile_polygons=self._profile_polygons,
                aperture_bounds=self._aperture_bounds,
                twiss_at_s=twiss_at_s,
                survey_at_s=survey_at_s,
                beam_data=beam_data,
                out_interpolated_apertures=interpolated_points,
                envelope_num_points=envelopes_num_points,
                out_envelope_at_max_sigma=envelope_at_max_sigma,
                ray_angles=ray_angles,
                num_ray_angles=num_rays,
                sigmas=sigmas,
            )
            n1s = sigmas
        else:
            raise NotImplementedError(f"Method `{method}` for getting aperture sigmas is unknown.")

        table_data = {
            'index': np.arange(len(sliced_twiss)),
            's': sliced_twiss.s,
            'n1': n1s,
        }
        if output_cross_sections:
            table_data['cross_section'] = interpolated_points
        if output_max_envelopes:
            table_data['envelope'] = envelope_at_max_sigma
        return Table(table_data, index='index'), sliced_twiss

    def get_hvd_aperture_sigmas_at_element(
            self,
            element_name: str,
            resolution: Optional[float] = None,
            twiss: Optional[TwissTable] = None,
    ) -> Tuple[np.ndarray, TwissTable, np.ndarray]:
        """Compute horizontal, vertical and horizontal max aperture sigmas at element ``element_name``.

        Parameters
        ----------
        elment_name
            The name of the element at which the sigmas should be computed.
        resolution
            The desired resolution, in meters along s, at which the sigmas should be computed. If not provided only the
            values at the entry and exit will be output.
        twiss
            Optionally provided twiss table from which to derive the initial beam parameters at the element.
        **kwargs
            Other parameters to be forwarded to :meth:`get_hvd_aperture_sigmas_at_s`.

        Returns
        -------
        See :meth:`get_hvd_aperture_sigmas_at_s`.
        """
        s_positions = self._get_cuts_at_element(element_name, resolution)
        twiss_init = twiss.get_twiss_init(at_element=element_name) if twiss else None
        return self.get_hvd_aperture_sigmas_at_s(s_positions=s_positions, twiss_init=twiss_init)

    def get_hvd_aperture_sigmas_at_s(
            self,
            s_positions: Iterable[float],
            twiss_init: Optional[TwissInit] = None,
    ) -> Tuple[np.ndarray, TwissTable, np.ndarray]:
        """Compute horizontal, vertical and horizontal max aperture sigmas.

        Parameters
        ----------
        s_positions : Iterable[float]
            Locations at which to compute the desired quantities.
        twiss_init : TwissInit, optional
            Initial conditions for the twiss.

        Returns
        -------
        A three-tuple ``(sigmas, sliced_twiss, aperture_polygons)``:
        - ```sigmas`` is an array of shape `(len(s_positions), 3)`, containing the maximum number of sigmas fitting in
          the aperture in the horizontal, vertical and horizontal directions at each s-position.
        - ``sliced_twiss`` is the twiss table computed as part of the calculation
        - ``aperture_polygons`` are the aperture cross-sections at each of the ``s_positions``: a numpy array of shape
          ``(len(s_positions), cross_sections_num_points, 2)``.
        """
        sliced_twiss = self._sliced_twiss_at_s(s_positions=s_positions, twiss_init=twiss_init)
        num_slices = len(sliced_twiss.s)

        twiss_at_s = TwissData.from_twiss_table(self.line.particle_ref, sliced_twiss)
        survey_at_s = self._survey_data.resample(twiss_at_s.s)

        beam_data = BeamData(**self.halo_params)
        interpolated_points = np.zeros(shape=(num_slices, self.num_profile_points, 2), dtype=FloatType._dtype)

        ray_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False, dtype=FloatType._dtype)
        ray_sigmas = np.zeros((num_slices, 8), dtype=FloatType._dtype)

        self._model.get_max_aperture_sigma_rays(
            survey=self._survey_data,
            profile_polygons=self._profile_polygons,
            aperture_bounds=self._aperture_bounds,
            twiss_at_s=twiss_at_s,
            survey_at_s=survey_at_s,
            beam_data=beam_data,
            out_interpolated_apertures=interpolated_points,
            envelope_num_points=0,
            out_envelope_at_max_sigma=None,
            ray_angles=ray_angles,
            num_ray_angles=8,
            sigmas=ray_sigmas,

        )

        sigmas_h = np.minimum(ray_sigmas[:, 0], ray_sigmas[:, 4])
        sigmas_v = np.minimum(ray_sigmas[:, 2], ray_sigmas[:, 6])
        sigmas_d = np.minimum.reduce([ray_sigmas[:, 1], ray_sigmas[:, 3], ray_sigmas[:, 5], ray_sigmas[:, 7]])
        ray_sigmas = np.c_[sigmas_h, sigmas_v, sigmas_d]

        return ray_sigmas, sliced_twiss, interpolated_points

    def get_envelope_at_element(
            self,
            element_name: str,
            sigmas: float,
            resolution: Optional[float] = None,
            twiss: Optional[TwissTable] = None,
            **kwargs,
    ) -> Tuple[np.ndarray, TwissTable]:
        """Compute beam-envelope polygons at the cuts of ``element_name`` for a fixed sigma value.

        Parameters
        ----------
        element_name
            The name of the element at which the envelope should be computed.
        sigmas
            The beam size, in sigmas, at which the envelope should be evaluated.
        resolution
            The desired resolution, in meters along s, at which the envelope should be computed. If not provided only
            the values at the entry and exit will be output.
        twiss
            Optionally provided twiss table from which to derive the initial beam parameters at the element.
        **kwargs
            Other parameters to be forwarded to ``Aperture.get_envelope_at_s``.

        Returns
        -------
        A two-tuple ``(envelopes, sliced_twiss)``, where:
        - ``envelopes`` are the beam cross-section polygons at the requested sigma: a numpy array of shape
          ``(num_cuts, envelopes_num_points, 2)``.
        - ``sliced_twiss`` is the twiss table computed as part of the calculation.
        """
        s_positions = self._get_cuts_at_element(element_name, resolution)
        twiss_init = twiss.get_twiss_init(at_element=element_name) if twiss else None
        return self.get_envelope_at_s(s_positions, sigmas, twiss_init, **kwargs)

    def get_envelope_at_s(
            self,
            s_positions: Iterable[float],
            sigmas: float,
            twiss_init: Optional[TwissInit] = None,
            envelopes_num_points: int = 128,
            include_aper_tols: bool = True,
    ) -> Tuple[np.ndarray, TwissTable]:
        """Compute beam-envelope polygons at the requested ``s_positions`` for a fixed sigma value.

        Parameters
        ----------
        s_positions
            List of s positions at which to compute the envelope.
        sigmas
            The beam size, in sigmas, at which the envelope should be evaluated.
        twiss_init
            Optionally provided initial twiss conditions.
        envelopes_num_points
            Number of points to use when discretising the beam cross-section polygon.
        include_aper_tols
            If true, include the aperture mechanical tolerances associated with the active profile at each ``s``.

        Returns
        -------
        A two-tuple ``(envelopes, sliced_twiss)``, where:
        - ``envelopes`` are the beam cross-section polygons at the requested sigma: a numpy array of shape
          ``(len(s_positions), envelopes_num_points, 2)``.
        - ``sliced_twiss`` is the twiss table computed as part of the calculation.
        """
        sliced_twiss = self._sliced_twiss_at_s(s_positions=s_positions, twiss_init=twiss_init)
        num_slices = len(sliced_twiss.s)
        twiss_at_s = TwissData.from_twiss_table(self.line.particle_ref, sliced_twiss)
        beam_data = BeamData(**self.halo_params)

        envelopes = np.zeros(shape=(num_slices, envelopes_num_points, 2), dtype=FloatType._dtype)

        self._model.get_beam_envelopes_at_sigma(
            aperture_bounds=self._aperture_bounds,
            twiss_at_s=twiss_at_s,
            beam_data=beam_data,
            sigmas=sigmas,
            envelope_num_points=envelopes_num_points,
            include_aper_tols=int(include_aper_tols),
            out_envelope=envelopes,
        )

        return envelopes, sliced_twiss

    def poses_at_s(self, s_positions: Collection[float]) -> HomogenousMatrices:
        """Return a local coordinate system (each represented by a homogeneous matrix) at all ``s_positions``."""
        sv_resampled = self._survey_data.resample(s_positions)
        return sv_resampled.pose.to_nparray()

    def cross_sections_at_element(
        self,
        element_name: str,
        resolution: Optional[float],
        extents: bool = False,
    ) -> Table:
        s_positions = self._get_cuts_at_element(element_name, resolution)
        return self.cross_sections_at_s(s_positions, extents=extents)

    def cross_sections_at_s(
        self,
        s_positions: Collection[float],
        extents: bool = False,
    ) -> Table:
        s_positions = np.array(s_positions, dtype=FloatType._dtype)
        sv_resampled = self._survey_data.resample(s_positions)

        cross_sections = np.zeros(shape=(len(s_positions), self.num_profile_points, 2), dtype=FloatType._dtype)
        tol_r = np.zeros(shape=len(s_positions), dtype=FloatType._dtype)
        tol_x = np.zeros(shape=len(s_positions), dtype=FloatType._dtype)
        tol_y = np.zeros(shape=len(s_positions), dtype=FloatType._dtype)

        self._model.cross_sections_at_s(
            survey_at_s=sv_resampled,
            profile_polygons=self._profile_polygons,
            aperture_bounds=self._aperture_bounds,
            survey=self._survey_data,
            cross_sections=cross_sections,
            tol_r=tol_r,
            tol_x=tol_x,
            tol_y=tol_y,
        )
        poses = sv_resampled.pose.to_nparray()

        table_data = {
            'index': np.arange(len(s_positions)),
            's': s_positions,
            'pose': poses,
            'cross_section': cross_sections,
            'tol_r': tol_r,
            'tol_x': tol_x,
            'tol_y': tol_y,
        }
        if extents:
            table_data.update(self._axis_extents_for_cross_sections(cross_sections))
        return Table(table_data, index='index')

    def plot_extents(
        self,
        s_positions: Collection[float],
        sigmas: Optional[float] = None,
        twiss_init: Optional[TwissInit] = None,
        method: Literal['bisection', 'rays', 'exact'] = 'rays',
        envelopes_num_points: int = 64,
        include_aper_tols: bool = False,
        plot_s_positions: Optional[Collection[float]] = None,
        axs=None,
    ):
        s_positions = np.asarray(s_positions, dtype=FloatType._dtype)
        plot_s_positions = np.asarray(
            s_positions if plot_s_positions is None else plot_s_positions,
            dtype=FloatType._dtype,
        )

        order = np.argsort(s_positions)
        undo_order = np.empty_like(order)
        undo_order[order] = np.arange(len(order))
        s_sorted = s_positions[order]

        if sigmas is None:
            n1_table, _ = self.get_aperture_sigmas_at_s(
                s_positions=s_sorted,
                twiss_init=twiss_init,
                method=method,
            )
            sigmas = float(np.min(n1_table.n1))

        envelopes, _ = self.get_envelope_at_s(
            s_positions=s_sorted,
            sigmas=sigmas,
            twiss_init=twiss_init,
            envelopes_num_points=envelopes_num_points,
            include_aper_tols=include_aper_tols,
        )
        sections_table = self.cross_sections_at_s(s_sorted, extents=True)

        min_envel_x = np.min(envelopes[:, :, 0], axis=1)[undo_order]
        max_envel_x = np.max(envelopes[:, :, 0], axis=1)[undo_order]
        min_envel_y = np.min(envelopes[:, :, 1], axis=1)[undo_order]
        max_envel_y = np.max(envelopes[:, :, 1], axis=1)[undo_order]

        min_aper_x = np.asarray(sections_table.min_x, dtype=FloatType._dtype)[undo_order]
        max_aper_x = np.asarray(sections_table.max_x, dtype=FloatType._dtype)[undo_order]
        min_aper_y = np.asarray(sections_table.min_y, dtype=FloatType._dtype)[undo_order]
        max_aper_y = np.asarray(sections_table.max_y, dtype=FloatType._dtype)[undo_order]

        tols_r = np.asarray(sections_table.tol_r, dtype=FloatType._dtype)[undo_order]
        tols_x = np.asarray(sections_table.tol_x, dtype=FloatType._dtype)[undo_order]
        tols_y = np.asarray(sections_table.tol_y, dtype=FloatType._dtype)[undo_order]

        if axs is None:
            from matplotlib import pyplot as plt
            fig, axs = plt.subplots(2, 1, sharex=True)
        else:
            fig = axs[0].figure

        ax_x, ax_y = axs

        ax_x.fill_between(plot_s_positions, min_envel_x, max_envel_x, color='royalblue', alpha=0.3)
        ax_x.plot(plot_s_positions, min_aper_x, color='k')
        ax_x.plot(plot_s_positions, max_aper_x, color='k')
        ax_x.plot(plot_s_positions, min_aper_x + tols_x + tols_r, linestyle='--', color='k')
        ax_x.plot(plot_s_positions, max_aper_x - tols_x - tols_r, linestyle='--', color='k')
        ax_x.set_ylabel(r'x [m]')

        ax_y.fill_between(plot_s_positions, min_envel_y, max_envel_y, color='indianred', alpha=0.3)
        ax_y.plot(plot_s_positions, min_aper_y, color='k')
        ax_y.plot(plot_s_positions, max_aper_y, color='k')
        ax_y.plot(plot_s_positions, min_aper_y + tols_y + tols_r, linestyle='--', color='k')
        ax_y.plot(plot_s_positions, max_aper_y - tols_y - tols_r, linestyle='--', color='k')
        ax_y.set_ylabel(r'y [m]')
        ax_y.set_xlabel('s [m]')

        return fig, axs

    def plot_at_element(self, name, resolution=0.1, sigmas=None, method=None, middle='beam', ax=None):
        """Display a transverse plot of the beam at an element ``name``.

        Parameters
        ----------
        name : str
            Name of the element at which to plot.
        resolution : float
            The desired resolution, in metres along s, of the plot.
        sigmas : Optional[float]
            The number of sigmas to plot. If None, compute n1 using ``method``.
        method : str
            If ``sigmas`` is None, plot the maximum sigma for element, calculated using ``method``.
        middle : str
            Whether the plot should be centred around the ``aperture`` middle, or ``beam`` reference.
        ax : matplotlib.axes.Axes
            Axes object to plot on, if not given, spawn a new one.
        """
        from matplotlib import pyplot as plt
        ax = ax or plt.gca()

        if sigmas is None:
            n1_tab, _ = self.get_aperture_sigmas_at_element(name, method=method, resolution=resolution)
            sigmas = min(n1_tab.n1)

        s_positions = self._get_cuts_at_element(name, resolution)
        beam_tols, _ = self.get_envelope_at_s(s_positions=s_positions, sigmas=sigmas, include_aper_tols=True)
        beam_no_tols, _ = self.get_envelope_at_s(s_positions=s_positions, sigmas=sigmas, include_aper_tols=False)
        profiles = self.cross_sections_at_s(s_positions=s_positions)

        polygons = profiles.cross_section

        if middle == 'aperture':
            middle = (np.min(polygons, axis=1) + np.max(polygons, axis=1)) / 2
        elif middle == 'beam':
            middle = np.zeros(shape=(len(s_positions), 2))
        else:
            raise ValueError("Middle must be either 'aperture' or 'beam'")

        seen = False
        for pt, mid in zip(polygons, middle):
            label = 'aperture' if not seen else ''
            ax.plot(pt[:, 0] - mid[0], pt[:, 1] - mid[1], c='gray', linestyle='--', label=label)
            seen = True

        seen = False
        for pt, mid in zip(beam_tols, middle):
            label = 'envelope (with tolerances)' if not seen else ''
            ax.plot(pt[:, 0] - mid[0], pt[:, 1] - mid[1], c='royalblue', linestyle='-', label=label)
            seen = True

        seen = False
        for pt, mid in zip(beam_no_tols, middle):
            label = 'envelope (no tolerances)' if not seen else ''
            ax.plot(pt[:, 0] - mid[0], pt[:, 1] - mid[1], c='skyblue', linestyle=':', label=label)
            seen = True

        ax.set_aspect('equal')
        ax.set_title(fr"Envelope at {name}, s $\in$ [{s_positions[0]:.2f}, {s_positions[-1]:.2f}], $n$ = {sigmas:.3f}")
        ax.legend()
        return ax

    def plot_n1_at_element(self, name, resolution=0.1, method='rays', middle='beam', ax=None, **kwargs):
        """Display a transverse plot of the beam at n1 at element ``name``.

        Parameters
        ----------
        name : str
            Name of the element at which to plot.
        resolution : float
            The desired resolution, in metres along s, of the plot.
        method : str
            The method to use to calculate ``n1`` and the envelope.
        middle : str
            Whether the plot should be centred around the ``aperture`` middle, or ``beam`` reference.
        ax : matplotlib.axes.Axes
            Axes object to plot on, if not given, spawn a new one.
        **kwargs
            More arguments to pass to matplotlib.
        """
        from matplotlib import pyplot as plt
        ax = ax or plt.gca()

        n1_table, _ = self.get_aperture_sigmas_at_element(
            element_name=name,
            resolution=resolution,
            method=method,
            envelopes_num_points=128,
            output_max_envelopes=True,
            output_cross_sections=True,
        )

        n1 = np.min(n1_table.n1)
        polygons = n1_table.cross_section
        beam = n1_table.envelope

        if middle == 'aperture':
            middle = (np.min(polygons, axis=1) + np.max(polygons, axis=1)) / 2
        elif middle == 'beam':
            middle = np.zeros(shape=(len(n1_table), 2))
        else:
            raise ValueError("Middle must be either 'aperture' or 'beam'")

        for pt, mid in zip(polygons, middle):
            ax.plot(pt[:, 0] - mid[0], pt[:, 1] - mid[1], c='gray', linestyle='--')

        seen = False
        colour = {'rays': 'r', 'bisection': 'b', 'exact': 'g'}[method]
        for pt, mid in zip(beam, middle):
            label = f'envelope ({method}, min($n_1$) = {n1:.3f})' if not seen else ''
            ax.plot(pt[:, 0] - mid[0], pt[:, 1] - mid[1], c=colour, label=label, **kwargs)
            seen = True

        ax.set_aspect('equal')
        ax.set_title(fr"Max envelopes at {name}, s $\in$ [{n1_table.s[0]:.2f}, {n1_table.s[-1]:.2f}], min($n_1$) = {n1:.3f}")
        ax.legend()
        return ax

    def plot_floor_projection(
        self,
        ax=None,
        len_points=4,
        colour: Literal['profile', 'pipe'] = 'pipe',
        legend=True,
    ):
        from matplotlib import pyplot as plt
        ax = ax or plt.gca()

        for pipe_position in self.pipe_positions:
            pipe = pipe_position.pipe
            sv_ref_transform = survey_relative_transform(self.survey, 0, pipe_position.survey_reference_name)
            transform = sv_ref_transform @ pipe_position.transformation
            pipe.plot_projection(
                ax=ax,
                plane='zx',
                transform=transform,
                len_points=len_points,
                colour=colour,
                legend=False
            )

        if legend:
            _deduplicate_legend(ax)
            ax.legend()

        return ax

    def _get_cuts_at_element(self, element_name: str, resolution: Optional[float]) -> List[float]:
        """Get list of s positions so that the element ``element_name`` is cut with a ``resolution``."""
        element = self.line[element_name]
        s_start = self.line._get_s_position(element_name)
        element_length = getattr(element, 'length', 0)
        s_end = s_start + element_length

        if resolution is not None:
            num_cuts = int(element_length / resolution)
            s_positions = np.linspace(s_start, s_end, num_cuts)
        else:
            s_positions = [s_start, s_end]

        return s_positions

    def _build_aperture_bounds(self, check_validity=True):
        # Pre-allocate the cross-sections with the correct sizes
        num_points = self.num_profile_points
        num_cross_sections = sum(len(self._model.pipe_for_position(pipe_pos).positions) for pipe_pos in self._model.pipe_positions)
        self._aperture_bounds = ApertureBounds(
            count=num_cross_sections,
            pipe_position_indices=num_cross_sections,
            profile_position_indices=num_cross_sections,
            s_positions=num_cross_sections,
            s_start=num_cross_sections,
            s_end=num_cross_sections,
            _context=self.context,
        )

        # Pre-allocate the profile polygons (generate once, so that we only need to compute transformations on them)
        num_profile_polys = len(self._model.profiles)
        self._profile_polygons = ProfilePolygons(
            count=num_profile_polys,
            len_points=num_points,
            points=(num_profile_polys, num_points),
            _context=self.context,
        )

        cross_section_idx_iter = iter(progress(range(num_cross_sections), desc='Building cross-sections', total=num_cross_sections))

        for pipe_pos_idx, pipe_pos in enumerate(cast(Iterable[PipePosition], self._model.pipe_positions)):
            pipe = self._model.pipe_for_position(pipe_pos)
            for profile_pos_idx, profile_pos in enumerate(cast(Iterable[ProfilePosition], pipe.positions)):
                idx = next(cross_section_idx_iter)
                self._aperture_bounds.pipe_position_indices[idx] = pipe_pos_idx
                self._aperture_bounds.profile_position_indices[idx] = profile_pos_idx

        self._model.build_profile_polygons(
            profile_polygons=self._profile_polygons,
            aperture_bounds=self._aperture_bounds,
            survey=self._survey_data,
        )

        if check_validity:
            self._check_aperture_bounds_validity()

    def _check_model_validity(self):
        for ii, pipe_pos in enumerate(self._model.pipe_positions):
            survey_ref_name = pipe_pos.survey_reference_name
            survey_ref_idx = pipe_pos.survey_index
            pipe_position_name = self._model.pipe_position_name_for_position_index(ii)

            try:
                survey_at_idx = self.survey.name[survey_ref_idx]
            except IndexError:
                survey_at_idx = None

            if survey_at_idx != survey_ref_name:
                raise ValueError(
                    f'Aperture model corrupted for pipe position {pipe_position_name}: the associated survey reference '
                    f'name `{survey_ref_name}` and index {survey_ref_idx} do not match. The element of the survey at '
                    f'the index is {survey_at_idx}.'
                )

    def _check_aperture_bounds_validity(self, s_tol = 1e-6):
        # Check validity
        last_right = -np.inf

        if self._aperture_bounds.count < 1:
            raise ValueError('No aperture bounds computed. Is the model empty?')

        for idx in range(self._aperture_bounds.count):
            left = self._aperture_bounds.s_start[idx]
            centre = self._aperture_bounds.s_positions[idx]
            right = self._aperture_bounds.s_end[idx]

            pipe_pos_idx = self._aperture_bounds.pipe_position_indices[idx]
            profile_pos_idx = self._aperture_bounds.profile_position_indices[idx]
            pipe_position_name = self._model.pipe_position_name_for_position_index(pipe_pos_idx)
            pipe_name, profile_name = self._model.pipe_profile_names_for_indices(pipe_pos_idx, profile_pos_idx)

            if not (centre - left > -s_tol and right - centre > -s_tol):
                raise ValueError(
                    f'Aperture model corrupted for {pipe_position_name} (pipe {pipe_name}) and profile {profile_name} '
                    f'at index {profile_pos_idx}: the computed s location {centre} is not inside the computed bounds '
                    f'[{left}, {right}]'
                )

            if last_right > left:
                raise ValueError(
                    f'Aperture model corrupted for pipe {pipe_name} and profile {profile_name}): the '
                    f'aperture bounds [{left}, {right}] overlap the preceding profile whose s_end = {last_right}'
                )

    def get_bounds_table(self):
        """Get a table representation of the aperture bounds: per installed profile span information."""
        ap_bounds = self._aperture_bounds
        table_size = ap_bounds.count

        pipe_position_indices = ap_bounds.pipe_position_indices.to_nparray().astype(np.uint32, copy=False)
        profile_position_indices = ap_bounds.profile_position_indices.to_nparray().astype(np.uint32, copy=False)

        pipe_position_names_all = np.array(self._model.pipe_position_names, dtype=object)
        pipe_names_all = np.array(self._model.pipe_names, dtype=object)
        profile_names_all = np.array(self._model.profile_names, dtype=object)

        num_pipe_positions = len(self._model.pipe_positions)
        pipe_indices_for_position = np.empty(num_pipe_positions, dtype=np.int32)
        for i in range(num_pipe_positions):
            pipe_indices_for_position[i] = self._model.pipe_positions[i].pipe_index

        pipe_indices = pipe_indices_for_position[pipe_position_indices]
        profile_indices = np.empty(table_size, dtype=np.int32)
        for pipe_index in np.unique(pipe_indices):
            in_pipe = pipe_indices == pipe_index
            pipe = self._model.pipes[pipe_index]
            profile_indices_for_pipe = np.fromiter(
                (profile_position.profile_index for profile_position in pipe.positions),
                dtype=np.int32,
                count=len(pipe.positions),
            )
            profile_indices[in_pipe] = profile_indices_for_pipe[profile_position_indices[in_pipe]]

        shapes_all = np.empty(len(self._model.profiles), dtype=object)
        shape_params_all = np.empty(len(self._model.profiles), dtype=object)
        for i, profile in enumerate(self._model.profiles):
            shape = profile.shape
            shapes_all[i] = type(shape).__name__
            shape_params_all[i] = shape._to_dict()

        table = Table(
            data={
                'name': pipe_position_names_all[pipe_position_indices],
                'pipe_name': pipe_names_all[pipe_indices],
                'profile_name': profile_names_all[profile_indices],
                's': ap_bounds.s_positions.to_nparray(),
                's_start': ap_bounds.s_start.to_nparray(),
                's_end': ap_bounds.s_end.to_nparray(),
                'shape': shapes_all[profile_indices],
                'shape_param': shape_params_all[profile_indices],
            },
            index='name',
        )
        return table

    def get_pipe_table(self):
        """Get a table representation of the pipe bounds: per installed pipe span information."""
        table_size = len(self._model.pipe_positions)

        pipe_position_names = np.array(self._model.pipe_position_names, dtype=object)
        pipe_indices = np.empty(table_size, dtype=np.int32)
        survey_references = np.empty(table_size, dtype=object)
        for i in range(table_size):
            pipe_position = self._model.pipe_positions[i]
            pipe_indices[i] = pipe_position.pipe_index
            survey_references[i] = pipe_position.survey_reference_name
        pipe_names = np.array(self._model.pipe_names, dtype=object)[pipe_indices]

        ap_bounds = self._aperture_bounds
        pipe_position_indices = ap_bounds.pipe_position_indices.to_nparray().astype(np.uint32, copy=False)
        s_start = np.full(table_size, np.inf, dtype=FloatType._dtype)
        s_end = np.full(table_size, -np.inf, dtype=FloatType._dtype)
        np.minimum.at(s_start, pipe_position_indices, ap_bounds.s_start.to_nparray())
        np.maximum.at(s_end, pipe_position_indices, ap_bounds.s_end.to_nparray())

        table = Table(
            data={
                'name': pipe_position_names,
                'pipe_name': pipe_names,
                'survey_reference': survey_references,
                's_start': s_start,
                's_end': s_end,
                'span': s_end - s_start,
            },
            index='name',
        )
        return table

    def _sliced_twiss_at_s(
            self,
            s_positions: Iterable[float],
            twiss_init: Optional[TwissInit] = None,
    ) -> TwissTable:
        """Get a twiss table for the line with entries at each requested `s`.

        Parameters
        ----------
        s_positions : Iterable[float]
            s-positions for the sliced twiss.
        twiss_init : TwissInit, optional
            Initial conditions for the twiss.
        """
        s_positions = np.array(s_positions, dtype=FloatType._dtype)
        line_sliced = self.line.copy()
        line_sliced.cut_at_s(s_positions)

        full_twiss = line_sliced.twiss(init=twiss_init, reverse=False)

        # "Authoritative" s-positions after slicing (up to cutting tolerances)
        tw_s = np.array(full_twiss.s, dtype=FloatType._dtype)

        # Bracket each requested s with the nearest row on the left and on the right.
        idx_left = np.searchsorted(tw_s, s_positions, side='right') - 1
        idx_right = np.searchsorted(tw_s, s_positions, side='left')

        idx_left = np.clip(idx_left, 0, len(tw_s) - 1)
        idx_right = np.clip(idx_right, 0, len(tw_s) - 1)

        # Compare the left/right candidates by distance in s.
        dist_left = np.abs(tw_s[idx_left] - s_positions)
        dist_right = np.abs(tw_s[idx_right] - s_positions)

        # Keep the nearest row; on ties prefer the rightmost candidate.
        tw_indices = np.where(dist_right <= dist_left, idx_right, idx_left)

        sliced_twiss = full_twiss.rows[tw_indices]
        return sliced_twiss

    @staticmethod
    def _axis_extents_for_cross_sections(cross_sections: np.ndarray) -> dict[str, np.ndarray]:
        x0 = cross_sections[:, :-1, 0]
        y0 = cross_sections[:, :-1, 1]
        x1 = cross_sections[:, 1:, 0]
        y1 = cross_sections[:, 1:, 1]

        with np.errstate(divide='ignore', invalid='ignore'):
            dy = y1 - y0
            t_x_axis = -y0 / dy
            crosses_x_axis = ((y0 < 0) & (y1 > 0)) | ((y0 > 0) & (y1 < 0))
            x_axis_interp = np.where(crosses_x_axis, x0 + t_x_axis * (x1 - x0), np.nan)
            x_axis_vertices = np.where(y0 == 0, x0, np.nan)
            x_axis_vertices_next = np.where(y1 == 0, x1, np.nan)
            on_x_axis = (y0 == 0) & (y1 == 0)
            x_axis_seg0 = np.where(on_x_axis, x0, np.nan)
            x_axis_seg1 = np.where(on_x_axis, x1, np.nan)
            x_candidates = np.concatenate(
                [x_axis_interp, x_axis_vertices, x_axis_vertices_next, x_axis_seg0, x_axis_seg1],
                axis=1,
            )

            dx = x1 - x0
            t_y_axis = -x0 / dx
            crosses_y_axis = ((x0 < 0) & (x1 > 0)) | ((x0 > 0) & (x1 < 0))
            y_axis_interp = np.where(crosses_y_axis, y0 + t_y_axis * (y1 - y0), np.nan)
            y_axis_vertices = np.where(x0 == 0, y0, np.nan)
            y_axis_vertices_next = np.where(x1 == 0, y1, np.nan)
            on_y_axis = (x0 == 0) & (x1 == 0)
            y_axis_seg0 = np.where(on_y_axis, y0, np.nan)
            y_axis_seg1 = np.where(on_y_axis, y1, np.nan)
            y_candidates = np.concatenate(
                [y_axis_interp, y_axis_vertices, y_axis_vertices_next, y_axis_seg0, y_axis_seg1],
                axis=1,
            )

        has_x = np.any(np.isfinite(x_candidates), axis=1)
        has_y = np.any(np.isfinite(y_candidates), axis=1)

        return {
            'min_x': np.where(has_x, np.nanmin(x_candidates, axis=1), np.nan),
            'max_x': np.where(has_x, np.nanmax(x_candidates, axis=1), np.nan),
            'min_y': np.where(has_y, np.nanmin(y_candidates, axis=1), np.nan),
            'max_y': np.where(has_y, np.nanmax(y_candidates, axis=1), np.nan),
        }

    @classmethod
    def _guess_original_mad_name(cls, element_name) -> Any:
        """Given a name of an element in a line, de-mangle the original MAD-X name.

        When importing a line from MAD-X, names can be mangled in two ways:
        1. ``:N`` may be added for the (N+1)-th repetition of the same element.
        2. ``/line_name`` may be appended if the same element appears in multiple sequences.
        This function only works if the elements were not named according to these patterns by the user,
        and as such is a bit of a hack. In corner cases it is best not to go through cpymad,
        but instead use the native loader with ``Aperture.from_line_with_associated_apertures``.

        Parameters
        ----------
        element_name : str
            Name of a beam element.
        """
        pattern = r"(?P<prefix>.*?)(?:[:]\d+)?(?:/[^/]+)?"
        match = re.fullmatch(pattern, element_name)
        return match.group('prefix')

    @classmethod
    def _get_per_pipe_madx_offsets(cls, madx_offsets):
        """Parse MAD-X imported aperture offsets metadata to obtain per-element (type) transformations."""
        offsets = {}
        for section in madx_offsets.values():
            reference_name = section['reference']
            for idx, name in enumerate(section['name']):
                dx = section['dx_off'][idx]
                dy = section['dy_off'][idx]

                rot_y_rad = np.arctan2(dx, 1)
                rot_x_rad = np.arctan2(dy, np.sqrt(1 + dx ** 2))

                offsets[name] = {
                    'survey_ref': reference_name,
                    's': section['s_ip'][idx],
                    'x': section['x_off'][idx],
                    'y': section['y_off'][idx],
                    'rot_y_rad': rot_y_rad,
                    'rot_x_rad': rot_x_rad,
                    'dx': dx,
                    'dy': dy,
                    'ddx': section['ddx_off'][idx],
                    'ddy': section['ddy_off'][idx],
                    'reversed': section['reversed'],
                }
        return offsets

    @classmethod
    def _is_broken_madx_aperture(cls, shape):
        if isinstance(shape, Circle):
            return shape.radius < 1e-6 or shape.radius > 9.98

        if isinstance(shape, Rectangle):
            return (shape.half_width < 1e-6 or shape.half_width > 9.98) and (shape.half_height < 1e-6 or shape.half_height > 9.98)

        if isinstance(shape, RectEllipse):
            return (
                (shape.half_width < 1e-6 or shape.half_width > 9.98) and
                (shape.half_height < 1e-6 or shape.half_height > 9.98) and
                (shape.half_major < 1e-6 or shape.half_major > 9.98) and
                (shape.half_minor < 1e-6 or shape.half_minor > 9.98)
            )

        return False
