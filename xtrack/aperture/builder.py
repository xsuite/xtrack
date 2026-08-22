from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, get_args

import numpy as np

import xobjects as xo
from xobjects.context import XContext

from xtrack.aperture.structures import (
    ApertureModel,
    Pipe,
    Profile,
    ProfilePosition,
    ShapeTypes,
    PipePosition,
    SurveyData,
)
from xtrack.aperture.transform import transform_matrix
from xtrack.general import parse_anchor_spec


SHAPE_CLASSES = get_args(ShapeTypes)
SHAPE_CLASSES_BY_NAME = {shape_cls.__name__: shape_cls for shape_cls in SHAPE_CLASSES}


def _survey_is_closed(line, tol: float = 1e-6) -> bool:
    survey = line.survey()
    start = np.array([survey.X[0], survey.Y[0], survey.Z[0]], dtype=float)
    end = np.array([survey.X[-1], survey.Y[-1], survey.Z[-1]], dtype=float)
    return np.linalg.norm(end - start) < tol


def _shape_from_input(shape: str | type, **shape_params):
    """Build a profile shape object from a string name or shape class.

    Parameters
    ----------
    shape : str or type
        Name of the shape to build, or the corresponding shape class.
    **shape_params
        Parameters forwarded to the corresponding shape constructor.

    Returns
    -------
    xo.Struct
        Instantiated shape object.

    Raises
    ------
    ValueError
        If the shape input is not supported.
    """
    if isinstance(shape, str):
        shape_cls = SHAPE_CLASSES_BY_NAME.get(shape)
    elif shape in SHAPE_CLASSES:
        shape_cls = shape
    else:
        shape_cls = None

    if shape_cls is None:
        raise ValueError(f"Unsupported aperture profile shape `{shape}`.")

    return shape_cls(**shape_params)


@dataclass
class ProfilePositionBlueprint:
    builder: ApertureBuilder
    profile_name: str
    shift_s: float = 0.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    rot_y_rad: float = 0.0
    rot_x_rad: float = 0.0
    rot_s_rad: float = 0.0

    @property
    def profile(self):
        return self.builder._profiles[self.profile_name]


@dataclass
class PipeBlueprint:
    builder: ApertureBuilder
    name: str
    curvature: float = 0.0
    positions: list[ProfilePositionBlueprint] = field(default_factory=list)

    def place_profile(
        self,
        name: str,
        shift_s: float = 0.0,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        rot_y_rad: float = 0.0,
        rot_x_rad: float = 0.0,
        rot_s_rad: float = 0.0,
    ) -> ProfilePositionBlueprint:
        """Create and append a profile position blueprint to this pipe.

        Parameters
        ----------
        name : str
            Name of the profile to place.
        shift_s : float, optional
            Longitudinal position of the profile in the pipe frame.
        shift_x, shift_y : float, optional
            Transverse offsets of the profile in the pipe frame.
        rot_y_rad, rot_x_rad, rot_s_rad : float, optional
            Rotations of the profile in the pipe frame.

        Returns
        -------
        ProfilePositionBlueprint
            The created profile position blueprint.
        """
        profile_position = self.builder.place_profile(
            name=name,
            shift_s=shift_s,
            shift_x=shift_x,
            shift_y=shift_y,
            rot_y_rad=rot_y_rad,
            rot_x_rad=rot_x_rad,
            rot_s_rad=rot_s_rad,
        )
        self.positions.append(profile_position)
        return profile_position

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, item: int) -> ProfilePositionBlueprint:
        return sorted(self.positions, key=lambda position: position.shift_s)[item]

    def __iter__(self):
        return iter(sorted(self.positions, key=lambda position: position.shift_s))

    @property
    def length(self):
        if not self.positions:
            return 0.0
        positions = sorted(self.positions, key=lambda position: position.shift_s)
        return positions[-1].shift_s - positions[0].shift_s

    @property
    def angle(self):
        return self.length * self.curvature

    def plot(self, *args, **kwargs):
        """Plot pipe projection using the same box-style view as :class:`PipeView`."""
        from xtrack.aperture.plot import plot_pipe_projection

        return plot_pipe_projection(self, *args, **kwargs)

    def plot_3d(self, *args, **kwargs):
        """Plot pipe as a 3D solid using the same view as :class:`PipeView`."""
        from xtrack.aperture.plot import plot_pipe_3d

        return plot_pipe_3d(self, *args, **kwargs)

    def place(self, name: str | Sequence[str], at: str | Sequence[str], **kwargs):
        """Install this pipe in the builder."""
        return self.builder.place_pipe(name=name, pipe_name=self.name, at=at, **kwargs)


@dataclass
class PipePositionBlueprint:
    name: str
    pipe_name: str
    survey_reference: str
    transformation: np.ndarray


class ApertureBuilder:
    def __init__(self, line):
        """Create an aperture model builder for a given line.

        Parameters
        ----------
        line : xtrack.Line
            Line whose survey is used when resolving installed pipe positions.
        """
        self.line = line
        self._profiles: dict[str, Profile] = {}
        self._pipes: dict[str, PipeBlueprint] = {}
        self._pipe_positions: list[PipePositionBlueprint] = []
        self._survey_data: SurveyData | None = None

    @property
    def profiles(self) -> dict[str, Profile]:
        """Registered profile blueprints keyed by name."""
        return self._profiles

    @property
    def pipes(self) -> dict[str, PipeBlueprint]:
        """Registered pipe blueprints keyed by name."""
        return self._pipes

    @property
    def pipe_positions(self) -> list[PipePositionBlueprint]:
        """Registered installed pipe positions."""
        return self._pipe_positions

    def new_profile(
        self,
        name: str,
        shape: str | type,
        tol_r: float = 0.0,
        tol_x: float = 0.0,
        tol_y: float = 0.0,
        **shape_params,
    ) -> str:
        """Create and register a new profile blueprint.

        Parameters
        ----------
        name : str
            Name of the new profile.
        shape : str or type
            Shape name or shape class used to construct the profile geometry.
        tol_r, tol_x, tol_y : float, optional
            Profile tolerances.
        **shape_params
            Parameters forwarded to the shape constructor.

        Returns
        -------
        str
            The profile name.

        Raises
        ------
        ValueError
            If a profile with the same name already exists.
        """
        if name in self._profiles:
            raise ValueError(f"Profile `{name}` already exists.")

        self._profiles[name] = Profile(
            shape=_shape_from_input(shape, **shape_params),
            tol_r=tol_r,
            tol_x=tol_x,
            tol_y=tol_y,
        )
        return name

    def place_profile(
        self,
        name: str,
        shift_s: float = 0.0,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        rot_y_rad: float = 0.0,
        rot_x_rad: float = 0.0,
        rot_s_rad: float = 0.0,
    ) -> ProfilePositionBlueprint:
        """Create a profile position blueprint.

        Parameters
        ----------
        name : str
            Name of the profile to place.
        shift_s : float, optional
            Longitudinal position of the profile in the pipe frame.
        shift_x, shift_y : float, optional
            Transverse offsets of the profile in the pipe frame.
        rot_y_rad, rot_x_rad, rot_s_rad : float, optional
            Rotations of the profile in the pipe frame.

        Returns
        -------
        ProfilePositionBlueprint
            The created profile position blueprint.
        """
        return ProfilePositionBlueprint(
            builder=self,
            profile_name=name,
            shift_s=shift_s,
            shift_x=shift_x,
            shift_y=shift_y,
            rot_y_rad=rot_y_rad,
            rot_x_rad=rot_x_rad,
            rot_s_rad=rot_s_rad,
        )

    def new_pipe(
        self,
        name: str,
        curvature: float = 0.0,
        positions: Optional[list[ProfilePositionBlueprint]] = None,
    ) -> PipeBlueprint:
        """Create and register a new aperture pipe blueprint.

        Parameters
        ----------
        name : str
            Name of the new pipe.
        curvature : float, optional
            Curvature assigned to the pipe.
        positions : list of ProfilePositionBlueprint, optional
            Initial profile positions to install in the pipe.

        Returns
        -------
        PipeBlueprint
            The created pipe blueprint.

        Raises
        ------
        ValueError
            If a pipe with the same name already exists.
        """
        if name in self._pipes:
            raise ValueError(f"Pipe `{name}` already exists.")

        pipe = PipeBlueprint(
            builder=self,
            name=name,
            curvature=curvature,
            positions=list(positions) if positions is not None else [],
        )
        self._pipes[name] = pipe
        return pipe

    def _get_survey_data(self) -> SurveyData:
        if self._survey_data is None:
            self._survey_data = SurveyData.from_survey_table(self.line.survey(), self.line)
        return self._survey_data

    @staticmethod
    def _row_value(row, field: str) -> float:
        return float(np.asarray(getattr(row, field)).item())

    @staticmethod
    def _pose_to_matrix(pose) -> np.ndarray:
        return np.array(pose, dtype=float, copy=True)

    @staticmethod
    def _row_by_exact_name(table, name: str):
        for row in table.rows:
            if row.name == name:
                return row
        raise KeyError(name)

    def _anchor_transform(self, survey_reference: str, anchor: str | None) -> np.ndarray:
        if anchor in (None, "start"):
            return np.identity(4)

        table = self.line.get_table()
        try:
            element_row = self._row_by_exact_name(table, survey_reference)
        except KeyError as error:
            raise ValueError(f"Unknown element `{survey_reference}` in aperture pipe placement.") from error

        if anchor in ("center", "centre"):
            anchor_s = self._row_value(element_row, "s_center")
        elif anchor == "end":
            anchor_s = self._row_value(element_row, "s_end")
        else:
            raise ValueError(f"Invalid anchor `{anchor}`.")

        survey = self.line.survey()
        try:
            survey_row = self._row_by_exact_name(survey, survey_reference)
        except KeyError as error:
            raise ValueError(f"Unknown survey reference `{survey_reference}` in aperture pipe placement.") from error

        reference_pose = np.identity(4)
        reference_pose[:3, :3] = survey_row.E_matrix
        reference_pose[:3, 3] = survey_row.XYZ

        anchor_survey = self._get_survey_data().resample([anchor_s])
        anchor_pose = self._pose_to_matrix(anchor_survey.pose.to_nplike()[0])
        return np.linalg.inv(reference_pose) @ anchor_pose

    def place_pipe(
        self,
        name: str | Sequence[str],
        pipe_name: str,
        at: str | Sequence[str],
        transformation: Optional[np.ndarray] = None,
        shift_x: Optional[float] = None,
        shift_y: Optional[float] = None,
        shift_z: Optional[float] = None,
        rot_y_rad: Optional[float] = None,
        rot_x_rad: Optional[float] = None,
        rot_z_rad: Optional[float] = None,
    ) -> PipePositionBlueprint | list[PipePositionBlueprint]:
        """Create and register a pipe-position blueprint.

        Parameters
        ----------
        name : str
            Name of the installed pipe position.
        pipe_name : str
            Name of the pipe to install.
        at : str
            Survey entry used as the installation reference. The syntax
            ``element@anchor`` can be used with anchors ``start``, ``center``,
            ``centre``, and ``end``. The stored survey reference remains the
            element name; the requested anchor offset is encoded in the stored
            transformation.
        transformation : np.ndarray, optional
            Full 4x4 homogeneous transform from the survey reference to the
            pipe frame.
        shift_x, shift_y, shift_z : float, optional
            Translation component used when ``transformation`` is not given.
        rot_y_rad, rot_x_rad, rot_z_rad : float, optional
            Rotation component used when ``transformation`` is not given.

        Returns
        -------
        PipePositionBlueprint
            The created pipe-position blueprint.

        Raises
        ------
        ValueError
            If the pipe-position name already exists, or if both a full matrix
            and transform components are supplied.
        """
        if not isinstance(at, str):
            at_values = list(at)
            if isinstance(name, str):
                names = [f"{name}.{ii}" for ii in range(len(at_values))]
            else:
                names = list(name)
                if len(names) != len(at_values):
                    raise ValueError("Expected `name` and `at` to have the same length.")
            return [
                self.place_pipe(
                    name=position_name,
                    pipe_name=pipe_name,
                    at=at_value,
                    transformation=transformation,
                    shift_x=shift_x,
                    shift_y=shift_y,
                    shift_z=shift_z,
                    rot_y_rad=rot_y_rad,
                    rot_x_rad=rot_x_rad,
                    rot_z_rad=rot_z_rad,
                )
                for position_name, at_value in zip(names, at_values)
            ]

        if any(pipe_position.name == name for pipe_position in self._pipe_positions):
            raise ValueError(f"Pipe position `{name}` already exists.")

        survey_reference, anchor = parse_anchor_spec(at, default_anchor="start")

        transform_fields = {
            "shift_x": shift_x,
            "shift_y": shift_y,
            "shift_z": shift_z,
            "rot_y_rad": rot_y_rad,
            "rot_x_rad": rot_x_rad,
            "rot_z_rad": rot_z_rad,
        }
        if transformation is not None and any(value is not None for value in transform_fields.values()):
            raise ValueError("Provide either `transformation` or transform components, not both.")

        if transformation is None:
            transformation = transform_matrix(
                **{key: (0.0 if value is None else value) for key, value in transform_fields.items()}
            )
        else:
            transformation = np.array(transformation, dtype=float, copy=True)

        transformation = self._anchor_transform(survey_reference, anchor) @ transformation

        pipe_position = PipePositionBlueprint(
            name=name,
            pipe_name=pipe_name,
            survey_reference=survey_reference,
            transformation=transformation,
        )
        self._pipe_positions.append(pipe_position)
        return pipe_position

    def build(self, context: Optional[XContext] = None) -> ApertureModel:
        """Build an :class:`ApertureModel` in the requested context.

        Parameters
        ----------
        context : XContext, optional
            Context in which to allocate the generated xobjects. If omitted,
            ``xo.context_default`` is used.

        Returns
        -------
        ApertureModel
            Fully materialised aperture model with names resolved to indices.
        """
        context = context or xo.context_default

        survey = self.line.survey()
        survey_name_to_index = dict(zip(survey.name, range(len(survey.name))))

        profile_names = list(self._profiles.keys())
        profile_name_to_index = {name: ii for ii, name in enumerate(profile_names)}
        profiles = [Profile(**self._profiles[name]._to_dict(), _context=context) for name in profile_names]

        pipe_names = list(self._pipes.keys())
        pipe_name_to_index = {name: ii for ii, name in enumerate(pipe_names)}
        pipes = []
        for pipe_name in pipe_names:
            pipe_blueprint = self._pipes[pipe_name]
            sorted_positions = sorted(pipe_blueprint.positions, key=lambda position: position.shift_s)
            positions = [
                ProfilePosition(
                    profile_index=profile_name_to_index[position.profile_name],
                    shift_s=position.shift_s,
                    shift_x=position.shift_x,
                    shift_y=position.shift_y,
                    rot_y_rad=position.rot_y_rad,
                    rot_x_rad=position.rot_x_rad,
                    rot_s_rad=position.rot_s_rad,
                    _context=context,
                )
                for position in sorted_positions
            ]
            pipes.append(
                Pipe(
                    curvature=pipe_blueprint.curvature,
                    positions=positions,
                    _context=context,
                )
            )

        pipe_position_names = [pipe_position.name for pipe_position in self._pipe_positions]
        pipe_positions = [
            PipePosition(
                pipe_index=pipe_name_to_index[pipe_position.pipe_name],
                survey_reference_name=pipe_position.survey_reference,
                survey_index=survey_name_to_index[pipe_position.survey_reference],
                transformation=pipe_position.transformation,
                _context=context,
            )
            for pipe_position in self._pipe_positions
        ]

        return ApertureModel(
            pipe_positions=pipe_positions,
            pipes=pipes,
            profiles=profiles,
            pipe_names=pipe_names,
            pipe_position_names=pipe_position_names,
            profile_names=profile_names,
            is_ring=_survey_is_closed(self.line),
            survey_length=self.line.get_length(),
            _context=context,
        )
