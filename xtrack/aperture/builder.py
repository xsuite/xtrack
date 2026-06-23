from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, get_args

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
)
from xtrack.aperture.transform import transform_matrix


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
    profile_name: str
    shift_s: float = 0.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    rot_y_rad: float = 0.0
    rot_x_rad: float = 0.0
    rot_s_rad: float = 0.0


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


@dataclass
class PipePositionBlueprint:
    name: str
    pipe_name: str
    survey_reference: str
    transformation: np.ndarray


TypeBlueprint = PipeBlueprint
TypePositionBlueprint = PipePositionBlueprint


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

    def new_type(self, *args, **kwargs):
        return self.new_pipe(*args, **kwargs)

    def place_pipe(
        self,
        name: str,
        pipe_name: str,
        survey_reference: str,
        transformation: Optional[np.ndarray] = None,
        shift_x: Optional[float] = None,
        shift_y: Optional[float] = None,
        shift_z: Optional[float] = None,
        rot_y_rad: Optional[float] = None,
        rot_x_rad: Optional[float] = None,
        rot_z_rad: Optional[float] = None,
    ) -> PipePositionBlueprint:
        """Create and register a pipe-position blueprint.

        Parameters
        ----------
        name : str
            Name of the installed pipe position.
        pipe_name : str
            Name of the pipe to install.
        survey_reference : str
            Name of the survey entry used as the installation reference.
        transformation : np.ndarray, optional
            Full 4x4 homogeneous transform from the survey reference to the
            pipe frame.
        shift_x, shift_y, shift_z : float, optional
            Translation components used when ``transformation`` is not given.
        rot_y_rad, rot_x_rad, rot_z_rad : float, optional
            Rotation components used when ``transformation`` is not given.

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
        if any(pipe_position.name == name for pipe_position in self._pipe_positions):
            raise ValueError(f"Pipe position `{name}` already exists.")

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

        pipe_position = PipePositionBlueprint(
            name=name,
            pipe_name=pipe_name,
            survey_reference=survey_reference,
            transformation=transformation,
        )
        self._pipe_positions.append(pipe_position)
        return pipe_position

    def place_type(self, *args, **kwargs):
        return self.place_pipe(*args, **kwargs)

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
