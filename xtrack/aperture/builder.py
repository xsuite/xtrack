from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, get_args

import numpy as np

import xobjects as xo
from xobjects.context import XContext

from xtrack.aperture.structures import (
    ApertureModel,
    ApertureType,
    Profile,
    ProfilePosition,
    ShapeTypes,
    TypePosition,
)
from xtrack.aperture.transform import transform_matrix


SHAPE_CLASSES = get_args(ShapeTypes)
SHAPE_CLASSES_BY_NAME = {shape_cls.__name__: shape_cls for shape_cls in SHAPE_CLASSES}


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
    s_position: float = 0.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    rot_y: float = 0.0
    rot_x: float = 0.0
    rot_s: float = 0.0


@dataclass
class TypeBlueprint:
    builder: ApertureBuilder
    name: str
    curvature: float = 0.0
    positions: list[ProfilePositionBlueprint] = field(default_factory=list)

    def place_profile(
        self,
        name: str,
        s_position: float = 0.0,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        rot_y: float = 0.0,
        rot_x: float = 0.0,
        rot_s: float = 0.0,
    ) -> ProfilePositionBlueprint:
        """Create and append a profile position blueprint to this type.

        Parameters
        ----------
        name : str
            Name of the profile to place.
        s_position : float, optional
            Longitudinal position of the profile in the type frame.
        shift_x, shift_y : float, optional
            Transverse offsets of the profile in the type frame.
        rot_y, rot_x, rot_s : float, optional
            Rotations of the profile in the type frame.

        Returns
        -------
        ProfilePositionBlueprint
            The created profile position blueprint.
        """
        profile_position = self.builder.place_profile(
            name=name,
            s_position=s_position,
            shift_x=shift_x,
            shift_y=shift_y,
            rot_y=rot_y,
            rot_x=rot_x,
            rot_s=rot_s,
        )
        self.positions.append(profile_position)
        return profile_position


@dataclass
class TypePositionBlueprint:
    name: str
    type_name: str
    survey_reference: str
    transformation: np.ndarray


class ApertureBuilder:
    def __init__(self, line):
        """Create an aperture model builder for a given line.

        Parameters
        ----------
        line : xtrack.Line
            Line whose survey is used when resolving installed type positions.
        """
        self.line = line
        self._profiles: dict[str, Profile] = {}
        self._types: dict[str, TypeBlueprint] = {}
        self._type_positions: list[TypePositionBlueprint] = []

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
        s_position: float = 0.0,
        shift_x: float = 0.0,
        shift_y: float = 0.0,
        rot_y: float = 0.0,
        rot_x: float = 0.0,
        rot_s: float = 0.0,
    ) -> ProfilePositionBlueprint:
        """Create a profile position blueprint.

        Parameters
        ----------
        name : str
            Name of the profile to place.
        s_position : float, optional
            Longitudinal position of the profile in the type frame.
        shift_x, shift_y : float, optional
            Transverse offsets of the profile in the type frame.
        rot_y, rot_x, rot_s : float, optional
            Rotations of the profile in the type frame.

        Returns
        -------
        ProfilePositionBlueprint
            The created profile position blueprint.
        """
        return ProfilePositionBlueprint(
            profile_name=name,
            s_position=s_position,
            shift_x=shift_x,
            shift_y=shift_y,
            rot_y=rot_y,
            rot_x=rot_x,
            rot_s=rot_s,
        )

    def new_type(
        self,
        name: str,
        curvature: float = 0.0,
        positions: Optional[list[ProfilePositionBlueprint]] = None,
    ) -> TypeBlueprint:
        """Create and register a new aperture type blueprint.

        Parameters
        ----------
        name : str
            Name of the new type.
        curvature : float, optional
            Curvature assigned to the type.
        positions : list of ProfilePositionBlueprint, optional
            Initial profile positions to install in the type.

        Returns
        -------
        TypeBlueprint
            The created type blueprint.

        Raises
        ------
        ValueError
            If a type with the same name already exists.
        """
        if name in self._types:
            raise ValueError(f"Type `{name}` already exists.")

        aperture_type = TypeBlueprint(
            builder=self,
            name=name,
            curvature=curvature,
            positions=list(positions) if positions is not None else [],
        )
        self._types[name] = aperture_type
        return aperture_type

    def place_type(
        self,
        name: str,
        type_name: str,
        survey_reference: str,
        transformation: Optional[np.ndarray] = None,
        shift_x: Optional[float] = None,
        shift_y: Optional[float] = None,
        shift_z: Optional[float] = None,
        rot_y: Optional[float] = None,
        rot_x: Optional[float] = None,
        rot_z: Optional[float] = None,
    ) -> TypePositionBlueprint:
        """Create and register a type-position blueprint.

        Parameters
        ----------
        name : str
            Name of the installed type position.
        type_name : str
            Name of the type to install.
        survey_reference : str
            Name of the survey entry used as the installation reference.
        transformation : np.ndarray, optional
            Full 4x4 homogeneous transform from the survey reference to the
            type frame.
        shift_x, shift_y, shift_z : float, optional
            Translation components used when ``transformation`` is not given.
        rot_y, rot_x, rot_z : float, optional
            Rotation components used when ``transformation`` is not given.

        Returns
        -------
        TypePositionBlueprint
            The created type-position blueprint.

        Raises
        ------
        ValueError
            If the type-position name already exists, or if both a full matrix
            and transform components are supplied.
        """
        if any(type_position.name == name for type_position in self._type_positions):
            raise ValueError(f"Type position `{name}` already exists.")

        transform_fields = {
            "shift_x": shift_x,
            "shift_y": shift_y,
            "shift_z": shift_z,
            "rot_y": rot_y,
            "rot_x": rot_x,
            "rot_z": rot_z,
        }
        if transformation is not None and any(value is not None for value in transform_fields.values()):
            raise ValueError("Provide either `transformation` or transform components, not both.")

        if transformation is None:
            transformation = transform_matrix(
                **{key: (0.0 if value is None else value) for key, value in transform_fields.items()}
            )
        else:
            transformation = np.array(transformation, dtype=float, copy=True)

        type_position = TypePositionBlueprint(
            name=name,
            type_name=type_name,
            survey_reference=survey_reference,
            transformation=transformation,
        )
        self._type_positions.append(type_position)
        return type_position

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

        type_names = list(self._types.keys())
        type_name_to_index = {name: ii for ii, name in enumerate(type_names)}
        types = []
        for type_name in type_names:
            type_blueprint = self._types[type_name]
            sorted_positions = sorted(type_blueprint.positions, key=lambda position: position.s_position)
            positions = [
                ProfilePosition(
                    profile_index=profile_name_to_index[position.profile_name],
                    s_position=position.s_position,
                    shift_x=position.shift_x,
                    shift_y=position.shift_y,
                    rot_y=position.rot_y,
                    rot_x=position.rot_x,
                    rot_s=position.rot_s,
                    _context=context,
                )
                for position in sorted_positions
            ]
            types.append(
                ApertureType(
                    curvature=type_blueprint.curvature,
                    positions=positions,
                    _context=context,
                )
            )

        type_position_names = [type_position.name for type_position in self._type_positions]
        type_positions = [
            TypePosition(
                type_index=type_name_to_index[type_position.type_name],
                survey_reference_name=type_position.survey_reference,
                survey_index=survey_name_to_index[type_position.survey_reference],
                transformation=type_position.transformation,
                _context=context,
            )
            for type_position in self._type_positions
        ]

        return ApertureModel(
            type_positions=type_positions,
            types=types,
            profiles=profiles,
            type_names=type_names,
            type_position_names=type_position_names,
            profile_names=profile_names,
            _context=context,
        )
