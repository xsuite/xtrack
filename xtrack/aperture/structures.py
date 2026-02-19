from typing import List, Union, get_args

import numpy as np
import xobjects as xo
from xtrack.particles import Particles
from xtrack.survey import SurveyTable
from xtrack.twiss import TwissTable


class Circle(xo.Struct):
    radius = xo.Float32


class Rectangle(xo.Struct):
    half_width = xo.Float32
    half_height = xo.Float32


class Ellipse(xo.Struct):
    half_major = xo.Float32
    half_minor = xo.Float32


class RectEllipse(xo.Struct):
    half_width = xo.Float32
    half_height = xo.Float32
    half_major = xo.Float32
    half_minor = xo.Float32


class Racetrack(xo.Struct):
    half_width = xo.Float32
    half_height = xo.Float32
    half_major = xo.Float32
    half_minor = xo.Float32


class Octagon(xo.Struct):
    half_width = xo.Float32
    half_height = xo.Float32
    half_diagonal = xo.Float32


class Polygon(xo.Struct):
    vertices = xo.Float32[:, 2]


class SVGShape(xo.Struct):
    svg_data = xo.String


ShapeTypes = Union[Circle, Rectangle, Ellipse, RectEllipse, Racetrack, Octagon, Polygon, SVGShape]


class Shape(xo.UnionRef):
    _reftypes = get_args(ShapeTypes)


class Profile(xo.Struct):
    """Structure representing a profile with associated tolerances.

    Parameters
    ----------
    tol_r: float
        Radial tolerance for point-in-aperture check.
    tol_x: float
        Horizontal tolerance for point-in-aperture check.
    tol_y: float
        Vertical tolerance for point-in-aperture check.
    """
    shape = Shape
    tol_r = xo.Float32
    tol_x = xo.Float32
    tol_y = xo.Float32


class ProfilePosition(xo.Struct):
    """Description of the placement of a profile in type (lab) frame.

    Parameters
    ----------
    profile_index: int
        The index identifying the profile in the associated ``Profiles`` object.
    s_position: float
        The position along the type axis where this profile sits.
    shift_x: float
        The horizontal shift of the profile centre from the type axis.
    shift_y: float
        The vertical shift of the profile centre from the type axis
    rot_x: float
        The rotation of the profile around the type axis in radians.
    rot_y: float
        The rotation of the profile around the vertical axis in radians.
    rot_z: float
        The rotation of the profile around the horizontal axis in radians.
    """
    profile_index = xo.Int32
    s_position = xo.Float32
    shift_x = xo.Float32
    shift_y = xo.Float32
    rot_x = xo.Float32
    rot_y = xo.Float32
    rot_z = xo.Float32

    def copy(self):
        return ProfilePosition(
            profile_index=self.profile_index,
            s_position=self.s_position,
            shift_x=self.shift_x,
            shift_y=self.shift_y,
            rot_x=self.rot_x,
            rot_y=self.rot_y,
            rot_z=self.rot_z,
        )


class ApertureType(xo.Struct):
    """Description of the type, i.e. a section consisting of pipes (profiles).

    Parameters
    ----------
    curvature: float
        curvature of the type axis assumed to be in the horizontal plane

    positions: List[ProfilePosition]
        The list of profile positions comprising the type.
    """
    curvature = xo.Float32
    positions = ProfilePosition[:]


class TypePosition(xo.Struct):
    type_index = xo.Int32
    survey_reference_name = xo.String  # identify a point in survey
    survey_index = xo.Int32  # index of the point in the survey
    transformation = xo.Float32[4, 4]  # 3D rigid transformation matrix from the survey entry to 0 s-position of type


class ApertureModel(xo.Struct):
    type_positions = TypePosition[:]
    types = ApertureType[:]
    profiles = Profile[:]

    def __init__(
        self,
        line_name: str,
        type_positions: List[TypePosition],
        types: List[ApertureType],
        profiles: List[Profile],
        type_names: List[str],
        profile_names: List[str],
        **kwargs,
    ):
        self.line_name = line_name

        if len(type_names) != len(types):
            raise ValueError("Length of type_names and type_names must match.")

        if len(profile_names) != len(profiles):
            raise ValueError("Length of profiles and profiles must match.")

        self.type_names = type_names
        self.profile_names = profile_names

        super().__init__(type_positions=type_positions, types=types, profiles=profiles, **kwargs)

    def type_name_for_index(self, idx: int) -> str:
        return self.type_names[idx]

    def profile_name_for_index(self, idx: int) -> str:
        return self.profile_names[idx]

    def type_for_position(self, type_position: TypePosition) -> ApertureType:
        return self.types[type_position.type_index]

    def type_name_for_position(self, type_position: TypePosition) -> str:
        return self.type_name_for_index(type_position.type_index)

    def profile_for_position(self, profile_position: ProfilePosition) -> Profile:
        return self.profiles[profile_position.profile_index]

    def profile_name_for_position(self, profile_position: ProfilePosition) -> str:
        return self.profile_name_for_index(profile_position.profile_index)


class CrossSections(xo.Struct):
    count = xo.UInt32
    num_points = xo.UInt32
    s_positions = xo.Float32[:]
    type_position_indices = xo.UInt32[:]
    profile_position_indices = xo.UInt32[:]
    points = xo.Float32[:, :, 2]


class TwissData(xo.Struct):
    s = xo.Float32[:]     # s position
    x = xo.Float32[:]     # closed orbit x
    y = xo.Float32[:]     # closed orbit y
    betx = xo.Float32[:]  # beta x
    bety = xo.Float32[:]  # beta y
    dx = xo.Float32[:]    # dispersion x
    dy = xo.Float32[:]    # dispersion y
    delta = xo.Float32[:] # relative energy deviation
    gamma = xo.Float32    # relativistic gamma

    @classmethod
    def from_twiss_table(cls, particle_ref: Particles, twiss_table: TwissTable) -> 'TwissData':
        twiss_data = cls(
            s=twiss_table.s,  # s position
            x=twiss_table.x,  # closed orbit x
            y=twiss_table.y,  # closed orbit y
            betx=twiss_table.betx,  # beta x
            bety=twiss_table.bety,  # beta y
            dx=twiss_table.dx,  # dispersion x
            dy=twiss_table.dy,  # dispersion y
            delta=twiss_table.delta,  # relative energy deviation
            gamma=particle_ref.gamma0,  # relativistic gamma
        )
        return twiss_data


class BeamData(xo.Struct):
    emitx_norm = xo.Float64        # normalized emittance x
    emity_norm = xo.Float64        # normalized emittance y
    delta_rms = xo.Float64         # rms energy spread
    tol_co = xo.Float64            # tolerance for closed orbit [co_radius]
    tol_disp = xo.Float64          # tolerance for normalized dispersion [dqf]
    tol_disp_ref_dx = xo.Float64   # tolerance for reference dispersion derivative [paras_dx]
    tol_disp_ref_beta = xo.Float64 # tolerance for reference dispersion beta [betaqfx]
    tol_energy = xo.Float64        # tolerance for energy error [twiss_deltap]
    tol_beta_beating = xo.Float64  # tolerance for beta beating in sigma [beta_beating]
    halo_x = xo.Float64            # n sigma of horizontal halo
    halo_y = xo.Float64            # n sigma of vertical halo
    halo_r = xo.Float64            # n sigma of 45 degree halo
    halo_primary = xo.Float64      # n sigma of primary halo


class SurveyData(xo.Struct):
    s = xo.Float32[:]
    tangent = xo.Float32[:, 4, 4]
    angle = xo.Float32[:]
    length = xo.Float32[:]
    tilt = xo.Float32[:]

    @classmethod
    def zeros(cls, length):
        return cls(
            s=np.zeros(shape=(length,), dtype=np.float32),
            tangent=np.zeros(shape=(length, 4, 4), dtype=np.float32),
            angle=np.zeros(shape=(length,), dtype=np.float32),
            length=np.zeros(shape=(length,), dtype=np.float32),
            tilt=np.zeros(shape=(length,), dtype=np.float32),
        )

    @classmethod
    def from_survey_table(cls, survey_table: SurveyTable) -> 'SurveyData':
        s = np.zeros(shape=(len(survey_table),), dtype=np.float32)
        tangents = np.zeros(shape=(len(survey_table), 4, 4), dtype=np.float32)
        angles = np.zeros_like(s)
        lengths = np.zeros_like(s)
        tilts = np.zeros_like(s)

        for idx, row in enumerate(survey_table.rows):
            row = survey_table.rows[idx]
            s[idx] = row.s
            tangents[idx, :3, 0] = row.ex
            tangents[idx, :3, 1] = row.ey
            tangents[idx, :3, 2] = row.ez
            tangents[idx, :, 3] = np.hstack([row.X, row.Y, row.Z, 1])
            angles[idx] = row.angle
            lengths[idx] = row.length
            tilts[idx] = row.rot_s_rad

        survey_data = cls(s=s, tangent=tangents, angle=angles, length=lengths, tilt=tilts)
        return survey_data
