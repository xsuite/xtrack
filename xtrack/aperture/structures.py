from typing import Collection, List, Tuple, Union, get_args

import numpy as np

import xobjects as xo
from xobjects.context import XContext
from xtrack.particles import Particles
from xtrack.survey import SurveyTable
from xtrack.twiss import TwissTable

FloatType = xo.Float64


class Circle(xo.Struct):
    radius = FloatType

    def __repr__(self):
        return f'Circle(radius={self.radius})'


class Rectangle(xo.Struct):
    half_width = FloatType
    half_height = FloatType

    def __repr__(self):
        return f'Rectangle(half_width={self.half_width}, half_height={self.half_height})'


class Ellipse(xo.Struct):
    half_major = FloatType
    half_minor = FloatType

    def __repr__(self):
        return f'Ellipse(half_major={self.half_major}, half_minor={self.half_minor})'


class RectEllipse(xo.Struct):
    half_width = FloatType
    half_height = FloatType
    half_major = FloatType
    half_minor = FloatType

    def __repr__(self):
        return (f'RectEllipse(half_width={self.half_width}, half_height={self.half_height}, '
                f'half_major={self.half_major}, half_minor={self.half_minor})')


class Racetrack(xo.Struct):
    half_width = FloatType
    half_height = FloatType
    half_major = FloatType
    half_minor = FloatType

    def __repr__(self):
        return (f'RectEllipse(half_width={self.half_width}, half_height={self.half_height}, '
                f'half_major={self.half_major}, half_minor={self.half_minor})')


class Octagon(xo.Struct):
    half_width = FloatType
    half_height = FloatType
    half_diagonal = FloatType

    def __repr__(self):
        return f'Octagon(half_width={self.half_width}, half_height={self.half_height}, half_diagonal={self.half_diagonal})'


class Polygon(xo.Struct):
    vertices = FloatType[:, 2]

    def __repr__(self):
        return f'Polygon({self.vertices._shape[0]} vertices)'


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
    tol_r = FloatType
    tol_x = FloatType
    tol_y = FloatType

    _extra_c_sources = [
        '#include "xtrack/aperture/headers/profile.h"',
    ]

    _kernels = {
        'build_polygon_for_profile': xo.Kernel(
            c_name='build_polygon_for_profile',
            args=[
                xo.Arg(FloatType, pointer=True, name='points'),
                xo.Arg(xo.UInt32, name='len_points'),
                xo.Arg(xo.ThisClass, name='profile'),
            ],
        ),
    }

    def build_polygon(self, len_points: int) -> np.ndarray:
        points = np.empty((len_points, 2), dtype=FloatType._dtype)
        self.compile_kernels(only_if_needed=True)
        self._context.kernels.build_polygon_for_profile(
            points=points,
            len_points=len_points,
            profile=self,
        )
        return points

    def __repr__(self):
        tols_str = ''
        if self.tol_r != 0:
            tols_str += f', tol_r={self.tol_r}'
        if self.tol_x != 0:
            tols_str += f', tol_x={self.tol_x}'
        if self.tol_y != 0:
            tols_str += f', tol_y={self.tol_y}'
        return f'Profile({self.shape!r}{tols_str})'

    def plot(self, len_points=128, ax=None, **kwargs):
        from matplotlib import pyplot as plt
        ax = ax or plt.gca()
        ax.set_aspect('equal')
        poly = self.build_polygon(len_points)
        ax.plot(poly[:, 0], poly[:, 1], **kwargs)
        return ax


class ProfilePosition(xo.Struct):
    """Description of the placement of a profile in pipe (lab) frame.

    Parameters
    ----------
    profile_index: int
        The index identifying the profile in the associated ``Profiles`` object.
    shift_s: float
        The position along the pipe axis where this profile sits.
    shift_x: float
        The horizontal shift of the profile centre from the pipe axis.
    shift_y: float
        The vertical shift of the profile centre from the pipe axis
    rot_x_rad: float
        The rotation of the profile around the horizontal axis in radians.
    rot_y_rad: float
        The rotation of the profile around the vertical axis in radians.
    rot_s_rad: float
        The rotation of the profile around the pipe axis in radians.
    """
    profile_index = xo.Int32
    shift_s = FloatType
    shift_x = FloatType
    shift_y = FloatType
    rot_x_rad = FloatType
    rot_y_rad = FloatType
    rot_s_rad = FloatType

    def copy(self):
        return ProfilePosition(
            profile_index=self.profile_index,
            shift_s=self.shift_s,
            shift_x=self.shift_x,
            shift_y=self.shift_y,
            rot_x_rad=self.rot_x_rad,
            rot_y_rad=self.rot_y_rad,
            rot_s_rad=self.rot_s_rad,
        )


class Pipe(xo.Struct):
    """Description of the pipe, i.e. a section consisting of profiles.

    Parameters
    ----------
    curvature: float
        curvature of the pipe axis assumed to be in the horizontal plane

    positions: List[ProfilePosition]
        The list of profile positions comprising the pipe.
    """
    curvature = FloatType
    positions = ProfilePosition[:]

    def __repr__(self):
        count = len(self.positions)
        params_str = '1 profile' if count == 1 else f'{count} profiles'
        if self.curvature:
            params_str += f', curvature={self.curvature}'
        return f'<Pipe: {params_str}>'


class PipePosition(xo.Struct):
    pipe_index = xo.Int32
    survey_reference_name = xo.String  # identify a point in survey
    survey_index = xo.Int32  # index of the point in the survey
    transformation = FloatType[4, 4]  # 3D rigid transformation matrix from the survey entry to 0 shift_s of pipe


class ApertureBounds(xo.Struct):
    count = xo.UInt32
    pipe_position_indices = xo.UInt32[:]
    profile_position_indices = xo.UInt32[:]
    s_positions = FloatType[:]
    s_start = FloatType[:]
    s_end = FloatType[:]


class ProfilePolygons(xo.Struct):
    count = xo.UInt32
    len_points = xo.UInt32
    points = FloatType[:, :, 2]


class TwissData(xo.Struct):
    s = FloatType[:]     # s position
    x = FloatType[:]     # closed orbit x
    y = FloatType[:]     # closed orbit y
    betx = FloatType[:]  # beta x
    bety = FloatType[:]  # beta y
    dx = FloatType[:]    # dispersion x
    dy = FloatType[:]    # dispersion y
    delta = FloatType[:] # relative energy deviation
    gamma = FloatType    # relativistic gamma
    beta = FloatType     # relativistic beta

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
            beta=particle_ref.beta0,  # relativistic beta
        )
        return twiss_data


class BeamData(xo.Struct):
    emitx_norm = xo.Float64        # normalized emittance x
    emity_norm = xo.Float64        # normalized emittance y
    delta_rms = xo.Float64         # rms energy spread
    tol_co = xo.Float64            # tolerance for closed orbit [co_radius]
    tol_disp = xo.Float64          # tolerance for normalized dispersion [dqf]
    tol_disp_ref = xo.Float64      # tolerance for reference dispersion derivative [paras_dx]
    tol_disp_ref_beta = xo.Float64 # tolerance for reference dispersion beta [betaqfx]
    tol_beta_beating = xo.Float64  # tolerance for beta beating in sigma [beta_beating]
    halo_x = xo.Float64            # n sigma of horizontal halo
    halo_y = xo.Float64            # n sigma of vertical halo
    halo_r = xo.Float64            # n sigma of 45 degree halo
    halo_primary = xo.Float64      # n sigma of primary halo


class SurveyData(xo.Struct):
    s = FloatType[:]
    pose = FloatType[:, 4, 4]
    angle = FloatType[:]
    length = FloatType[:]
    tilt = FloatType[:]

    _extra_c_sources = [
        '#include "xtrack/aperture/headers/survey_tools.h"',
    ]

    _kernels = {
        'resample_survey_table': xo.Kernel(
            c_name='resample_survey_table',
            args=[
                xo.Arg(xo.ThisClass, name='survey'),
                xo.Arg(FloatType, pointer=True, name='s'),
                xo.Arg(xo.ThisClass, name='sliced'),
            ],
        ),
    }

    @classmethod
    def zeros(cls, length, context: XContext = None) -> 'SurveyData':
        return cls(
            s=np.zeros(shape=(length,), dtype=FloatType._dtype),
            pose=np.zeros(shape=(length, 4, 4), dtype=FloatType._dtype),
            angle=np.zeros(shape=(length,), dtype=FloatType._dtype),
            length=np.zeros(shape=(length,), dtype=FloatType._dtype),
            tilt=np.zeros(shape=(length,), dtype=FloatType._dtype),
            _context=context,
        )

    @classmethod
    def from_survey_table(cls, survey_table: SurveyTable, context: XContext = None) -> 'SurveyData':
        s = np.zeros(shape=(len(survey_table),), dtype=FloatType._dtype)
        poses = np.zeros(shape=(len(survey_table), 4, 4), dtype=FloatType._dtype)
        angles = np.zeros_like(s)
        lengths = np.zeros_like(s)
        tilts = np.zeros_like(s)

        for idx, row in enumerate(survey_table.rows):
            row = survey_table.rows[idx]
            s[idx] = row.s[0]
            poses[idx, :3, 0] = row.ex[0]
            poses[idx, :3, 1] = row.ey[0]
            poses[idx, :3, 2] = row.ez[0]
            poses[idx, :, 3] = np.hstack([row.X[0], row.Y[0], row.Z[0], 1])
            angles[idx] = row.angle[0]
            lengths[idx] = row.length[0]
            tilts[idx] = row.rot_s_rad[0]

        survey_data = cls(s=s, pose=poses, angle=angles, length=lengths, tilt=tilts, _context=context)
        return survey_data

    def resample(self, s_positions: Collection[float]) -> 'SurveyData':
        s_positions = np.array(s_positions, dtype=FloatType._dtype)
        resampled = SurveyData.zeros(len(s_positions), context=self._context)
        self.compile_kernels(only_if_needed=True)
        self._context.kernels.resample_survey_table(survey=self, s=s_positions, sliced=resampled)
        return resampled


class ApertureModel(xo.Struct):
    pipe_positions = PipePosition[:]
    pipes = Pipe[:]
    profiles = Profile[:]

    _extra_c_sources = [
        '#include "xtrack/aperture/headers/cross_sections.h"',
        '#include "xtrack/aperture/headers/beam_aperture.h"',
        '#include "xtrack/aperture/headers/survey_tools.h"',
    ]

    _kernels = {
        'build_profile_polygons': xo.Kernel(
            c_name='build_profile_polygons',
            args=[
                xo.Arg(xo.ThisClass, name='model'),
                xo.Arg(ProfilePolygons, name='profile_polygons'),
                xo.Arg(ApertureBounds, name='aperture_bounds'),
                xo.Arg(SurveyData, name='survey'),
            ],
        ),
        'cross_sections_at_s': xo.Kernel(
            c_name='cross_sections_at_s',
            args=[
                xo.Arg(SurveyData, name='survey_at_s'),
                xo.Arg(xo.ThisClass, name='model'),
                xo.Arg(ProfilePolygons, name='profile_polygons'),
                xo.Arg(ApertureBounds, name='aperture_bounds'),
                xo.Arg(SurveyData, name='survey'),
                xo.Arg(FloatType, pointer=True, name='cross_sections'),
                xo.Arg(FloatType, pointer=True, name='tol_r'),
                xo.Arg(FloatType, pointer=True, name='tol_x'),
                xo.Arg(FloatType, pointer=True, name='tol_y'),
            ],
        ),
        'compute_max_aperture_sigma_bisection': xo.Kernel(
            c_name='compute_max_aperture_sigma_bisection',
            args=[
                xo.Arg(xo.ThisClass, name='model'),
                xo.Arg(SurveyData, name='survey'),
                xo.Arg(ProfilePolygons, name='profile_polygons'),
                xo.Arg(ApertureBounds, name='aperture_bounds'),
                xo.Arg(TwissData, name='twiss_at_s'),
                xo.Arg(SurveyData, name='survey_at_s'),
                xo.Arg(BeamData, name='beam_data'),
                xo.Arg(FloatType, pointer=True, name='out_interpolated_apertures'),
                xo.Arg(xo.UInt32, name='envelope_num_points'),
                xo.Arg(FloatType, pointer=True, name='out_envelope_at_max_sigma'),
                xo.Arg(FloatType, pointer=True, name='sigmas'),
            ],
        ),
        'compute_max_aperture_sigma_rays': xo.Kernel(
            c_name='compute_max_aperture_sigma_rays',
            args=[
                xo.Arg(xo.ThisClass, name='model'),
                xo.Arg(SurveyData, name='survey'),
                xo.Arg(ProfilePolygons, name='profile_polygons'),
                xo.Arg(ApertureBounds, name='aperture_bounds'),
                xo.Arg(TwissData, name='twiss_at_s'),
                xo.Arg(SurveyData, name='survey_at_s'),
                xo.Arg(BeamData, name='beam_data'),
                xo.Arg(FloatType, pointer=True, name='out_interpolated_apertures'),
                xo.Arg(xo.UInt32, name='envelope_num_points'),
                xo.Arg(FloatType, pointer=True, name='out_envelope_at_max_sigma'),
                xo.Arg(FloatType, pointer=True, name='ray_angles'),
                xo.Arg(xo.UInt32, name='num_ray_angles'),
                xo.Arg(FloatType, pointer=True, name='sigmas'),
            ],
        ),
        'compute_max_aperture_sigma_exact': xo.Kernel(
            c_name='compute_max_aperture_sigma_exact',
            args=[
                xo.Arg(xo.ThisClass, name='model'),
                xo.Arg(SurveyData, name='survey'),
                xo.Arg(ProfilePolygons, name='profile_polygons'),
                xo.Arg(ApertureBounds, name='aperture_bounds'),
                xo.Arg(TwissData, name='twiss_at_s'),
                xo.Arg(SurveyData, name='survey_at_s'),
                xo.Arg(BeamData, name='beam_data'),
                xo.Arg(FloatType, pointer=True, name='out_interpolated_apertures'),
                xo.Arg(xo.UInt32, name='envelope_num_points'),
                xo.Arg(FloatType, pointer=True, name='out_envelope_at_max_sigma'),
                xo.Arg(FloatType, pointer=True, name='ray_angles'),
                xo.Arg(xo.UInt32, name='num_ray_angles'),
                xo.Arg(FloatType, pointer=True, name='sigmas'),
            ],
        ),
        'compute_beam_envelopes_at_sigma': xo.Kernel(
            c_name='compute_beam_envelopes_at_sigma',
            args=[
                xo.Arg(xo.ThisClass, name='model'),
                xo.Arg(ApertureBounds, name='aperture_bounds'),
                xo.Arg(TwissData, name='twiss_at_s'),
                xo.Arg(BeamData, name='beam_data'),
                xo.Arg(FloatType, name='sigmas'),
                xo.Arg(xo.UInt32, name='envelope_num_points'),
                xo.Arg(xo.Int8, name='include_aper_tols'),
                xo.Arg(FloatType, pointer=True, name='out_envelope'),
            ],
        ),
        '_points_inside_polygon': xo.Kernel(
            c_name='_points_inside_polygon',
            args=[
                xo.Arg(FloatType, pointer=True, name='points'),
                xo.Arg(FloatType, pointer=True, name='poly_points'),
                xo.Arg(xo.UInt32, name='len_points'),
                xo.Arg(xo.UInt32, name='len_poly_points'),
            ],
            ret=xo.Arg(xo.Int8),
        ),
        '_is_point_inside_polygon': xo.Kernel(
            c_name='_is_point_inside_polygon',
            args=[
                xo.Arg(FloatType, pointer=True, name='point'),
                xo.Arg(FloatType, pointer=True, name='points'),
                xo.Arg(xo.UInt32, name='len_points'),
            ],
            ret=xo.Arg(xo.Int8),
        ),
    }

    def __init__(
        self,
        pipe_positions: List[PipePosition],
        pipes: List[Pipe],
        profiles: List[Profile],
        pipe_names: List[str],
        profile_names: List[str],
        pipe_position_names: List[str],
        **kwargs,
    ):
        if len(pipe_names) != len(pipes):
            raise ValueError("Length of pipe_names and pipe_names must match.")

        if len(profile_names) != len(profiles):
            raise ValueError("Length of profiles and profiles must match.")

        if len(pipe_position_names) != len(pipe_positions):
            raise ValueError("Length of pipe_position_names and pipe_positions must match.")

        self.pipe_names = pipe_names
        self.profile_names = profile_names
        self.pipe_position_names = pipe_position_names

        super().__init__(pipe_positions=pipe_positions, pipes=pipes, profiles=profiles, **kwargs)

    def pipe_name_for_index(self, idx: int) -> str:
        return self.pipe_names[idx]

    def pipe_position_name_for_index(self, idx: int) -> str:
        return self.pipe_position_names[idx]

    def profile_name_for_index(self, idx: int) -> str:
        return self.profile_names[idx]

    def pipe_for_position(self, pipe_position: PipePosition) -> Pipe:
        return self.pipes[pipe_position.pipe_index]

    def pipe_name_for_position(self, pipe_position: PipePosition) -> str:
        return self.pipe_name_for_index(pipe_position.pipe_index)

    def pipe_position_name_for_position_index(self, idx: int) -> str:
        return self.pipe_position_name_for_index(idx)

    def profile_for_position(self, profile_position: ProfilePosition) -> Profile:
        return self.profiles[profile_position.profile_index]

    def profile_name_for_position(self, profile_position: ProfilePosition) -> str:
        return self.profile_name_for_index(profile_position.profile_index)

    def pipe_position_profile_names_for_indices(self, pipe_position_index, profile_position_index) -> Tuple[str, str]:
        pipe_pos_name = self.pipe_position_name_for_index(pipe_position_index)
        pipe_pos = self.pipe_positions[pipe_position_index]
        pipe = self.pipe_for_position(pipe_pos)
        profile_pos = pipe.positions[profile_position_index]
        profile_name = self.profile_name_for_position(profile_pos)
        return pipe_pos_name, profile_name

    def pipe_profile_names_for_indices(self, pipe_position_index, profile_position_index) -> Tuple[str, str]:
        pipe_pos = self.pipe_positions[pipe_position_index]
        pipe_name = self.pipe_name_for_position(pipe_pos)
        pipe = self.pipe_for_position(pipe_pos)
        profile_pos = pipe.positions[profile_position_index]
        profile_name = self.profile_name_for_position(profile_pos)
        return pipe_name, profile_name

    @property
    def types(self):
        return self.pipes

    @property
    def type_positions(self):
        return self.pipe_positions

    @property
    def type_names(self):
        return self.pipe_names

    @property
    def type_position_names(self):
        return self.pipe_position_names

    def type_name_for_index(self, idx: int) -> str:
        return self.pipe_name_for_index(idx)

    def type_position_name_for_index(self, idx: int) -> str:
        return self.pipe_position_name_for_index(idx)

    def type_for_position(self, type_position: PipePosition) -> Pipe:
        return self.pipe_for_position(type_position)

    def type_name_for_position(self, type_position: PipePosition) -> str:
        return self.pipe_name_for_position(type_position)

    def type_position_name_for_position_index(self, idx: int) -> str:
        return self.pipe_position_name_for_position_index(idx)

    def type_position_profile_names_for_indices(self, type_position_index, profile_position_index) -> Tuple[str, str]:
        return self.pipe_position_profile_names_for_indices(type_position_index, profile_position_index)

    def type_profile_names_for_indices(self, type_position_index, profile_position_index) -> Tuple[str, str]:
        return self.pipe_profile_names_for_indices(type_position_index, profile_position_index)

    def to_dict(self) -> dict:
        out = self._to_dict()
        out['pipe_names'] = self.pipe_names
        out['pipe_position_names'] = self.pipe_position_names
        out['profile_names'] = self.profile_names
        return out
    @classmethod
    def from_dict(cls, src: dict, context: XContext = None) -> 'ApertureModel':
        return cls(**src, _context=context)

    def build_profile_polygons(self, **kwargs) -> None:
        self.compile_kernels(only_if_needed=True)
        self._context.kernels.build_profile_polygons(model=self, **kwargs)

    def cross_sections_at_s(self, **kwargs) -> None:
        self.compile_kernels(only_if_needed=True)
        self._context.kernels.cross_sections_at_s(model=self, **kwargs)

    def compute_max_aperture_sigma_bisection(self, **kwargs) -> None:
        self.compile_kernels(only_if_needed=True)
        self._context.kernels.compute_max_aperture_sigma_bisection(model=self, **kwargs)

    def compute_max_aperture_sigma_rays(self, **kwargs) -> None:
        self.compile_kernels(only_if_needed=True)
        self._context.kernels.compute_max_aperture_sigma_rays(model=self, **kwargs)

    def compute_max_aperture_sigma_exact(self, **kwargs) -> None:
        self.compile_kernels(only_if_needed=True)
        self._context.kernels.compute_max_aperture_sigma_exact(model=self, **kwargs)

    def compute_beam_envelopes_at_sigma(self, **kwargs) -> None:
        self.compile_kernels(only_if_needed=True)
        self._context.kernels.compute_beam_envelopes_at_sigma(model=self, **kwargs)

    def _points_inside_polygon(self, **kwargs) -> bool:
        self.compile_kernels(only_if_needed=True)
        return bool(self._context.kernels._points_inside_polygon(**kwargs))

    def _is_point_inside_polygon(self, **kwargs) -> bool:
        self.compile_kernels(only_if_needed=True)
        return bool(self._context.kernels._is_point_inside_polygon(**kwargs))


ApertureType = Pipe
TypePosition = PipePosition
