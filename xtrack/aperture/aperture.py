import bisect
import re
from collections.abc import Collection
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, cast

import numpy as np
import xobjects as xo
from xobjects.context import XContext
from xtrack import TwissInit, TwissTable
from xtrack.aperture.kernels import build_aperture_kernels
from xtrack.aperture.profile_converters import (
    LimitTypes,
    profile_from_limit_element,
    profile_from_madx_aperture
)
from xtrack.aperture.structures import (
    ApertureModel,
    ApertureType,
    BeamData,
    CrossSections,
    Profile,
    ProfilePosition,
    ShapeTypes,
    SurveyData,
    TwissData,
    TypePosition
)
from xtrack.environment import Environment
from xtrack.progress_indicator import progress

PolygonPoints32 = np.ndarray[Tuple[int, Literal[2]], np.dtype[np.float32]]
HomogenousMatrices32 = np.ndarray[Tuple[int, Literal[4], Literal[4]], np.dtype[np.float32]]


def transform_matrix(dx=0, dy=0, ds=0, theta=0, phi=0, psi=0):
    """Generate a 3D transformation matrix.

    Parameters
    ----------
    dx, dy, ds : float
        Shifts in x, y, and s directions
    theta : float
        Rotation around the y-axis (positive s to x) in radians
    phi
        Rotation around the x-axis (positive s to y) in radians
    psi
        Rotation around the s-axis (positive y to x) in radians
    """
    s_phi, c_phi = np.sin(phi), np.cos(phi)
    s_theta, c_theta = np.sin(theta), np.cos(theta)
    s_psi, c_psi = np.sin(psi), np.cos(psi)
    matrix = np.array(
        [
            [-s_phi * s_psi * s_theta + c_psi * c_theta,
                -c_psi * s_phi * s_theta - c_theta * s_psi, c_phi * s_theta, dx],
            [c_phi * s_psi, c_phi * c_psi, s_phi, dy],
            [-c_theta * s_phi * s_psi - c_psi * s_theta,
                -c_psi * c_theta * s_phi + s_psi * s_theta, c_phi * c_theta, ds],
            [0, 0, 0, 1],
        ]
    )
    return matrix


class Aperture:
    halo_params = {
        "emitx_norm": 3.5e-6,  # normalized emittance x
        "emity_norm": 3.5e-6,  # normalized emittance y
        "delta_rms": 0.0,  # rms energy spread
        "tol_co": 0.0,  # tolerance for closed orbit
        "tol_disp": 0.0,  # tolerance for normalized dispersion
        "tol_disp_ref_dx": 1.8,  # tolerance for reference dispersion derivative
        "tol_disp_ref_beta": 170,  # tolerance for reference dispersion beta
        "tol_energy": 0.0,  # tolerance for energy error
        "tol_beta_beating": 1.0,  # tolerance for beta beating in sigma
        "halo_x": 6.0,  # n sigma of horizontal halo
        "halo_y": 6.0,  # n sigma of vertical halo
        "halo_r": 6.0,  # n sigma of 45 degree halo
        "halo_primary": 6.0,  # n sigma of primary halo
    }

    def __init__(
        self,
        env,
        model: ApertureModel,
        cross_sections,
        halo_params=None,
        context: Optional[XContext] = None,
        s_tol=1e-3,
    ):
        self.env = env
        self.model = model  # positioning of types in line frame
        self.cross_sections = cross_sections
        self.halo_params = self.halo_params.copy()
        self.context = context or xo.ContextCpu()
        self.s_tol = s_tol

        if halo_params is not None:
            self.halo_params.update(halo_params)

    def call_kernel(self, name, **kwargs):
        if name not in self.context.kernels:
            build_aperture_kernels(self.context)

        return self.context.kernels[name](**kwargs)

    @classmethod
    def from_line_with_madx_metadata(cls, line, line_name=None, context=None):
        env = line.env
        survey = line.survey()
        survey_names = survey.name[:-1]  # _end_point is not an element
        name_to_sv_index = dict(zip(survey.name, range(len(survey))))
        layout_data = line.metadata['layout_data']

        name_iter_with_progress = progress(
            survey_names,
            desc="Building apertures",
            total=len(survey_names),
        )

        profiles = []
        types = []
        aperture_indices = {}
        type_positions_list = []

        for element_name in name_iter_with_progress:
            element = line.element_dict[element_name]

            # Discard line name suffix to get the aperture name
            aper_name = cls._guess_original_mad_name(element_name)
            if aper_name not in layout_data:
                continue

            element_metadata = layout_data[aper_name]

            if 'aperture' not in element_metadata:
                continue

            if aper_name not in aperture_indices:
                shape_name, params, tols = element_metadata['aperture']
                shape = profile_from_madx_aperture(shape_name, params)

                if not shape:
                    # There is not really an aperture here, continue
                    continue

                tol_r, tol_x, tol_y = tols
                profile = Profile(shape=shape, tol_r=tol_r, tol_x=tol_x, tol_y=tol_y)

                assert len(types) == len(profiles)  # in MAD-X we will have just one type per profile

                aper_idx = len(types)
                aperture_indices[aper_name] = aper_idx

                profile_position = ProfilePosition(profile_index=aper_idx)
                offset_x, offset_y = 0, 0  # TODO: fill properly based on metadata['aperture_offset'][...]['x_off', ...]
                profile_position.shift_x = offset_x
                profile_position.shift_y = offset_y
                # TODO: any other transformations from metadata?

                if element.isthick:
                    # Place two profiles on either side of the element
                    profile_position_start = profile_position
                    profile_position_end = profile_position.copy()
                    profile_position_start.s_position = 0
                    profile_position_end.s_position = element.length
                    positions = [profile_position_start, profile_position_end]
                    curvature = getattr(element, 'h', 0)
                else:
                    # Place single profile at center of element
                    positions = [profile_position]
                    curvature = 0

                aperture_type = ApertureType(curvature=curvature, positions=positions)
                types.append(aperture_type)

                profiles.append(profile)

            # Apply element transformations to type position
            if element.transformations_active:
                matrix = transform_matrix(
                    dx=element.shift_x,
                    dy=element.shift_y,
                    ds=element.shift_s,
                    theta=element.rot_y_rad,
                    phi=element.rot_x_rad,
                    psi=element.rot_s_rad_no_frame,
                )
            else:
                matrix = np.identity(4)

            type_position = TypePosition(
                type_index=aperture_indices[aper_name],
                survey_reference_name=element_name,
                survey_index=name_to_sv_index[element_name],
                transformation=matrix,
            )
            type_positions_list.append(type_position)

        aperture = cls._build_aperture_model(
            env=env,
            line_name=line_name or line.name,
            type_indices=aperture_indices,
            type_list=types,
            type_position_list=type_positions_list,
            profile_indices=aperture_indices,
            profile_list=profiles,
            context=context,
        )
        return aperture

    @classmethod
    def from_line_with_associated_apertures(cls, line, line_name=None, context=None):
        env = line.env
        survey = line.survey()
        survey_names = survey.name[:-1]  # _end_point is not an element
        name_to_sv_index = dict(zip(survey.name, range(len(survey_names))))

        profiles = []
        types = []
        aperture_indices = {}
        type_positions_list = []

        for survey_name in progress(survey_names, desc="Building aperture data", total=len(survey_names)):
            # Discard line name suffix to get the aperture name
            element = line[survey_name]
            aper_name = getattr(element, 'name_associated_aperture', None)

            if not aper_name:
                continue

            aper_element = line.element_dict[aper_name]

            if aper_name not in aperture_indices:
                profile, offset_x, offset_y = profile_from_limit_element(aper_element)

                assert len(types) == len(profiles)  # in Xsuite with associated apertures we will have just one type per profile

                aper_idx = len(types)
                aperture_indices[aper_name] = aper_idx

                profile_position = ProfilePosition(profile_index=aper_idx)
                profile_position.shift_x = offset_x
                profile_position.shift_y = offset_y
                # TODO: any other transformations from metadata?

                if element.isthick:
                    # Place two profiles on either side of the element
                    profile_position_start = profile_position
                    profile_position_end = profile_position.copy()
                    profile_position_start.s_position = 0
                    profile_position_end.s_position = element.length
                    positions = [profile_position_start, profile_position_end]
                    curvature = getattr(element, 'h', 0)
                else:
                    # Place single profile at center of element
                    positions = [profile_position]
                    curvature = 0

                aperture_type = ApertureType(curvature=curvature, positions=positions)
                types.append(aperture_type)

                profiles.append(Profile(shape=profile))

            # Apply element transformations to type position
            if element.transformations_active:
                # TODO: Need to correctly handle the situation where both the element and the aperture are misaligned.
                #  The matrix then needs to combine the two in a correct way. Curvature will probably complicate this
                #  even more.
                raise NotImplementedError('Aperture model not yet supported with element transformations.')

            if aper_element.transformations_active:
                matrix = transform_matrix(
                    dx=aper_element.shift_x,
                    dy=aper_element.shift_y,
                    ds=aper_element.shift_s,
                    theta=aper_element.rot_y_rad,
                    phi=aper_element.rot_x_rad,
                    psi=aper_element.rot_s_rad_no_frame,
                )
            else:
                matrix = np.identity(4)

            type_position = TypePosition(
                type_index=aperture_indices[aper_name],
                survey_reference_name=survey_name,
                survey_index=name_to_sv_index[survey_name],
                transformation=matrix,
            )
            type_positions_list.append(type_position)

        aperture = cls._build_aperture_model(
            env=env,
            line_name=line_name or line.name,
            type_indices=aperture_indices,
            type_list=types,
            type_position_list=type_positions_list,
            profile_indices=aperture_indices,
            profile_list=profiles,
            context=context,
        )
        return aperture

    @classmethod
    def from_line_with_limits(cls, line, line_name=None, context=None):
        env = line.env
        survey = line.survey()
        survey_names = survey.name[:-1]  # _end_point is not a limit
        name_to_sv_index = dict(zip(survey.name, range(len(survey_names))))

        profiles = []
        type_list = []
        indices = {}
        type_positions_list = []

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
                profile_position.s_position = element.shift_s
                profile_position.shift_x = element.shift_x
                profile_position.shift_y = element.shift_y
                # TODO: Is this really how it should be??
                profile_position.rot_x = element.rot_s_rad_no_frame
                profile_position.rot_y = element.rot_x_rad
                profile_position.rot_z = element.rot_y_rad

            aperture_type = ApertureType(curvature=0, positions=[profile_position])
            type_list.append(aperture_type)

            type_position = TypePosition(
                type_index=aper_idx,
                survey_reference_name=name,
                survey_index=name_to_sv_index[name],
                transformation=np.identity(4),
            )
            type_positions_list.append(type_position)

            aper_idx += 1

        aperture = cls._build_aperture_model(
            env=env,
            line_name=line_name or line.name,
            type_indices=indices,
            type_list=type_list,
            type_position_list=type_positions_list,
            profile_indices=indices,
            profile_list=profiles,
            context=context,
        )
        return aperture

    def polygon_for_profile(self, profile: Profile, num_points: int) -> PolygonPoints32:
        points = np.ndarray(shape=(num_points, 2), dtype=np.float32)
        self.call_kernel('build_polygon_for_profile', points=points, num_points=num_points, profile=profile)
        return points

    @classmethod
    def _build_aperture_model(
            cls,
            env: Environment,
            line_name: str,
            type_indices: Dict[str, int],
            type_list: List[ApertureType],
            type_position_list: List[TypePosition],
            profile_indices: Dict[str, int],
            profile_list: List[ShapeTypes],
            context: XContext,
    ) -> 'Aperture':
        """Build the Aperture class and its comprising xobjects.

        Parameters
        ----------
        env
            The environment of the line for which the aperture model is built.
        line_name
            The name of the line for which the aperture model is built.
        type_indices
            A mapping between the name of an aperture type and its index in ``type_list``.
        type_list
            List of aperture types featured in the model.
        type_position_list
            List of aperture type positions that define the model.
        profile_indices
            A mapping between the name of an aperture type and its index in ``profile_list``.
        profile_list
            List of all profiles featured in the model. The order must be consistent with the indices used inside
            each of the type definitions in ``type_list``.
        """
        if list(type_indices.values()) != list(range(len(type_list))):
            raise ValueError('Expected type_indices to be ordered by index')

        if list(profile_indices.values()) != list(range(len(profile_indices))):
            raise ValueError('Expected profile_indices to be ordered by index')

        context = context or xo.ContextCpu()

        model = ApertureModel(
            line_name=line_name,
            type_positions=type_position_list,
            types=type_list,
            profiles=profile_list,
            type_names=list(type_indices.keys()),
            profile_names=list(profile_indices.keys()),
            _context=context,
        )

        aperture = cls(
            env=env,
            model=model,
            cross_sections=None,
            context=context,
        )

        return aperture

    def get_aperture_sigmas_at_element(
            self,
            line_name: str,
            element_name: str,
            resolution: Optional[float] = None,
            twiss: Optional[TwissTable] = None,
            **kwargs,
    ) -> Tuple[np.ndarray, TwissTable, np.ndarray, Optional[np.ndarray]]:
        """Compute the maximum number of sigmas at which the beam fits in the aperture at element ``element_name``.

        Parameters
        ----------
        line_name
            The name of the line for which the aperture model is built.
        elment_name
            The name of the element at which the sigmas should be computed.
        resolution
            The desired resolution, in meters along s, at which the sigmas should be computed. If not provided only the
            values at the entry and exit will be output.
        twiss
            Optionally provided twiss table from which to derive the initial beam parameters at the element.
        **kwargs
            Other parameters to be forwarded to ``Aperture.get_aperture_sigmas_at_s``.
        """
        s_positions = self._get_cuts_at_element(element_name, line_name, resolution)
        twiss_init = twiss.get_twiss_init(at_element=element_name) if twiss else None
        return self.get_aperture_sigmas_at_s(line_name, s_positions, twiss_init, **kwargs)

    def get_aperture_sigmas_at_s(
            self,
            line_name: str,
            s_positions: Iterable[float],
            twiss_init: Optional[TwissInit] = None,
            method: Literal['bisection', 'rays'] = 'rays',
            cross_sections_num_points: int = 36,
            envelopes_num_points: int = 36,
    ) -> Tuple[np.ndarray, TwissTable, np.ndarray, Optional[np.ndarray]]:
        """Compute the maximum number of sigmas at which the beam fits in the aperture at element ``element_name``.

        Parameters
        ----------
        line_name
            The name of the line for which the aperture model is built.
        s_positions
            List of s positions at which to calculate the sigmas.
        twiss_init
            Optionally provided initial twiss conditions.
        method
            A method to use for the computation:
            - 'rays' - the horizontal, vertical, and diagonal sigmas are computed (fast)
            - 'bisection' - the smallest number of sigmas for the beam to fit in the aperture is computed by bisecting
              on a polygon-inside-polygon problem (slow)
        cross_sections_num_points:
            Number of points to use in when discretising of the aperture profiles.
        envelopes_num_points:
            Only for method `bisection`: number of points to use when discretising the beam cross-section.
        **kwargs
            Other parameters to be forwarded to ``Aperture.get_aperture_sigmas_at_s``.

        Returns
        -------
        A four-tuple (sigmas, sliced_twiss, aperture_polygons, envelope_at_max_sigma), where:
        - ``sigmas`` is the computed maximum number of sigmas, either a single number ``n1`` or three numbers
          ``(n1_horizontal, n1_vertical, n1_diagonal)``, depending on ``method``.
        - ``sliced_twiss`` is the twiss table computed as part of the calculation
        - ``aperture_polygons`` are the aperture cross-sections at each of the ``s_positions``: a numpy array of shape
          ``(len(s_positions), cross_sections_num_points, 2)``.
        - ``envelope_at_max_sigma`` are the beam cross-section polygons at the computed ``n1`` if ``bisection`` method
          was selected: a numpy array of the same shape as ``aperture_polygons``.
        """
        line = self.env[line_name]
        line_sliced = line.copy()
        line_sliced.cut_at_s(s_positions)
        s_start, s_end = s_positions[0], s_positions[-1]

        self.cross_sections = self._build_cross_sections(line_name, cross_sections_num_points)

        sliced_twiss = line_sliced.twiss(init=twiss_init).rows[s_start:s_end:'s']

        num_slices = len(sliced_twiss.s)
        twiss_data = TwissData.from_twiss_table(line.particle_ref, sliced_twiss)
        beam_data = BeamData(**self.halo_params)
        interpolated_points = np.zeros(shape=(num_slices, self.cross_sections.num_points, 2), dtype=np.float32)

        if method == 'bisection':
            envelope_at_max_sigma = np.zeros(shape=(num_slices, envelopes_num_points, 2), dtype=np.float32)
            sigmas = np.zeros(num_slices, dtype=np.float32)

            self.call_kernel(
                'compute_max_aperture_sigma',
                model=self.model,
                cross_sections=self.cross_sections,
                twiss_data=twiss_data,
                beam_data=beam_data,
                out_interpolated_apertures=interpolated_points,
                envelope_num_points=envelopes_num_points,
                out_envelope_at_max_sigma=envelope_at_max_sigma,
                sigmas=sigmas,
            )
            return sigmas, sliced_twiss, interpolated_points, envelope_at_max_sigma
        elif method == 'rays':
            sigmas_h = np.zeros(num_slices, dtype=np.float32)
            sigmas_v = np.zeros(num_slices, dtype=np.float32)
            sigmas_d = np.zeros(num_slices, dtype=np.float32)

            self.call_kernel(
                'compute_horizontal_vertical_diagonal_aperture_sigmas',
                model=self.model,
                cross_sections=self.cross_sections,
                twiss_data=twiss_data,
                beam_data=beam_data,
                out_interpolated_apertures=interpolated_points,
                out_sigmas_h=sigmas_h,
                out_sigmas_v=sigmas_v,
                out_sigmas_d=sigmas_d,

            )
            return np.c_[sigmas_h, sigmas_v, sigmas_d], sliced_twiss, interpolated_points, None
        else:
            raise NotImplementedError(f"Method `{method}` for getting aperture sigmas is unknown.")

    def get_apertures_and_envelope_at_element(
            self,
            line_name: str,
            element_name: str,
            sigmas: float,
            resolution: Optional[float] = None,
            twiss: Optional[TwissTable] = None,
            **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray, TwissTable]:
        s_positions = self._get_cuts_at_element(element_name, line_name, resolution)
        twiss_init = twiss.get_twiss_init(at_element=element_name) if twiss else None
        return self.get_apertures_and_envelope_at_s(line_name, s_positions, sigmas, twiss_init, **kwargs)

    def get_apertures_and_envelope_at_s(
            self,
            line_name: str,
            s_positions: Iterable[float],
            sigmas: float,
            twiss_init: Optional[TwissInit] = None,
            cross_sections_num_points: int = 128,
            envelopes_num_points: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray, TwissTable]:
        line = self.env[line_name]
        line_sliced = line.copy()
        line_sliced.cut_at_s(s_positions)
        s_start, s_end = s_positions[0], s_positions[-1]

        self.cross_sections = self._build_cross_sections(line_name, cross_sections_num_points)

        sliced_twiss = line_sliced.twiss(init=twiss_init).rows[s_start:s_end:'s']
        num_slices = len(sliced_twiss.s)
        twiss_data = TwissData.from_twiss_table(line.particle_ref, sliced_twiss)
        beam_data = BeamData(**self.halo_params)
        interpolated_points = np.zeros(shape=(num_slices, self.cross_sections.num_points, 2), dtype=np.float32)

        envelopes = np.zeros(shape=(num_slices, envelopes_num_points, 2), dtype=np.float32)

        self.call_kernel(
            'compute_beam_envelopes_at_sigma',
            model=self.model,
            cross_sections=self.cross_sections,
            twiss_data=twiss_data,
            beam_data=beam_data,
            sigmas=sigmas,
            out_interpolated_apertures=interpolated_points,
            envelope_num_points=envelopes_num_points,
            out_envelope=envelopes,
        )

        return envelopes, interpolated_points, sliced_twiss

    def tangents_at_s(self, line_name: str, s_positions: Collection[float]) -> HomogenousMatrices32:
        """Return a local coordinate system (each represented by a homogeneous matrix) at all ``s_positions``."""
        tangents = np.zeros(shape=(len(s_positions), 4, 4), dtype=np.float32)
        line = self.env[line_name].copy()
        line.cut_at_s(s_positions)
        survey_sliced = line.survey()
        sv_indices = np.searchsorted(survey_sliced.s, s_positions)

        for idx, sv_idx in enumerate(sv_indices):
            row = survey_sliced.rows[sv_idx]
            tangents[idx, :3, 0] = row.ex
            tangents[idx, :3, 1] = row.ey
            tangents[idx, :3, 2] = row.ez
            tangents[idx, :, 3] = np.hstack([row.X, row.Y, row.Z, 1])

        return tangents

    def profiles_at_s(self, line_name: str, s_positions: Collection[float]) -> Tuple[PolygonPoints32, HomogenousMatrices32]:
        s_positions = np.array(s_positions, dtype=np.float32)
        shape = np.array([(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 50)], dtype=np.float32)
        placeholders = cast(PolygonPoints32, np.tile(shape, (len(s_positions), 1, 1)))

        sv_data = SurveyData.from_survey_table(self.env[line_name].survey())
        sv_sliced = SurveyData.zeros(len(s_positions))
        self.call_kernel(
            'resample_survey_table',
            survey=sv_data,
            s=np.array(s_positions, dtype=np.float32),
            sliced=sv_sliced,
        )
        return placeholders, sv_sliced.tangent.to_nparray()

    def _get_cuts_at_element(self, element_name: str, line_name: str, resolution: Optional[float]) -> List[float]:
        """Get list of s positions so that the element ``element_name`` is cut with a ``resolution``."""
        line = self.env[line_name]
        element = line[element_name]
        s_start = line.get_s_position(element_name)
        element_length = getattr(element, 'length', 0)
        s_end = s_start + element_length

        if resolution is not None:
            num_cuts = int(element_length / resolution)
            s_positions = np.linspace(s_start, s_end, num_cuts)
        else:
            s_positions = [s_start, s_end]

        return s_positions

    def _build_cross_sections(self, line_name: str, num_points: int) -> CrossSections:
        survey = self.env[line_name].survey()
        num_cross_sections = sum(len(self.model.type_for_position(type_pos).positions) for type_pos in self.model.type_positions)

        # Pre-allocate the cross-sections with the correct sizes
        cross_sections = CrossSections(
            count=num_cross_sections,
            num_points=num_points,
            s_positions=num_cross_sections,
            type_position_indices=num_cross_sections,
            profile_position_indices=num_cross_sections,
            points=(num_cross_sections, num_points),
        )

        cross_section_idx_iter = iter(progress(range(num_cross_sections), desc='Building cross-sections', total=num_cross_sections))

        for type_pos_idx, type_pos in enumerate(cast(Iterable[TypePosition], self.model.type_positions)):
            aper_type = self.model.type_for_position(type_pos)

            for profile_pos_idx, profile_pos in enumerate(cast(Iterable[ProfilePosition], aper_type.positions)):
                profile = self.model.profile_for_position(profile_pos)

                # TODO: We need to correctly handle transformations here, and if needed generate two cross-sections!
                #  (When generating two cross sections, remember to adapt the calculation of num_cross_sections above.)
                assert np.all(type_pos.transformation.to_nparray() == np.identity(4))
                assert profile_pos.rot_x == profile_pos.rot_y == profile_pos.rot_z == 0

                s_position = survey.s[type_pos.survey_index] + profile_pos.s_position

                idx = next(cross_section_idx_iter)
                cross_sections.s_positions[idx] = s_position
                cross_sections.type_position_indices[idx] = type_pos_idx
                cross_sections.profile_position_indices[idx] = profile_pos_idx

        self.call_kernel('build_profile_polygons', model=self.model, cross_sections=cross_sections)

        return cross_sections


    def _find_type_positions(self, s_start: float, s_end: float, line_name: str) -> List[TypePosition]:
        type_bounds = self._type_bounds()

        bound_idx_start = bisect.bisect_right(type_bounds, s_start, key=lambda bound: bound[0]) - 1
        bound_idx_end = bisect.bisect_left(type_bounds, s_end, lo=bound_idx_start, key=lambda bound: bound[1]) + 1

        type_positions = [bound[2] for bound in type_bounds[bound_idx_start:bound_idx_end] if bound[2] is not None]
        return type_positions


    def _type_bounds(self) -> List[Tuple[float, float, Optional[TypePosition]]]:
        """Compute the bounds of each aperture type along the line.

        Returns
        -------
        type_bounds
            List of tuples ``(s_start, s_end, type_position)`` where each entry
            corresponds to a unique occurrence of a type position along the
            line ``line_name``. The entries are sorted and contiguous, and if for
            some range ``(s_start, s_end)`` there is no associated type_position
            (i.e. there's a gap in the aperture model), ``type_position`` is None.
        """
        line_name = self.model.line_name
        line = self.env[line_name]
        survey = line.survey()
        line_length = survey.s[-1]

        type_positions = list(self.model.type_positions)
        if not type_positions:
            return [(0.0, line_length, None)]

        ref_s_list = []
        type_ranges = []

        for type_pos in type_positions:
            aperture_type = self.model.type_for_position(type_pos)
            positions = list(aperture_type.positions)

            if not positions:
                print(f"Warning: aperture type {self.model.type_name_for_position(type_pos)} has no profile positions.")
                continue

            s_positions = [float(p.s_position) for p in positions]
            if any(s_positions[i] > s_positions[i + 1] for i in range(len(s_positions) - 1)):
                raise ValueError(
                    f"Profile positions are not ordered for the type relative to {type_pos.survey_reference_name} "
                    f"(type index {type_pos.type_index})."
                )

            min_local = s_positions[0]
            max_local = s_positions[-1]

            sv_point = survey.rows[type_pos.survey_index]
            ref_s = sv_point.s[0]
            # TODO: include the transformation at some point

            ref_s_list.append(ref_s)
            type_ranges.append((ref_s + min_local, ref_s + max_local, type_pos))

        for i in range(len(ref_s_list) - 1):
            if ref_s_list[i] > ref_s_list[i + 1]:
                raise ValueError("Type positions are not ordered by increasing s.")

        type_bounds = []
        current_s = 0.0

        for start, end, type_pos in type_ranges:
            if current_s - start > self.s_tol:
                raise ValueError(f"Overlapping aperture types detected. Previous ends at {current_s}, current starts at {start}.")

            if start - current_s > self.s_tol:
                type_bounds.append((current_s, start, None))

            type_bounds.append((start, end, type_pos))
            current_s = end

        if current_s > line_length:
            raise ValueError("Aperture type bounds exceed line length.")

        if current_s < line_length:
            type_bounds.append((current_s, line_length, None))

        return type_bounds

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
