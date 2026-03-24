import bisect
import re
from collections.abc import Collection
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, cast

import numpy as np
import xobjects as xo
from xdeps.table import Table
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
    ApertureBounds,
    Circle,
    FloatType,
    Profile,
    ProfilePolygons,
    ProfilePosition,
    Rectangle,
    RectEllipse,
    ShapeTypes,
    SurveyData,
    TwissData,
    TypePosition,
)
from xtrack.line import Line
from xtrack.progress_indicator import progress

DTypeFloat = np.dtype[FloatType._dtype]
NDArrayNx2 = np.ndarray[Tuple[int, Literal[2]], DTypeFloat]
NDArrayNxMx2 = np.ndarray[Tuple[int, int, Literal[2]], DTypeFloat]
HomogenousMatrix = np.ndarray[Tuple[Literal[4], Literal[4]], DTypeFloat]
HomogenousMatrices = np.ndarray[Tuple[int, Literal[4], Literal[4]], DTypeFloat]


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


def survey_relative_transform(survey, source, destination):
    src_row = survey.rows[source]
    dest_row = survey.rows[destination]

    def _row_to_matrix(row):
        matrix = np.identity(4)
        matrix[:3, :3] = row.W
        matrix[:3, 3] = row.p0
        return matrix

    src_mat = _row_to_matrix(src_row)
    dest_mat = _row_to_matrix(dest_row)

    return np.linalg.inv(src_mat) @ dest_mat



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
        line: Line,
        model: ApertureModel,
        num_profile_points: int = 128,
        halo_params: Optional[dict] = None,
        context: Optional[XContext] = None,
        s_tol=1e-6,
        _skip_validity_check=False,
    ):
        self.line = line
        self.model = model  # positioning of types in line frame
        self.halo_params = self.halo_params.copy()
        self.context = context or xo.ContextCpu()
        self.s_tol = s_tol

        self.survey = line.survey()

        if not _skip_validity_check:
            self._check_model_validity()

        self.survey_data = SurveyData.from_survey_table(self.survey, context=self.context)
        self.num_profile_points = num_profile_points

        self._aperture_bounds: Optional[ApertureBounds] = None
        self._profile_polygons: Optional[ProfilePolygons] = None
        self._build_aperture_bounds(check_validity=not _skip_validity_check)

        if halo_params is not None:
            self.halo_params.update(halo_params)

    def call_kernel(self, name, **kwargs):
        if name not in self.context.kernels:
            build_aperture_kernels(self.context)

        return self.context.kernels[name](**kwargs)

    @classmethod
    def from_line_with_madx_metadata(cls, line, include_offsets=True, context=None, **kwargs):
        survey = line.survey()
        survey_names = survey.name[:-1]  # _end_point is not an element
        name_to_sv_index = dict(zip(survey.name, range(len(survey))))
        layout_data = line.metadata['layout_data']

        if include_offsets:
            aperture_offsets = cls._get_per_type_madx_offsets(line.metadata.get('aperture_offsets', {}))
        else:
            aperture_offsets = {}

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

            offset_data = aperture_offsets.get(aper_name, {})

            if offset_data:
                rel_survey_mat = survey_relative_transform(survey, offset_data['survey_ref'], element_name)
                s_ref = rel_survey_mat[2, 3]
                matrix = transform_matrix(
                    dx=offset_data['x'],
                    dy=offset_data['y'],
                    ds=s_ref,
                )
                survey_reference_name = offset_data['survey_ref']
            else:
                matrix = np.identity(4)
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

                assert len(types) == len(profiles)  # in MAD-X we will have just one type per profile

                aper_idx = len(types)

                if element.isthick and not offset_data:
                    # Place two profiles on either side of the element
                    position_entry = ProfilePosition(profile_index=aper_idx)
                    position_exit = ProfilePosition(profile_index=aper_idx, s_position=element.length)
                    positions = [position_entry, position_exit]
                    # If no MAD-X offset data is present, the curvature follows
                    # the element
                    curvature = getattr(element, 'h', 0)
                elif element.isthick and offset_data:
                    # If MAD-X offset data is given, place profiles
                    # on the described parabola with 10cm resolution
                    length = element.length
                    positions = []

                    for s in np.linspace(0, length, max(2, int(length / 0.1))):
                        position = ProfilePosition(profile_index=aper_idx)
                        position.s_position = s
                        position.shift_x = s * offset_data['dx'] + s**2 * offset_data['ddx']
                        position.shift_y = s * offset_data['dy'] + s**2 * offset_data['ddy']
                        positions.append(position)

                    # If we have offset data, assume the type is straight
                    curvature = 0
                else:
                    # Place a single profile for a thin element
                    positions = [ProfilePosition(profile_index=aper_idx)]
                    curvature = 0

                aperture_indices[aper_name] = aper_idx

                aperture_type = ApertureType(curvature=curvature, positions=positions)
                types.append(aperture_type)
                profiles.append(profile)

            type_position = TypePosition(
                type_index=aperture_indices[aper_name],
                survey_reference_name=survey_reference_name,
                survey_index=name_to_sv_index[survey_reference_name],
                transformation=matrix,
            )
            type_positions_list.append(type_position)

        aperture = cls._build_aperture_model(
            line=line,
            type_indices=aperture_indices,
            type_list=types,
            type_position_list=type_positions_list,
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
            line=line,
            type_indices=aperture_indices,
            type_list=types,
            type_position_list=type_positions_list,
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
                profile_position.rot_s = element.rot_s_rad_no_frame
                profile_position.rot_x = element.rot_x_rad
                profile_position.rot_y = element.rot_y_rad

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
            line=line,
            type_indices=indices,
            type_list=type_list,
            type_position_list=type_positions_list,
            profile_indices=indices,
            profile_list=profiles,
            context=context,
            **kwargs,
        )
        return aperture

    def polygon_for_profile(self, profile: Profile, num_points: int) -> NDArrayNx2:
        points = np.ndarray(shape=(num_points, 2), dtype=FloatType._dtype)
        self.call_kernel('build_polygon_for_profile', points=points, len_points=num_points, profile=profile)
        return points

    @classmethod
    def _build_aperture_model(
            cls,
            line: Line,
            type_indices: Dict[str, int],
            type_list: List[ApertureType],
            type_position_list: List[TypePosition],
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
        kwargs
            Further parameters to be passed to the initialiser of `Aperture`.
        """
        if list(type_indices.values()) != list(range(len(type_list))):
            raise ValueError('Expected type_indices to be ordered by index')

        if list(profile_indices.values()) != list(range(len(profile_indices))):
            raise ValueError('Expected profile_indices to be ordered by index')

        context = context or xo.ContextCpu()

        model = ApertureModel(
            type_positions=type_position_list,
            types=type_list,
            profiles=profile_list,
            type_names=list(type_indices.keys()),
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
    ) -> Tuple[np.ndarray, TwissTable, np.ndarray, Optional[np.ndarray]]:
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
    ) -> Tuple[np.ndarray, TwissTable, np.ndarray, Optional[np.ndarray]]:
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
              A is the number of aperture points, and K is the number of bisection steps; currently K <= 20, this
              depends on the tolerance and search space set in ``beam_envelope.h``).
        envelopes_num_points:
            Only for method `bisection`: number of points to use when discretising the beam cross-section.
        num_rays:
            Only for methods `rays` and `exact`: number of evenly-spaced ray directions to sample in [0, 2 * pi).

        Returns
        -------
        A four-tuple (sigmas, sliced_twiss, aperture_polygons, envelope_at_max_sigma), where:
        - ``sigmas`` is the computed maximum number of sigmas ``n1`` at each requested location.
        - ``sliced_twiss`` is the twiss table computed as part of the calculation
        - ``aperture_polygons`` are the aperture cross-sections at each of the ``s_positions``: a numpy array of shape
          ``(len(s_positions), cross_sections_num_points, 2)``.
        - ``envelope_at_max_sigma`` are the beam cross-section polygons at the computed ``n1`` if ``bisection`` method
          was selected: a numpy array of the same shape as ``aperture_polygons``.
        """
        sliced_twiss = self._sliced_twiss_at_s(s_positions=s_positions, twiss_init=twiss_init)
        num_slices = len(sliced_twiss.s)
        twiss_at_s = TwissData.from_twiss_table(self.line.particle_ref, sliced_twiss)
        survey_at_s = self.survey_data.resample(twiss_at_s.s)
        beam_data = BeamData(**self.halo_params)
        interpolated_points = np.zeros(shape=(num_slices, self.num_profile_points, 2), dtype=FloatType._dtype)

        if method == 'bisection':
            envelope_at_max_sigma = np.zeros(shape=(num_slices, envelopes_num_points, 2), dtype=FloatType._dtype)
            sigmas = np.zeros(num_slices, dtype=FloatType._dtype)

            self.call_kernel(
                'compute_max_aperture_sigma_bisection',
                model=self.model,
                survey=self.survey_data,
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
            return sigmas, sliced_twiss, interpolated_points, envelope_at_max_sigma
        elif method == 'rays':
            ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False, dtype=FloatType._dtype)
            ray_sigmas = np.zeros((num_slices, num_rays), dtype=FloatType._dtype)

            self.call_kernel(
                'compute_max_aperture_sigma_rays',
                model=self.model,
                survey=self.survey_data,
                profile_polygons=self._profile_polygons,
                aperture_bounds=self._aperture_bounds,
                twiss_at_s=twiss_at_s,
                survey_at_s=survey_at_s,
                beam_data=beam_data,
                out_interpolated_apertures=interpolated_points,
                ray_angles=ray_angles,
                num_ray_angles=num_rays,
                sigmas=ray_sigmas,

            )
            return np.min(ray_sigmas, axis=1), sliced_twiss, interpolated_points, None
        elif method == 'exact':
            ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False, dtype=FloatType._dtype)
            sigmas = np.zeros(num_slices, dtype=FloatType._dtype)

            self.call_kernel(
                'compute_max_aperture_sigma_exact',
                model=self.model,
                survey=self.survey_data,
                profile_polygons=self._profile_polygons,
                aperture_bounds=self._aperture_bounds,
                twiss_at_s=twiss_at_s,
                survey_at_s=survey_at_s,
                beam_data=beam_data,
                out_interpolated_apertures=interpolated_points,
                ray_angles=ray_angles,
                num_ray_angles=num_rays,
                sigmas=sigmas,
            )
            return sigmas, sliced_twiss, interpolated_points, None
        else:
            raise NotImplementedError(f"Method `{method}` for getting aperture sigmas is unknown.")

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
        survey_at_s = self.survey_data.resample(twiss_at_s.s)

        beam_data = BeamData(**self.halo_params)
        interpolated_points = np.zeros(shape=(num_slices, self.num_profile_points, 2), dtype=FloatType._dtype)

        ray_angles = np.linspace(0, 2 * np.pi, 8, endpoint=False, dtype=FloatType._dtype)
        ray_sigmas = np.zeros((num_slices, 8), dtype=FloatType._dtype)

        self.call_kernel(
            'compute_max_aperture_sigma_rays',
            model=self.model,
            survey=self.survey_data,
            profile_polygons=self._profile_polygons,
            aperture_bounds=self._aperture_bounds,
            twiss_at_s=twiss_at_s,
            survey_at_s=survey_at_s,
            beam_data=beam_data,
            out_interpolated_apertures=interpolated_points,
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

        self.call_kernel(
            'compute_beam_envelopes_at_sigma',
            model=self.model,
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
        sv_resampled = self.survey_data.resample(s_positions)
        return sv_resampled.pose.to_nparray()

    def cross_sections_at_element(
        self,
        element_name: str,
        resolution: Optional[float],
        return_tolerances: bool = False,
    ) -> Table:
        s_positions = self._get_cuts_at_element(element_name, resolution)
        return self.cross_sections_at_s(s_positions, return_tolerances=return_tolerances)

    def cross_sections_at_s(
        self,
        s_positions: Collection[float],
        return_tolerances: bool = False,
    ) -> Table:
        s_positions = np.array(s_positions, dtype=FloatType._dtype)
        sv_resampled = self.survey_data.resample(s_positions)

        cross_sections = np.zeros(shape=(len(s_positions), self.num_profile_points, 2), dtype=FloatType._dtype)
        tol_r = np.zeros(shape=len(s_positions), dtype=FloatType._dtype)
        tol_x = np.zeros(shape=len(s_positions), dtype=FloatType._dtype)
        tol_y = np.zeros(shape=len(s_positions), dtype=FloatType._dtype)

        self.call_kernel(
            'cross_sections_at_s',
            survey_at_s=sv_resampled,
            model=self.model,
            profile_polygons=self._profile_polygons,
            aperture_bounds=self._aperture_bounds,
            survey=self.survey_data,
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
        }

        if return_tolerances:
            table_data['tol_r'] = tol_r
            table_data['tol_x'] = tol_x
            table_data['tol_y'] = tol_y

        return Table(table_data, index='index')

    def _get_cuts_at_element(self, element_name: str, resolution: Optional[float]) -> List[float]:
        """Get list of s positions so that the element ``element_name`` is cut with a ``resolution``."""
        element = self.line[element_name]
        s_start = self.line.get_s_position(element_name)
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
        num_cross_sections = sum(len(self.model.type_for_position(type_pos).positions) for type_pos in self.model.type_positions)
        self._aperture_bounds = ApertureBounds(
            count=num_cross_sections,
            type_position_indices=num_cross_sections,
            profile_position_indices=num_cross_sections,
            s_positions=num_cross_sections,
            s_start=num_cross_sections,
            s_end=num_cross_sections,
        )

        # Pre-allocate the profile polygons (generate once, so that we only need to compute transformations on them)
        num_profile_polys = len(self.model.profiles)
        self._profile_polygons = ProfilePolygons(
            count=num_profile_polys,
            len_points=num_points,
            points=(num_profile_polys, num_points),
        )

        cross_section_idx_iter = iter(progress(range(num_cross_sections), desc='Building cross-sections', total=num_cross_sections))

        for type_pos_idx, type_pos in enumerate(cast(Iterable[TypePosition], self.model.type_positions)):
            aper_type = self.model.type_for_position(type_pos)
            for profile_pos_idx, profile_pos in enumerate(cast(Iterable[ProfilePosition], aper_type.positions)):
                idx = next(cross_section_idx_iter)
                self._aperture_bounds.type_position_indices[idx] = type_pos_idx
                self._aperture_bounds.profile_position_indices[idx] = profile_pos_idx

        self.call_kernel(
            'build_profile_polygons',
            model=self.model,
            profile_polygons=self._profile_polygons,
            aperture_bounds=self._aperture_bounds,
            survey=self.survey_data,
        )

        if check_validity:
            self._check_aperture_bounds_validity()

    def _check_model_validity(self):
        for type_pos in self.model.type_positions:
            survey_ref_name = type_pos.survey_reference_name
            survey_ref_idx = type_pos.survey_index

            try:
                survey_at_idx = self.survey.name[survey_ref_idx]
            except IndexError:
                survey_at_idx = None

            if survey_at_idx != survey_ref_name:
                raise ValueError(
                    f'Aperture model corrupted for type position {type_pos.name}: the associate survey reference name '
                    f'`{survey_ref_name}` and index `{survey_ref_idx}` do not match. The element of the survey at the '
                    f'index is {survey_at_idx}.'
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

            type_pos_idx = self._aperture_bounds.type_position_indices[idx]
            profile_pos_idx = self._aperture_bounds.profile_position_indices[idx]
            type_name, profile_name = self.model.type_profile_names_for_indices(type_pos_idx, profile_pos_idx)

            if not (centre - left > -s_tol and right - centre > -s_tol):
                raise ValueError(
                    f'Aperture model corrupted for type {type_name} and profile {profile_name}: the '
                    f'computed s location {centre} is not inside the computed bounds [{left}, {right}]'
                )

            if last_right > left:
                raise ValueError(
                    f'Aperture model corrupted for type {type_name} and profile {profile_name}): the '
                    f'aperture bounds [{left}, {right}] overlap the preceding profile whose s_end = {last_right}'
                )

    def get_bounds_table(self):
        type_names = []
        profile_names = []
        s_positions = []
        s_starts = []
        s_ends = []
        shapes = []
        shape_params = []

        ap_bounds = self._aperture_bounds
        for i in range(ap_bounds.count):
            type_pos_idx = ap_bounds.type_position_indices[i]
            type_pos = self.model.type_positions[type_pos_idx]
            type_ = self.model.type_for_position(type_pos)
            type_name = self.model.type_name_for_position(type_pos)

            profile_pos_idx = ap_bounds.profile_position_indices[i]
            profile_pos = type_.positions[profile_pos_idx]
            profile_name = self.model.profile_name_for_position(profile_pos)
            profile = self.model.profile_for_position(profile_pos)

            shape = profile.shape

            type_names.append(type_name)
            profile_names.append(profile_name)
            s_positions.append(ap_bounds.s_positions[i])
            s_starts.append(ap_bounds.s_start[i])
            s_ends.append(ap_bounds.s_end[i])
            shapes.append(type(shape).__name__)
            shape_params.append(shape._to_dict())

        table = Table(
            data={
                'name': np.array([f'{pn}_in_{tn}' for pn, tn in zip(profile_names, type_names)], dtype=np.str_),
                'type_name': np.array(type_names, dtype=np.str_),
                'profile_name': np.array(profile_names, dtype=np.str_),
                's': np.array(s_positions, dtype=FloatType._dtype),
                's_start': np.array(s_starts, dtype=FloatType._dtype),
                's_end': np.array(s_ends, dtype=FloatType._dtype),
                'shape': np.array(shapes, dtype=object),
                'shape_param': np.array(shape_params, dtype=object),
            },
            index='name',
        )
        return table

    def _sliced_twiss_at_s(
            self,
            s_positions: Iterable[float],
            twiss_init: Optional[TwissInit] = None,
    ) -> TwissTable:
        """Get a twiss table for the line with entries at each `s_position`.

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
        full_twiss = line_sliced.twiss(init=twiss_init)

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

        return full_twiss.rows[tw_indices]

    def _find_type_positions(self, s_start: float, s_end: float) -> List[TypePosition]:
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
            corresponds to a unique occurrence of a type position along the line.
            The entries are sorted and contiguous, and if for some range
            ``(s_start, s_end)`` there is no associated type_position (i.e.
            there's a gap in the aperture model), ``type_position`` is None.
        """
        survey = self.survey
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

    @classmethod
    def _get_per_type_madx_offsets(cls, madx_offsets):
        """Parse MAD-X imported aperture offsets metadata to obtain per-element (type) transformations."""
        offsets = {}
        for section in madx_offsets.values():
            reference_name = section['reference']
            for idx, name in enumerate(section['name']):
                dx = section['dx_off'][idx]
                dy = section['dy_off'][idx]

                theta = np.atan2(dx, 1)
                phi = np.atan2(dy, np.sqrt(1 + dx ** 2))

                offsets[name] = {
                    'survey_ref': reference_name,
                    's': section['s_ip'][idx],
                    'x': section['x_off'][idx],
                    'y': section['y_off'][idx],
                    'rot_y': theta,
                    'rot_x': phi,
                    'dx': dx,
                    'dy': dy,
                    'ddx': section['ddx_off'][idx],
                    'ddy': section['ddy_off'][idx],
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
