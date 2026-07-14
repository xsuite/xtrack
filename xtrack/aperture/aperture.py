from __future__ import annotations

import re
from collections.abc import Collection, Iterable
from typing import Literal, cast

import numpy as np

import xobjects as xo
from xtrack.beam_elements.apertures import LimitPolygon
from xdeps.table import Table
from xobjects.context import XContext
from xtrack.twiss import TwissInit, TwissTable
from xtrack.api_categorization import GroupedAPICollector, doc_group, property_with_doc_group
from xtrack.aperture.profile_converters import (
    LimitElement, profile_from_limit_element, profile_from_madx_aperture
)
from xtrack.aperture.structures import (
    ApertureBounds, ApertureModel, BeamData, Circle, FloatType, Pipe,
    PipePosition, Profile, ProfilePolygons, ProfilePosition,
    Rectangle, RectEllipse, ShapeTypes, SurveyData, TwissData,
)
from xtrack.json import dump as json_dump
from xtrack.json import load as json_load
from xtrack.line import Line
from xtrack.progress_indicator import progress
from xtrack.survey import survey_relative_transform
from xtrack.aperture.views import (
    PipePositionsView, PipesView, ProfilesView,
)
from xtrack.aperture.transform import transform_matrix

DTypeFloat = np.dtype[FloatType._dtype]
NDArrayNx2 = np.ndarray[tuple[int, Literal[2]], DTypeFloat]
NDArrayNxMx2 = np.ndarray[tuple[int, int, Literal[2]], DTypeFloat]
HomogenousMatrix = np.ndarray[tuple[Literal[4], Literal[4]], DTypeFloat]
HomogenousMatrices = np.ndarray[tuple[int, Literal[4], Literal[4]], DTypeFloat]

SigmasCalculationEnum = Literal['bisection', 'rays', 'exact']


def _survey_is_closed(survey: Table, tol: float = 1e-6) -> bool:
    if len(survey.Z) < 2:
        return False

    dx = survey.X[-1] - survey.X[0]
    dy = survey.Y[-1] - survey.Y[0]
    dz = survey.Z[-1] - survey.Z[0]
    return dx * dx + dy * dy + dz * dz < tol


def _shortest_circular_interval(points: np.ndarray, line_length: float) -> tuple[float, float, float]:
    """Return the shortest wrapped ``s`` interval covering ``points`` on a ring.

    The returned ``(s_start, s_end, span)`` uses wrapped interval semantics:
    when the interval crosses the end of the line, ``s_start > s_end`` and the
    covered arc is understood modulo ``line_length``.
    """
    if points.size == 0:
        raise ValueError('Cannot determine a circular interval from an empty point set.')

    points = np.mod(points, line_length)
    points = np.sort(points)

    if points.size == 1:
        point = float(points[0])
        return point, point, 0.0

    gaps = np.diff(points, append=points[0] + line_length)
    gap_idx = int(np.argmax(gaps))
    s_start = float(points[(gap_idx + 1) % points.size])
    s_end = float(points[gap_idx])
    span = float((s_end - s_start) % line_length)
    return s_start, s_end, span


def _split_wrapped_s_interval(
    start: float,
    end: float,
    *,
    line_length: float,
    wrap: bool,
    s_tol: float,
) -> list[tuple[float, float]]:
    """Split an ``s`` interval into non-wrapping segments.

    If ``wrap`` is false, the interval is returned unchanged as ``[(start, end)]``.
    If ``wrap`` is true, ``start`` and ``end`` are normalised modulo
    ``line_length`` and a wrapped interval is expanded into two ordinary
    segments, ``[(start, line_length), (0, end)]``.
    """
    if wrap:
        start = float(np.mod(start, line_length))
        end = float(np.mod(end, line_length))
        if start > end + s_tol:
            return [
                (segment_start, segment_end)
                for segment_start, segment_end in (
                    (start, line_length),
                    (0.0, end),
                )
                if segment_end - segment_start > s_tol
            ]
    return [(start, end)]


_aperture_doc_groups = GroupedAPICollector([
    "Loading and Serialization",
    "Aperture Computations",
    "Introspection",
    "Visualization",
])

DEFAULT_HALO_PARAMS = {
    "emitx_norm": 3.5e-6,
    "emity_norm": 3.5e-6,
    "delta_rms": 0.0,
    "tol_co": 0.0,
    "tol_disp": 0.0,
    "tol_disp_ref": 1.8,
    "tol_disp_ref_beta": 170,
    "tol_beta_beating": 1.0,
    "halo_x": 6.0,
    "halo_y": 6.0,
    "halo_r": 6.0,
    "halo_primary": 6.0,
}


class Aperture:
    def __init__(
        self,
        line: Line,
        model: ApertureModel,
        num_profile_points: int = 128,
        halo_params: dict | None = None,
        context: XContext | None = None,
        s_tol=1e-6,
        is_ring: bool | Literal['auto'] = 'auto',
        _skip_validity_check=False,
    ):
        """Bind an aperture model to a line and precompute the derived geometry."""
        self.line = line
        self._model = model  # positioning of pipes in line frame
        self._halo_params = DEFAULT_HALO_PARAMS.copy()
        self.context = context or xo.ContextCpu()
        self.s_tol = s_tol

        self.survey = line.survey()
        self.is_ring = _survey_is_closed(self.survey) if is_ring == 'auto' else bool(is_ring)
        self._model.is_ring = int(self.is_ring)
        self._model.survey_length = self.line.get_length()

        if not _skip_validity_check:
            self._check_model_validity()

        self._survey_data = SurveyData.from_survey_table(self.survey, line, context=self.context)
        self.num_profile_points = num_profile_points

        self._aperture_bounds: ApertureBounds | None = None
        self._profile_polygons: ProfilePolygons | None = None
        self._build_aperture_bounds(check_validity=not _skip_validity_check)

        if halo_params is not None:
            self.halo_params.update(halo_params)

    @classmethod
    def _generate_doc_rst(cls, *, include_summary_table=True):
        """Generate API documentation in RST format."""
        from xtrack.api_docs import generate_grouped_class_rst

        return generate_grouped_class_rst(cls, include_summary_table=include_summary_table)

    @property_with_doc_group("Introspection")
    def halo_params(self) -> dict:
        """Dictionary of halo parameters controlling beam-envelope and aperture-sigma computations.

        The keys and their default values are:

        ========================  =======  =====================================================
        Key                       Default  Description
        ========================  =======  =====================================================
        ``emitx_norm``            3.5e-6   Normalised horizontal emittance [m·rad]
        ``emity_norm``            3.5e-6   Normalised vertical emittance [m·rad]
        ``delta_rms``             0.0      RMS momentum spread
        ``tol_co``                0.0      Closed-orbit tolerance [m]
        ``tol_disp``              0.0      Normalised dispersion tolerance [m]
        ``tol_disp_ref``          1.8      Reference dispersion derivative tolerance [m]
        ``tol_disp_ref_beta``     170      Reference dispersion beta-function [m]
        ``tol_beta_beating``      1.0      Beta-beating tolerance [sigma]
        ``halo_x``                6.0      Horizontal halo size [sigma]
        ``halo_y``                6.0      Vertical halo size [sigma]
        ``halo_r``                6.0      45° halo size [sigma]
        ``halo_primary``          6.0      Primary halo size [sigma]
        ========================  =======  =====================================================

        The dictionary is mutable; individual entries can be changed with
        ``aperture.halo_params['key'] = value`` or in bulk with
        ``aperture.halo_params.update({...})``.
        """
        return self._halo_params

    @halo_params.setter
    def halo_params(self, value: dict):
        self._halo_params = value

    @property_with_doc_group("Introspection")
    def profiles(self) -> ProfilesView:
        """Return the profile collection view."""
        return ProfilesView(self._model)

    @property_with_doc_group("Introspection")
    def pipe_positions(self) -> PipePositionsView:
        """Return the pipe-position collection view."""
        return PipePositionsView(self._model)

    @property_with_doc_group("Introspection")
    def pipes(self) -> PipesView:
        """Return the pipe collection view."""
        return PipesView(self._model)

    @doc_group("Loading and Serialization")
    def to_json(self, filename):
        """Serialize the aperture model and halo parameters to JSON."""
        json = {
            'model': self._model.to_dict(),
            'halo_params': self.halo_params,
        }
        json_dump(json, filename)

    @doc_group("Loading and Serialization")
    @classmethod
    def from_json(cls, filename, line, **kwargs):
        """Load an aperture from JSON and bind it to ``line``."""
        context = kwargs.pop('context', None)
        if context is None:
            context = getattr(line, '_context', None)
        json = json_load(filename)
        model = ApertureModel(**json['model'], _context=context)
        halo_params = json['halo_params']
        ring = kwargs.pop('ring', bool(model.is_ring))
        return cls(
            line=line,
            model=model,
            halo_params=halo_params,
            is_ring=ring,
            context=context,
            **kwargs,
        )

    @doc_group("Loading and Serialization")
    @classmethod
    def from_line_with_madx_metadata(cls, line, include_offsets=True, context=None, **kwargs):
        """Build an aperture from MAD-X layout metadata attached to a line."""
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

    @doc_group("Loading and Serialization")
    @classmethod
    def from_line_with_associated_apertures(cls, line, context=None, **kwargs):
        """Build an aperture from Xsuite elements that reference associated apertures."""
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

    @doc_group("Loading and Serialization")
    @classmethod
    def from_line_with_limits(cls, line, context=None, **kwargs):
        """Build an aperture from limit elements installed in the line."""
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
            if not isinstance(element, LimitElement):
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

    @classmethod
    def _build_aperture_model(
            cls,
            line: Line,
            pipe_indices: dict[str, int],
            pipe_list: list[Pipe],
            pipe_position_list: list[PipePosition],
            pipe_position_names: list[str],
            profile_indices: dict[str, int],
            profile_list: list[ShapeTypes],
            context: XContext,
            **kwargs,
    ) -> Aperture:
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

    @doc_group("Aperture Computations")
    def get_aperture_sigmas_at_element(
            self,
            element_name: str,
            resolution: float | None = None,
            twiss: TwissTable | None = None,
            **kwargs,
    ) -> tuple[Table, TwissTable]:
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

    @doc_group("Aperture Computations")
    def get_aperture_sigmas_at_s(
            self,
            s_positions: Iterable[float],
            twiss_init: TwissInit | None = None,
            method: SigmasCalculationEnum = 'rays',
            envelopes_num_points: int = 36,
            num_rays: int = 32,
            output_max_envelopes: bool = False,
            output_cross_sections: bool = False,
    ) -> tuple[Table, TwissTable]:
        """Compute the maximum number of sigmas at which the beam fits in the aperture at the given ``s_positions``.

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
        table = self.get_aperture_sigmas_for_twiss(
            sliced_twiss=sliced_twiss,
            method=method,
            envelopes_num_points=envelopes_num_points,
            num_rays=num_rays,
            output_max_envelopes=output_max_envelopes,
            output_cross_sections=output_cross_sections,
        )
        return table, sliced_twiss

    @doc_group("Aperture Computations")
    def get_aperture_sigmas_for_twiss(
        self,
        sliced_twiss: TwissTable,
        method: SigmasCalculationEnum = 'rays',
        envelopes_num_points: int = 36,
        num_rays: int = 32,
        output_max_envelopes: bool = False,
        output_cross_sections: bool = False,
    ) -> Table:
        """Compute the maximum aperture sigmas from an already sampled Twiss table.

        Unlike :meth:`get_aperture_sigmas_at_s`, this method does not slice the
        line or calculate Twiss parameters. Each row of ``sliced_twiss`` is used
        directly to determine the maximum beam size that fits in the aperture.

        Parameters
        ----------
        sliced_twiss
            Twiss table containing the longitudinal positions and optical
            quantities at which to compute the aperture sigmas.
        method
            Algorithm used to determine the limiting sigma:

            - ``'rays'`` estimates the limit along evenly spaced ray directions.
            - ``'exact'`` samples the halo racetrack and emits additional rays
              from those points.
            - ``'bisection'`` searches for the largest envelope polygon contained
              in the aperture polygon.
        envelopes_num_points
            Number of points used to discretise beam-envelope polygons.
        num_rays
            Number of evenly spaced ray directions used by the ``'rays'`` and
            ``'exact'`` methods.
        output_max_envelopes
            Whether to include beam-envelope polygons at the computed sigma.
        output_cross_sections
            Whether to include the interpolated aperture cross-sections.

        Returns
        -------
        table
            Table with one row per row of ``sliced_twiss`` and the following
            columns:

            - ``index``: row index.
            - ``s``: longitudinal position.
            - ``n1``: maximum number of beam sigmas that fit in the aperture.
            - ``cross_section``: aperture polygon, included when
              ``output_cross_sections`` is true.
            - ``envelope``: beam-envelope polygon at ``n1``, included when
              ``output_max_envelopes`` is true.
        """
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
        return Table(table_data, index='index')

    @doc_group("Aperture Computations")
    def get_hvd_aperture_sigmas_at_element(
            self,
            element_name: str,
            resolution: float | None = None,
            twiss: TwissTable | None = None,
    ) -> tuple[Table, TwissTable]:
        """Compute horizontal, vertical and diagonal (45°) max aperture sigmas at element ``element_name``.

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

    @doc_group("Aperture Computations")
    def get_hvd_aperture_sigmas_at_s(
            self,
            s_positions: Iterable[float],
            twiss_init: TwissInit | None = None,
    ) -> tuple[Table, TwissTable]:
        """Compute horizontal, vertical and diagonal (45°) max aperture sigmas at the given ``s_positions``.

        Parameters
        ----------
        s_positions
            Locations at which to compute the desired quantities.
        twiss_init
            Initial conditions for the twiss.

        Returns
        -------
        A two-tuple ``(table, sliced_twiss)``:
        - ``table`` is an :class:`xdeps.table.Table` with columns ``s``,
          ``n1_horizontal``, ``n1_vertical``, ``n1_diagonal``, and
          ``cross_section``.
        - ``sliced_twiss`` is the twiss table computed as part of the calculation.
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
        table = Table(
            {
                'index': np.arange(len(sliced_twiss)),
                's': sliced_twiss.s,
                'n1_horizontal': sigmas_h,
                'n1_vertical': sigmas_v,
                'n1_diagonal': sigmas_d,
                'cross_section': interpolated_points,
            },
            index='index',
        )

        return table, sliced_twiss

    @doc_group("Aperture Computations")
    def get_envelope_at_element(
            self,
            element_name: str,
            sigmas: float,
            resolution: float | None = None,
            twiss: TwissTable | None = None,
            **kwargs,
    ) -> tuple[Table, TwissTable]:
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
        - ``envelopes`` is the table returned by :meth:`get_envelope_at_s`.
        - ``sliced_twiss`` is the twiss table computed as part of the calculation.
        """
        s_positions = self._get_cuts_at_element(element_name, resolution)
        twiss_init = twiss.get_twiss_init(at_element=element_name) if twiss else None
        return self.get_envelope_at_s(s_positions, sigmas, twiss_init, **kwargs)

    @doc_group("Aperture Computations")
    def get_envelope_at_s(
            self,
            s_positions: Iterable[float],
            sigmas: float,
            twiss_init: TwissInit | None = None,
            envelopes_num_points: int = 128,
            include_aper_tols: bool = True,
            polygons: bool = True,
            extents: bool = False,
    ) -> tuple[Table, TwissTable]:
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
        polygons
            Whether to include the beam-envelope polygons in the output table.
        extents
            Whether to include the horizontal and vertical envelope extents.

        Returns
        -------
        A two-tuple ``(envelopes, sliced_twiss)``, where:
        - ``envelopes`` is a table containing the sampled longitudinal positions and
          the requested polygon and extent outputs.
        - ``sliced_twiss`` is the twiss table computed as part of the calculation.
        """
        sliced_twiss = self._sliced_twiss_at_s(s_positions=s_positions, twiss_init=twiss_init)
        envelope_table = self.get_envelope_for_twiss(
            sliced_twiss=sliced_twiss,
            sigmas=sigmas,
            envelopes_num_points=envelopes_num_points,
            include_aper_tols=include_aper_tols,
            polygons=polygons,
            extents=extents,
        )
        return envelope_table, sliced_twiss

    @doc_group("Aperture Computations")
    def get_envelope_for_twiss(
        self,
        sliced_twiss: TwissTable,
        sigmas: float,
        envelopes_num_points: int,
        include_aper_tols: bool,
        polygons: bool,
        extents: bool,
    ) -> Table:
        """Compute beam envelopes from an already sampled Twiss table.

        Unlike :meth:`get_envelope_at_s`, this method does not slice the line or
        calculate Twiss parameters. Each row of ``sliced_twiss`` is used directly
        to construct the beam envelope at the requested sigma level.

        Parameters
        ----------
        sliced_twiss
            Twiss table containing the longitudinal positions and optical
            quantities at which to compute the envelopes.
        sigmas
            Sigma level at which to evaluate the beam envelope.
        envelopes_num_points
            Number of points used to discretise each envelope polygon.
        include_aper_tols
            Whether to enlarge the beam envelope by the mechanical tolerances of
            the active aperture profile at each longitudinal position.
        polygons
            Whether to include the discretised envelope polygons in the output.
        extents
            Whether to include the minimum and maximum horizontal and vertical
            coordinates of each envelope.

        Returns
        -------
        table
            Table with one row per row of ``sliced_twiss`` and the following
            columns:

            - ``index``: row index.
            - ``s``: longitudinal position.
            - ``cross_section``: envelope polygon, included when ``polygons`` is
              true.
            - ``min_x`` and ``max_x``: horizontal extents, included when
              ``extents`` is true.
            - ``min_y`` and ``max_y``: vertical extents, included when
              ``extents`` is true.
        """
        num_slices = len(sliced_twiss.s)
        twiss_at_s = TwissData.from_twiss_table(self.line.particle_ref, sliced_twiss)
        beam_data = BeamData(**self.halo_params)

        envelopes = (
            np.zeros(shape=(num_slices, envelopes_num_points, 2), dtype=FloatType._dtype)
            if polygons else None
        )
        min_x = np.zeros(num_slices, dtype=FloatType._dtype) if extents else None
        max_x = np.zeros(num_slices, dtype=FloatType._dtype) if extents else None
        min_y = np.zeros(num_slices, dtype=FloatType._dtype) if extents else None
        max_y = np.zeros(num_slices, dtype=FloatType._dtype) if extents else None

        self._model.get_beam_envelopes_at_sigma(
            aperture_bounds=self._aperture_bounds,
            twiss_at_s=twiss_at_s,
            beam_data=beam_data,
            sigmas=sigmas,
            envelope_num_points=envelopes_num_points,
            include_aper_tols=int(include_aper_tols),
            out_envelope=envelopes,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )

        table_data = {
            'index': np.arange(num_slices),
            's': np.asarray(sliced_twiss.s, dtype=FloatType._dtype),
        }
        if polygons:
            table_data['cross_section'] = envelopes
        if extents:
            table_data.update(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)

        return Table(table_data, index='index')

    @doc_group("Aperture Computations")
    def poses_at_s(self, s_positions: Collection[float]) -> HomogenousMatrices:
        """Return a local coordinate system (each represented by a homogeneous matrix) at all ``s_positions``."""
        sv_resampled = self._survey_data.resample(s_positions)
        return sv_resampled.pose.to_nparray()

    @doc_group("Aperture Computations")
    def cross_sections_at_element(
        self,
        element_name: str,
        resolution: float | None,
        extents: bool = False,
    ) -> Table:
        """Return aperture cross-sections sampled across an element."""
        s_positions = self._get_cuts_at_element(element_name, resolution)
        return self.cross_sections_at_s(s_positions, extents=extents)

    @doc_group("Aperture Computations")
    def cross_sections_at_s(
        self,
        s_positions: Collection[float],
        extents: bool = False,
        polygons: bool = True,
    ) -> Table:
        """Return aperture cross-sections at the requested `s` positions.

        Parameters
        ----------
        s_positions
            Longitudinal positions at which to evaluate the aperture.
        extents
            Whether to include the aperture intersections with the transverse
            coordinate axes.
        polygons
            Whether to include the interpolated cross-section polygons.
        """
        s_positions = np.array(s_positions, dtype=FloatType._dtype)
        sv_resampled = self._survey_data.resample(s_positions)

        cross_sections = (
            np.zeros(shape=(len(s_positions), self.num_profile_points, 2), dtype=FloatType._dtype)
            if polygons else None
        )
        tol_r = np.zeros(shape=len(s_positions), dtype=FloatType._dtype)
        tol_x = np.zeros(shape=len(s_positions), dtype=FloatType._dtype)
        tol_y = np.zeros(shape=len(s_positions), dtype=FloatType._dtype)
        min_x = np.zeros(shape=len(s_positions), dtype=FloatType._dtype) if extents else None
        max_x = np.zeros(shape=len(s_positions), dtype=FloatType._dtype) if extents else None
        min_y = np.zeros(shape=len(s_positions), dtype=FloatType._dtype) if extents else None
        max_y = np.zeros(shape=len(s_positions), dtype=FloatType._dtype) if extents else None

        self._model.cross_sections_at_s(
            survey_at_s=sv_resampled,
            profile_polygons=self._profile_polygons,
            aperture_bounds=self._aperture_bounds,
            survey=self._survey_data,
            cross_sections=cross_sections,
            tol_r=tol_r,
            tol_x=tol_x,
            tol_y=tol_y,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )
        poses = sv_resampled.pose.to_nparray()

        table_data = {
            'index': np.arange(len(s_positions)),
            's': s_positions,
            'pose': poses,
            'tol_r': tol_r,
            'tol_x': tol_x,
            'tol_y': tol_y,
        }
        if polygons:
            table_data['cross_section'] = cross_sections
        if extents:
            table_data.update(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
        return Table(table_data, index='index')

    @doc_group("Aperture Computations")
    def get_limit_elements(self, s_positions: list[float]) -> dict[float, LimitElement]:
        """Obtain interpolated cross-sections as limit beam elements."""
        cross_sections_table = self.cross_sections_at_s(s_positions)
        limit_elements = {}
        for s, row in zip(s_positions, cross_sections_table.rows):
            cross_section = row.cross_section
            limit_poly = LimitPolygon(
                x_vertices=cross_section[:-1, 0],
                y_vertices=cross_section[:-1, 1],
            )
            limit_elements[s] = limit_poly

        return limit_elements

    @doc_group("Visualization")
    def plot_extents(
        self,
        s_positions: Collection[float],
        sigmas: float | None = None,
        twiss_init: TwissInit | None = None,
        method: SigmasCalculationEnum = 'rays',
        envelopes_num_points: int = 64,
        include_aper_tols: bool = False,
        plot_s_positions: Collection[float] | None = None,
        axs=None,
    ):
        """Plot beam-envelope and aperture extents along the beam line.

        Parameters
        ----------
        s_positions
            Longitudinal positions at which the aperture cross-sections and beam
            envelopes are evaluated.
        sigmas
            Sigma level used to build the beam envelope. If omitted, the minimum
            available aperture sigma across ``s_positions`` is computed using
            ``method``.
        twiss_init
            Twiss initial conditions forwarded to the envelope and aperture-sigma
            computations.
        method
            Method used to compute the maximum aperture sigmas when ``sigmas`` is not given.
        envelopes_num_points
            Number of points used to discretise each transverse beam envelope.
        include_aper_tols
            Whether aperture tolerances should be included in the beam-envelope
            computation.
        plot_s_positions
            Coordinates to be used on the horizontal axis. If omitted, ``s_positions``
            are used directly. This is useful when the data are evaluated at one
            set of longitudinal positions but should be displayed against another
            abscissa, for example a shifted, reversed, or externally defined coordinate.
        axs
            Two axes on which to draw the horizontal and vertical extents. If
            not provided, a new figure with two shared-x subplots is created.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure containing the plots.
        axs : sequence of matplotlib.axes.Axes
            The x- and y-extent axes, in that order.
        """
        s_positions = np.asarray(s_positions, dtype=FloatType._dtype)
        plot_s_positions = np.asarray(
            s_positions if plot_s_positions is None else plot_s_positions,
            dtype=FloatType._dtype,
        )

        order = np.argsort(s_positions)
        undo_order = np.empty_like(order)
        undo_order[order] = np.arange(len(order))
        s_sorted = s_positions[order]

        sliced_twiss = self._sliced_twiss_at_s(
            s_positions=s_sorted,
            twiss_init=twiss_init,
        )
        sigmas_was_computed = sigmas is None
        if sigmas is None:
            n1_table = self.get_aperture_sigmas_for_twiss(
                sliced_twiss=sliced_twiss,
                method=method,
            )
            sigmas = float(np.nanmin(n1_table.n1))

        envelopes_table = self.get_envelope_for_twiss(
            sliced_twiss=sliced_twiss,
            sigmas=sigmas,
            envelopes_num_points=envelopes_num_points,
            include_aper_tols=include_aper_tols,
            polygons=False,
            extents=True,
        )
        sections_table = self.cross_sections_at_s(s_sorted, extents=True, polygons=False)

        min_envel_x = np.asarray(envelopes_table.min_x, dtype=FloatType._dtype)[undo_order]
        max_envel_x = np.asarray(envelopes_table.max_x, dtype=FloatType._dtype)[undo_order]
        min_envel_y = np.asarray(envelopes_table.min_y, dtype=FloatType._dtype)[undo_order]
        max_envel_y = np.asarray(envelopes_table.max_y, dtype=FloatType._dtype)[undo_order]

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

        sigma_label = 'n_1' if sigmas_was_computed else 'n'
        fig.suptitle(
            fr"Extents and beam envelope at ${sigma_label} = {sigmas:.3g}$ and "
            fr"$s \in [{np.min(plot_s_positions):.3g}, {np.max(plot_s_positions):.3g}]$"
        )

        return fig, axs

    @doc_group("Visualization")
    def plot_transverse(
            self,
            name: str | None = None,
            s_positions: Collection[float] | None = None,
            resolution: float = 0.1,
            sigmas: float | None = None,
            method: SigmasCalculationEnum = 'rays',
            twiss_init: TwissInit | None = None,
            middle='beam',
            ax=None,
    ):
        """Display transverse aperture cross-sections and beam envelopes.

        Parameters
        ----------
        name
            Name of the element at which to plot. If given, ``s_positions`` are
            obtained from the element entry, exit, and optional resolution cuts.
        s_positions
            Longitudinal positions to plot directly. Provide either ``name`` or
            ``s_positions``.
        resolution
            The desired resolution, in metres along s, when plotting an element.
        sigmas
            The number of sigmas to plot. If ``None``, compute and plot the
            limiting ``n1`` using ``method``.
        method
            If ``sigmas`` is ``None``, method used to compute the limiting
            ``n1``.
        twiss_init
            Optional initial Twiss conditions forwarded to the envelope and
            aperture-sigma calculations.
        middle
            Whether the plot should be centred around the ``aperture`` middle, or ``beam`` reference.
        ax
            Axes object to plot on, if not given, spawn a new one.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Plot's axes object.
        """
        from matplotlib import pyplot as plt
        ax = ax or plt.gca()

        if (name is None) == (s_positions is None):
            raise ValueError("Provide exactly one of `name` or `s_positions`.")

        if name is not None:
            s_positions = self._get_cuts_at_element(name, resolution)
            title_location = name
        else:
            s_positions = np.asarray(s_positions, dtype=FloatType._dtype)
            title_location = 'requested s positions'

        if sigmas is None:
            n1_tab, _ = self.get_aperture_sigmas_at_s(
                s_positions=s_positions,
                twiss_init=twiss_init,
                method=method,
                envelopes_num_points=128,
                output_max_envelopes=True,
                output_cross_sections=True,
            )
            sigmas = float(np.min(n1_tab.n1))
            beam_tols = n1_tab.envelope
            beam_no_tols = None
            polygons = n1_tab.cross_section
            beam_label = f'envelope ({method}, min($n_1$) = {sigmas:.3f})'
            title_sigma = fr"min($n_1$) = {sigmas:.3f}"
        else:
            sliced_twiss = self._sliced_twiss_at_s(
                s_positions=s_positions,
                twiss_init=twiss_init,
            )
            beam_tols = self.get_envelope_for_twiss(
                sliced_twiss=sliced_twiss,
                sigmas=sigmas,
                envelopes_num_points=128,
                include_aper_tols=True,
                polygons=True,
                extents=False,
            )
            beam_no_tols = self.get_envelope_for_twiss(
                sliced_twiss=sliced_twiss,
                sigmas=sigmas,
                envelopes_num_points=128,
                include_aper_tols=False,
                polygons=True,
                extents=False,
            )
            beam_tols = beam_tols.cross_section
            beam_no_tols = beam_no_tols.cross_section
            profiles = self.cross_sections_at_s(s_positions=s_positions)
            polygons = profiles.cross_section
            beam_label = 'envelope (with tolerances)'
            title_sigma = fr"$n$ = {sigmas:.3f}"

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
            label = beam_label if not seen else ''
            ax.plot(pt[:, 0] - mid[0], pt[:, 1] - mid[1], c='royalblue', linestyle='-', label=label)
            seen = True

        if beam_no_tols is not None:
            seen = False
            for pt, mid in zip(beam_no_tols, middle):
                label = 'envelope (no tolerances)' if not seen else ''
                ax.plot(pt[:, 0] - mid[0], pt[:, 1] - mid[1], c='skyblue', linestyle=':', label=label)
                seen = True

        ax.set_aspect('equal')
        ax.set_title(
            fr"Transverse aperture at {title_location}, "
            fr"s $\in$ [{s_positions[0]:.2f}, {s_positions[-1]:.2f}], {title_sigma}"
        )
        ax.legend()
        return ax

    @doc_group("Visualization")
    def plot_floor_projection(
        self,
        ax=None,
        max_curve_angle_rad: float = np.deg2rad(1),
        origin: str | None = None,
        s_range: tuple[float, float] | None = None,
        aspect: Literal['auto', 'equal'] = 'auto',
    ):
        """Plot installed pipe segments projected onto the floor plane.

        Parameters
        ----------
        ax
            Axes object to plot on. If not given, use the current axes.
        max_curve_angle_rad
            Maximum angular step used to draw curved pipe boundaries and axes.
        origin
            Name of a pipe position to use as the plotting origin. When given,
            the floor projection is expressed in the local frame of that pipe
            position.
        s_range
            Longitudinal window, relative to ``origin`` when provided, used to
            restrict which pipe segments are plotted. On rings, wrapped ranges
            are handled across the end of the line.
        aspect
            Aspect ratio applied to the axes after plotting.

        Returns
        -------
        ax : matplotlib.axes.Axes
            Axes containing the floor projection.
        """
        from xtrack.aperture.plot import plot_floor_projection

        return plot_floor_projection(
            self,
            ax=ax,
            max_curve_angle_rad=max_curve_angle_rad,
            origin=origin,
            s_range=s_range,
            aspect=aspect,
        )

    def _get_cuts_at_element(self, element_name: str, resolution: float | None) -> list[float]:
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
        self._aperture_bounds.sort_by_s()
        self._aperture_bounds.reorder_for_tolerated_pipe_overlaps(
            s_tol=self.s_tol,
            is_ring=self.is_ring,
        )

        if check_validity:
            self._check_pipe_bounds_validity()

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

    def _check_pipe_bounds_validity(self):
        pipe_table = self.get_pipe_table()
        line_length = self.line.get_length()

        intervals = []
        for row in pipe_table.rows:
            if row.length <= self.s_tol:
                continue
            intervals.extend(
                (start, end, row.name)
                for start, end in _split_wrapped_s_interval(
                    row.s_start,
                    row.s_end,
                    line_length=line_length,
                    wrap=self.is_ring,
                    s_tol=self.s_tol,
                )
            )

        intervals.sort(key=lambda interval: interval[0])

        last_end = -np.inf
        last_name = None
        for start, end, name in intervals:
            if start < last_end - self.s_tol:
                raise ValueError(
                    f'Aperture model corrupted: pipe position {name} overlaps pipe position {last_name} '
                    f'around s = {start}.'
                )

            if end > last_end:
                last_end = end
                last_name = name

    @doc_group("Introspection")
    def get_bounds_table(self) -> Table:
        """Return per-profile aperture-bound information as a table.

        Returns
        -------
        bounds_table
            Table with the following columns:
            - ``name``: name of the aperture bound, formed from the pipe-position
              name and, when needed, a ``::i`` suffix identifying the profile
              order within the pipe
            - ``pipe_name``: name of the pipe in which the installed profile appears
            - ``profile_name``: name of the installed profile
            - ``s``: survey position at which the installed profile plane
              intersects the reference curve
            - ``s_start``, ``s_end``: longitudinal footprint of the installed
              profile on the survey
            - ``shape``: profile shape name
            - ``shape_param``: dictionary of profile shape parameters
        """
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

        bounds_table = Table(
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
        return bounds_table

    @doc_group("Introspection")
    def s_around_transitions(
        self,
        tol: float | None = None,
        resolution: float | None = None,
        s_range: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Return sampling positions around aperture-profile transitions.

        The positions are built from the longitudinal locations of the
        installed aperture bounds. For each stored ``s`` position, the method
        emits points at ``s - tol`` and ``s + tol``. This is useful when
        sampling quantities that can change abruptly at profile transitions.

        Parameters
        ----------
        tol
            Offset applied on both sides of each transition bound. If omitted,
            use ``self.s_tol``.
        resolution
            If provided, add a regular grid of sampling points spaced by this
            step size and union it with the transition-based points.
        s_range
            If provided, restrict the returned positions to this longitudinal
            interval. For rings, wrapped intervals are supported.

        Returns
        -------
        np.ndarray
            Sorted, unique ``s`` positions clipped to the line extent.
        """
        line_length = float(self.line.get_length())
        tol = self.s_tol if tol is None else float(tol)

        if tol < 0:
            raise ValueError('`tol` must be non-negative.')
        if resolution is not None and resolution <= 0:
            raise ValueError('`resolution` must be positive.')
        if s_range is not None and not self.is_ring and s_range[0] > s_range[1]:
            raise ValueError('Wrapped `s_range` is only supported for ring apertures.')

        bounds_table = self.get_bounds_table()
        bound_positions = np.asarray(bounds_table.s, dtype=FloatType._dtype)
        bound_positions = bound_positions[np.isfinite(bound_positions)]

        # Sample immediately on both sides of each installed aperture bound.
        s_positions = np.concatenate([bound_positions - tol, bound_positions + tol])

        if resolution is not None:
            if s_range is None:
                range_segments = [(0.0, line_length)]
            else:
                range_segments = _split_wrapped_s_interval(
                    *s_range,
                    line_length=line_length,
                    wrap=self.is_ring,
                    s_tol=tol,
                )

            # On wrapped ring intervals, build the regular grid segment by segment.
            grid = [
                np.arange(seg_start, seg_end + 0.5 * resolution, resolution, dtype=FloatType._dtype)
                for seg_start, seg_end in range_segments
            ]
            s_positions = np.concatenate([s_positions] + grid)

        s_positions = np.clip(s_positions, 0.0, line_length)

        if s_range is not None:
            range_segments = _split_wrapped_s_interval(
                float(s_range[0]),
                float(s_range[1]),
                line_length=line_length,
                wrap=self.is_ring,
                s_tol=tol,
            )
            # Keep points inside the requested window, including wrapped windows on rings
            mask = np.zeros(len(s_positions), dtype=bool)
            for seg_start, seg_end in range_segments:
                mask |= (s_positions >= seg_start) & (s_positions <= seg_end)
            s_positions = s_positions[mask]

        return np.unique(s_positions)

    @doc_group("Introspection")
    def get_wrapped_s_interval(self, start: float, end: float) -> list[tuple[float, float]]:
        """Return an ``s`` interval split at the ring boundary when needed.

        For ring apertures, ``start`` and ``end`` are interpreted modulo the
        line length. If the interval wraps around the end of the line, the
        result contains two non-wrapping segments. For non-ring apertures, the
        interval is returned unchanged.
        """
        return _split_wrapped_s_interval(
            start,
            end,
            line_length=float(self.line.get_length()),
            wrap=self.is_ring,
            s_tol=self.s_tol,
        )

    @doc_group("Introspection")
    def get_pipe_table(self):
        """Return installed-pipe interval information as a table.

        Returns
        -------
        pipe_table
            Table with the following columns:
            - ``name``: pipe-position name
            - ``pipe_name``: underlying pipe (type) name
            - ``survey_reference``: survey element used as the placement reference
            - ``s_start``, ``s_end``: interval covered by the installed profile
              centre positions
            - ``length``: length of that centre-position interval
            - ``s_span_start``, ``s_span_end``: longitudinal footprint of the
              projected aperture itself
            - ``span``: length of that aperture-footprint interval

            For rings, wrapped intervals are represented with ``s_start > s_end``
            and likewise for ``s_span_start > s_span_end``.
        """
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
        ap_s_positions = ap_bounds.s_positions.to_nparray()
        ap_s_start = ap_bounds.s_start.to_nparray()
        ap_s_end = ap_bounds.s_end.to_nparray()
        line_length = self.line.get_length()
        use_wrapped_span = self.is_ring

        s_start = np.full(table_size, np.nan, dtype=FloatType._dtype)
        s_end = np.full(table_size, np.nan, dtype=FloatType._dtype)
        length = np.full(table_size, np.nan, dtype=FloatType._dtype)
        s_span_start = np.full(table_size, np.nan, dtype=FloatType._dtype)
        s_span_end = np.full(table_size, np.nan, dtype=FloatType._dtype)
        span = np.full(table_size, np.nan, dtype=FloatType._dtype)

        bounds_order = np.argsort(pipe_position_indices, kind='stable')
        sorted_pipe_position_indices = pipe_position_indices[bounds_order]
        unique_pipe_positions, first_indices = np.unique(sorted_pipe_position_indices, return_index=True)
        last_indices = np.r_[first_indices[1:], len(bounds_order)]

        for pipe_position_index, first_idx, last_idx in zip(unique_pipe_positions, first_indices, last_indices):
            bound_indices = bounds_order[first_idx:last_idx]
            pipe_s_positions = ap_s_positions[bound_indices]
            pipe_s_start = ap_s_start[bound_indices]
            pipe_s_end = ap_s_end[bound_indices]

            if use_wrapped_span:
                # The shortest-arc inference is not robust for pipe spans > 180 degrees / half the ring.
                # This is acceptable for now because large-arc curved pipe support is not yet complete.
                pipe_s_start_val, pipe_s_end_val, pipe_length = _shortest_circular_interval(
                    pipe_s_positions,
                    line_length=line_length,
                )
                pipe_s_span_start_val, pipe_s_span_end_val, pipe_span = _shortest_circular_interval(
                    np.concatenate((pipe_s_start, pipe_s_end)),
                    line_length=line_length,
                )
            else:
                pipe_s_start_val = float(np.min(pipe_s_positions))
                pipe_s_end_val = float(np.max(pipe_s_positions))
                pipe_length = pipe_s_end_val - pipe_s_start_val
                pipe_s_span_start_val = float(np.min(pipe_s_start))
                pipe_s_span_end_val = float(np.max(pipe_s_end))
                pipe_span = pipe_s_span_end_val - pipe_s_span_start_val

            s_start[pipe_position_index] = pipe_s_start_val
            s_end[pipe_position_index] = pipe_s_end_val
            length[pipe_position_index] = pipe_length
            s_span_start[pipe_position_index] = pipe_s_span_start_val
            s_span_end[pipe_position_index] = pipe_s_span_end_val
            span[pipe_position_index] = pipe_span

        table = Table(
            data={
                'name': pipe_position_names,
                'pipe_name': pipe_names,
                'survey_reference': survey_references,
                's_start': s_start,
                's_end': s_end,
                'length': length,
                's_span_start': s_span_start,
                's_span_end': s_span_end,
                'span': span,
            },
            index='name',
        )
        order = np.argsort(s_start, kind='stable')
        return table.rows[order]

    def _sliced_twiss_at_s(
            self,
            s_positions: Iterable[float],
            twiss_init: TwissInit | None = None,
    ) -> TwissTable:
        """Get a twiss table for the line with entries at each requested `s`.

        Parameters
        ----------
        s_positions
            s-positions for the sliced twiss.
        twiss_init
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

    @classmethod
    def _guess_original_mad_name(cls, element_name: str) -> str:
        """Given a name of an element in a line, de-mangle the original MAD-X name.

        When importing a line from MAD-X, names can be mangled in two ways:
        1. ``:N`` may be added for the (N+1)-th repetition of the same element.
        2. ``/line_name`` may be appended if the same element appears in multiple sequences.
        This function only works if the elements were not named according to these patterns by the user,
        and as such is a bit of a hack. In corner cases it is best not to go through cpymad,
        but instead use the native loader with ``Aperture.from_line_with_associated_apertures``.

        Parameters
        ----------
        element_name
            Name of a beam element.
        """
        pattern = r"(?P<prefix>.*?)(?:[:]\d+)?(?:/[^/]+)?"
        match = re.fullmatch(pattern, element_name)
        return match.group('prefix')

    @classmethod
    def _get_per_pipe_madx_offsets(cls, madx_offsets):
        """Parse MAD-X imported aperture offsets metadata to obtain per-element (pipe) transformations."""
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
        """Given a shape instance created based on a MAD-X description, guess if the shape is invalid."""
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


Aperture.__doc_groups__ = _aperture_doc_groups.collect(Aperture)
