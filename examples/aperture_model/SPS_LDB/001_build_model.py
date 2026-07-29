import csv
import re
from itertools import repeat
from typing import Literal

import cernlayoutdb as layout
from functools import reduce
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import operator
import xdeps as xd
import xtrack as xt
import xobjects as xo
from xtrack.aperture import Aperture, ApertureBuilder
from xtrack.aperture.structures import Polygon
from xtrack.aperture.transform import transform_matrix
from warnings import warn

# Run 000_download_model.py to download the lattice, optics, and LDB snapshot file before running this script.

class LDBConverterWarning(UserWarning):
    pass


PIPE_OVERLAP_TOL = 3.1e-2

context = xo.ContextCpu(omp_num_threads='auto')


def _format_list(lst):
    if len(lst) < 3:
        return ", ".join(lst)
    return ", ".join(lst[:3]) + ", ..."

def _write_pipe_overlap_report(pipe_table, line_length, filename, s_tol=1e-6, excluded_pipe_positions=()):
    intervals = []
    pipe_info = {}
    excluded_pipe_positions = set(excluded_pipe_positions)

    for index, row in enumerate(pipe_table.rows):
        if row.name in excluded_pipe_positions:
            continue

        s_start = float(row.s_start)
        s_end = float(row.s_end)
        length = float(row.length)
        if not np.isfinite(s_start) or not np.isfinite(s_end) or not np.isfinite(length):
            continue
        if length <= s_tol:
            continue

        pipe_info[index] = row
        if s_start <= s_end:
            if s_end - s_start > s_tol:
                intervals.append((s_start, s_end, index))
        else:
            if line_length - s_start > s_tol:
                intervals.append((s_start, line_length, index))
            if s_end > s_tol:
                intervals.append((0.0, s_end, index))

    events = []
    for start, end, index in intervals:
        events.append((start, 1, index))
        events.append((end, 0, index))
    # End events first: pipe endpoints that only touch are not overlaps.
    events.sort(key=lambda event: (event[0], event[1]))

    active = set()
    prev_s = None
    segments = []
    for s, event_kind, index in events:
        if prev_s is not None and s - prev_s > s_tol and len(active) >= 2:
            segments.append((prev_s, s, tuple(sorted(active))))
        if event_kind == 0:
            active.discard(index)
        else:
            active.add(index)
        prev_s = s

    grouped_segments = []
    for start, end, active_indices in segments:
        if (grouped_segments
                and grouped_segments[-1][1] == start
                and grouped_segments[-1][2] == active_indices):
            grouped_segments[-1] = (grouped_segments[-1][0], end, active_indices)
        else:
            grouped_segments.append((start, end, active_indices))

    rows = []
    for start, end, active_indices in grouped_segments:
        overlap_length = end - start
        if overlap_length <= s_tol:
            continue

        overlapping_pipe_positions = []
        for index in active_indices:
            pipe = pipe_info[index]
            overlapping_pipe_positions.append(
                f'{pipe.name}[{pipe.pipe_name}; ref={pipe.survey_reference}; '
                f's={pipe.s_start:.12g}->{pipe.s_end:.12g}; '
                f'length={pipe.length:.12g}; '
                f'span_s={pipe.s_span_start:.12g}->{pipe.s_span_end:.12g}; '
                f'span={pipe.span:.12g}]'
            )

        rows.append({
            'overlap_length': overlap_length,
            'overlap_order': len(active_indices),
            'overlapping_pipe_positions': ' | '.join(overlapping_pipe_positions),
            's_start': start,
            's_end': end,
        })

    rows.sort(key=lambda row: (row['overlap_order'], -row['overlap_length'], row['s_start']))

    with open(filename, 'w', newline='') as fid:
        writer = csv.DictWriter(
            fid,
            fieldnames=['overlap_length', 'overlap_order', 'overlapping_pipe_positions', 's_start', 's_end'],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def _straight_pose_in_survey_frame(straight_pose, survey_reference_s):
    """Express a pose from the straight LDB frame in an Xtrack survey frame."""
    straight_reference_pose = transform_matrix(shift_z=survey_reference_s)
    pose_in_straight_reference = np.linalg.inv(straight_reference_pose) @ straight_pose

    # The straight and curved frames have the same local axes at the reference
    # point. Xtrack supplies the curved world pose through `survey_reference`,
    # so only the local residual from the straight frame is stored.
    return pose_in_straight_reference


def _pipe_middle_shift_s(pipe):
    positions_s = [position.shift_s for position in pipe.positions]
    if not positions_s:
        return 0.0
    return 0.5 * (min(positions_s) + max(positions_s))


# See https://edms.cern.ch/document/2405052/1.0
LDB_SHAPE_TO_XS = {
    'CIRCLE': ('Circle', ['radius']),
    'ELLIPSE': ('Ellipse', ['half_major', 'half_minor']),
    'OCTAGON': ('Octagon', ['half_width', 'half_height', 'half_diagonal']),
    'RACETRACK': ('Racetrack', ['half_width', 'half_height', 'half_major', 'half_minor']),
    'RECTANGLE': ('Rectangle', ['half_width', 'half_height']),
    'RECTELLIPSE': ('RectEllipse', ['half_width', 'half_height', 'half_major', 'half_minor']),
}
LDB_PARAMS = ('ellipse_a' ,'ellipse_b' ,'ellipse_c' ,'ellipse_d')

PROFILE_OVERRIDES = {
    'AP163': ('RectEllipse', 0.068, 0.0328, 0.0101, 0.0101),
}

LONGITUDINAL_PLACEMENT_PATCHES = {}

TYPE_APERTURE_PLACEMENT_PATCHES = {}

IGNORED_TRANSFORMS = re.compile(
    r'^(MB|QF|QD|LS|LO|AC|MD|TCE|MP|QM|BPV\.33108|BTVE\.61798|AEPA\.33404|AERB\.31302|BCTDC\.41435|BPMBV\.51303'
    r'|BTVE\.61831|BTVE\.61876|BPCN\.20902|VVGST\.61737|VCTSC\.61752|VVGST\.61752|VTTE\.62005|BGIHA\.51634'
    r'|BGIVA\.51674)')

ldb_model = layout.Machine.from_pickle("SPS.pickle")

# Load the SPS with its nominal curved survey
sps = xt.load('sps.json')

# Check that they can twiss
sps.twiss_default['method'] = '4d'
sps.twiss()

sv = sps.survey()

# plt.plot(sv.s, sv.X, c='blue', label='SPS', marker='o')
#
# ldb_curv = ldb_model.get_ref_curve()
# dcum = ldb_curv.dcum
# plt.plot(dcum, np.zeros_like(dcum), c='gray', label='LDB reference', marker='o')

plt.legend()
plt.title('SPS (survey)')
plt.xlabel('$Z$ = $s_{B0}$ [m]')
plt.ylabel('$X$ = $x_{B0}$ [m]')

# Build the model
builder = ApertureBuilder(line=sps)

# Build the profiles
ignored_profiles = []
for profile_name, profile in ldb_model.profiles.items():
    shape, param_names = LDB_SHAPE_TO_XS.get(profile.shape, (None, None))

    if not shape:
        ignored_profiles.append(profile_name)
        continue

    ldb_params = tuple(getattr(profile, param_name) for param_name in LDB_PARAMS[:len(param_names)])

    if profile_name in PROFILE_OVERRIDES.keys():
        warn(
            f'Overriding the shape definition for {profile_name}: was {shape} with params {ldb_params!r}.',
            LDBConverterWarning,
        )
        shape = PROFILE_OVERRIDES[profile_name][0]
        ldb_params = PROFILE_OVERRIDES[profile_name][1:]

    shape_params = dict(zip(param_names, ldb_params))

    builder.new_profile(profile_name, shape, **shape_params)

    if not builder._profiles[profile_name].shape.valid():
        # raise ValueError(f'{profile_name} has invalid shape: {builder._profiles[profile_name].shape}')
        builder._profiles.pop(profile_name)

if ignored_profiles:
    warn(
        f'Ignored {len(ignored_profiles)} profiles with no shapes: {_format_list(ignored_profiles)}',
        LDBConverterWarning,
    )

# Build the pipes
ignored_types = []
for type_name, type_ in ldb_model.types.items():
    if (aperture_straight := type_.aperture) is None:
        ignored_types.append(type_name)
        continue

    profiles = aperture_straight.aperture_alias
    offsets = (aperture_straight.offset_x, aperture_straight.offset_y, aperture_straight.offset_z)
    s_patches = TYPE_APERTURE_PLACEMENT_PATCHES.get(type_name, repeat(0, len(profiles)))

    pipe = builder.new_pipe(type_name, curvature=0)

    for profile, off_x, off_y, off_z, s_patch in zip(profiles, *offsets, s_patches):
        mad_off = layout.LDBPoint(x=off_x, y=off_y, z=off_z).to_madpoint()
        pipe.place_profile(
            profile,
            shift_s=mad_off.z + s_patch,
            shift_x=mad_off.x,
            shift_y=mad_off.y,
        )

    pipe.positions = sorted(pipe.positions, key=lambda p: p.shift_s)


if ignored_types:
    warn(
        f'Ignored {len(ignored_types)} types without apertures for this beam: {_format_list(ignored_types)}',
        LDBConverterWarning,
    )


# Map pipes to their positions
pipes_loc = {}
pipes_loc_middles = {}
ignored_transforms = []
for transform_name, transformation in ldb_model.transformations.items():
    if (target_type := transformation.target_type) is None:
        ignored_transforms.append(transform_name)
        continue

    if target_type not in builder._pipes:
        ignored_transforms.append(transform_name)
        continue

    loc = ldb_model.get_abs_points(transform_name)['MECHANICAL START'].to_madpoint()
    loc_middle = ldb_model.get_abs_points(transform_name)['MECHANICAL MIDDLE'].to_madpoint()

    s_patch = LONGITUDINAL_PLACEMENT_PATCHES.get(transform_name, 0)

    if s_patch is None:
        # Don't even include the element
        ignored_transforms.append(transform_name)
        continue

    loc.z += s_patch
    loc_middle.z += s_patch

    pipes_loc[transform_name] = loc
    pipes_loc_middles[transform_name] = loc_middle

pipes_loc = sorted(pipes_loc.items(), key=lambda x: x[1].z)

if ignored_transforms:
    warn(f'Ignored {len(ignored_transforms)} transforms without target types, whose target types are not valid '
         f'apertures, or which were excluded purposefully due to data errors: {_format_list(ignored_transforms)}',
         LDBConverterWarning)


# Refer pipe locations to survey elements
# We use the following heuristic: the reference element is the one on which the middle of the pipe falls, and we take
# side='right' which means we default to thick elements as references (ensure main dipoles are all correct)
s_mid_positions = [pipes_loc_middles[transform_name].z for transform_name, _ in pipes_loc]
sv_indices = np.searchsorted(sv.s, s_mid_positions, side='right') - 1
sv_indices = np.clip(sv_indices, 0, len(sv.s) - 1)
for ii, sv_index in enumerate(sv_indices):
    while sv_index > 0 and sv.length[sv_index] == 0:
        sv_index -= 1
    while sv_index > 0 and sv.name[sv_index].startswith('||'):
        sv_index -= 1
    sv_indices[ii] = sv_index
sv_names = [sv.name[idx] for idx in sv_indices]
assert len(set(sv.name)) == len(sv.name), "There are non-unique survey names, which might be a problem"
transform_to_sv_point = dict(zip([transform_name for transform_name, _ in pipes_loc], sv_names))

line_table = sps.get_table()
element_type_by_name = dict(zip(line_table.name, line_table.element_type))
s_center_by_name = dict(zip(line_table.name, line_table.s_center))

# Place pipes in the model
for transform_name, mad_point in pipes_loc:
    if re.match(IGNORED_TRANSFORMS, transform_name):
        continue

    type_name = ldb_model.transformations[transform_name].target_type
    pipe_to_place = type_name

    survey_ref = transform_to_sv_point[transform_name]
    if element_type_by_name[survey_ref] == 'RBend':
        # RBends have a straight body. The selected survey reference is still
        # determined by the pipe middle, but installation must be relative to
        # the magnet centre so the straight pipe body is aligned with the
        # straight magnet body rather than with the entry frame.
        pipe_middle_pose = pipes_loc_middles[transform_name].matrix
        pipe_middle_shift_s = _pipe_middle_shift_s(builder._pipes[pipe_to_place])
        mad_point_pose = (
            pipe_middle_pose
            @ transform_matrix(shift_x=-2 * sps[survey_ref]._x0_mid)  # RBend offset
            @ transform_matrix(shift_z=-pipe_middle_shift_s)
        )
        survey_ref_s = float(s_center_by_name[survey_ref])
        at = f'{survey_ref}@centre'
    else:
        mad_point_pose = mad_point.matrix
        survey_ref_s = float(sv['s', survey_ref])
        at = survey_ref

    # The LDB pose is expressed in a straight global frame, with z equal to
    # longitudinal s. Convert it to the local frame of the selected Xtrack
    # survey point; Aperture then supplies that point's curved world pose.
    from_ref_to_here = _straight_pose_in_survey_frame(
        mad_point_pose,
        survey_ref_s,
    )

    if pipe_to_place:
        builder.place_pipe(transform_name, pipe_to_place, transformation=from_ref_to_here, at=at)

aperture_model = builder.build(context=context)
aperture = Aperture(model=aperture_model, line=sps, s_tol=PIPE_OVERLAP_TOL, context=context)

p_tab = aperture.get_pipe_table()
pipe_overlap_rows = _write_pipe_overlap_report(
    p_tab,
    sps.get_length(),
    'pipe_overlaps_summary.csv',
    s_tol=PIPE_OVERLAP_TOL,
)
print(f'Wrote {len(pipe_overlap_rows)} pipe overlap rows to pipe_overlaps_summary.csv')

ax = plt.gca()

sps_sliced = sps.copy()
sps_sliced.slice_thick_elements(slicing_strategies=[
    xt.Strategy(slicing=None),
    xt.Strategy(slicing=xt.Uniform(5), element_type=xt.RBend)
])
sliced_sv = sps_sliced.survey()
plt.plot(sliced_sv.Z, sliced_sv.X, color='blue')
plt.plot(sv.Z, sv.X, color='blue', marker='o', linestyle='none')

aperture.plot_floor_projection(ax=ax, aspect='equal')
plt.title('SPS Aperture and Survey Floor Plot')
plt.show()

aperture.plot_extents(s_positions=aperture.s_around_transitions(resolution=0.1))
plt.show()

aperture.to_json('sps_aperture.json')
