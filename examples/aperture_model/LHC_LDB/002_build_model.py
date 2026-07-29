import csv
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
from xtrack.aperture.transform import poly2d_to_homogeneous
from xtrack.survey import survey_relative_transform
from warnings import warn

# Run 000_download_model.py to download the lattice, optics, and LDB snapshot file before running this script.

class LDBConverterWarning(UserWarning):
    pass


PIPE_OVERLAP_TOL = 1e-3
PIPE_OVERLAP_ASSEMBLY_FILTER: Literal['none', 'top', 'sub'] = 'top'

context = xo.ContextCpu(omp_num_threads='auto')


def _format_list(lst):
    if len(lst) < 3:
        return ", ".join(lst)
    return ", ".join(lst[:3]) + ", ..."


def _pipe_position_ancestors(transformations, name):
    ancestors = []
    seen = {name}
    current = name

    while current in transformations:
        parent = transformations[current].ref
        if parent in seen:
            break
        if parent not in transformations:
            break

        ancestors.append(parent)
        seen.add(parent)
        current = parent

    return ancestors


def _pipe_positions_excluded_by_assembly_filter(pipe_position_names, transformations, assembly_filter):
    if assembly_filter == 'none':
        return set()
    if assembly_filter not in {'top', 'sub'}:
        raise ValueError(
            f'Unknown pipe assembly overlap filter {assembly_filter!r}; expected one of '
            f"'none', 'top', or 'sub'."
        )

    placed_names = set(pipe_position_names)
    top_assemblies = set()
    subassemblies = set()

    for name in placed_names:
        placed_ancestors = [
            ancestor for ancestor in _pipe_position_ancestors(transformations, name)
            if ancestor in placed_names
        ]
        if placed_ancestors:
            subassemblies.add(name)
            top_assemblies.update(placed_ancestors)

    if assembly_filter == 'top':
        return subassemblies
    return top_assemblies


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

ELEMENT_APERTURE_TYPE_OVERRIDES = {
    'HCACSGA001': 'I',  # ACSGA.5L4.B[12]  This doesn't seem right, most profiles are I but installed in both beams
    'HCVCDAR': 'D',  # VCDAV.1L8.X
    'HCBPMQBCZA': 'IE',  # BPMQBCZA.(4L1 | 4R5 | 4L5 | 4R1).B2
    'HCVCDAV': 'D',  # VCDAV.1L8.X
    'HCVCDAJ': 'D',  # VCDAJ.A1L2.X
    'HCVCDAW': 'D',  # VCDAW.A5L8.R
    'HCVAMXK': 'D',  # VAMXK.7R8.R
    'HCVCDAY': 'D',  # VCDAY.A5L4.B
    'HCVC5SX001': 'IE',  # VC5SX.(A1L5 | A1R5).X
    'HCVCBQRDEA110': 'IE',  # VCBQRDEA.(4R5 | A4L1 | 4R1 | A4L5).X
    'HCVMZLIFFF480': 'IE',  # VMZLIFFF.(A1R1 | A1R5 | A1L5 | A1R1).X
    'HCVSMSL038': 'IE',  # VSMSL.(B2R1 | A2L5 | A2R1 | B2L5 | A2R5 | B2L1 | A2L1 | B2R5).X
    'HCTDIS_T_0006': 'IE',  # TDIS_T.(A4R8 | B4L2)
}

PROFILE_OVERRIDES = {
    'AP207': ('Rectangle', 0.04978, 0.04978),  # ditto
    'AP163': ('RectEllipse', 0.068, 0.0328, 0.0101, 0.0101)  # Originally half_minor was zero...
}

LONGITUDINAL_PLACEMENT_PATCHES = {
    # These overrides are clean (aperture is clearly shifted and after correction the pipe is continuous within 1e-6)
    'VC8C.1R8.X': -0.96,
    'VCDRH.4R6.B': -0.7065,
    'VVGSH.6R3.B': -0.024,
    'VVGSH.B5R6.B': -0.005,
    'VSMSL.4L5.X': -0.0001,
    'VSMSL.A3L5.X': -0.0001,
    'VSMSL.3L5.X': -0.0001,
    'VSMSL.4L1.X': -0.0001,
    'VSMSL.A3L1.X': -0.0001,
    'VSMSL.3L1.X': -0.0001,
    'VCACS.5R4.B': 0.217,
    'VVGSH.A5R6.B': 0.071,
    'VVGSW.A5R3.B': 0.0425,
    'VMJNC.5R3.B': 0.02,
    'VMDBA.A7R2.B': 0.01,
    'VMDBA.A7R3.B': 0.01,
    'VMDBA.A7R1.B': 0.01,
    'VMDBA.A7R4.B': 0.01,
    'VMDBA.A7R5.B': 0.01,
    'VMDBA.A7R7.B': 0.01,
    'VMDBA.A7R8.B': 0.01,
    'VSMSL.3R5.X': 0.0001,
    'VSMSL.A3R5.X': 0.0001,
    'VSMSL.4R5.X': 0.0001,
    'VSMSL.A3R1.X': 0.0001,
    'VSMSL.3R1.X': 0.0001,
    'VSMSL.4R1.X': 0.0001,
    'BQSV.5R4.B1': 0.14,
    # Clean overrides sub-1mm
    # 'VCDLM.B5L4.B': -0.000002032933,
    # 'VCSD.E4R6.B': -0.000001733984,
    # 'VCSD.C4R6.B': -0.000001366017,
    # 'VCSD.F4R6.B': -0.000001222787,
    # 'VAMTA.I6L7.B': -0.000001024786137175173,
    # 'VMPBB.A4R8.B': -0.000001034928076424733,
    # 'VCRLU.A4L8.B': -0.000000899999078601792,
    # 'VMTBB.A4L8.B': 0.000001696211679702646,
    # These corrections are random assumptions, done to get rid of remaining overlaps. Need to be reviewed!
    'VCACS.5L4.B': 0.017,  # shift into the 17 mm gap after it; overlap range s=(9988.63800348, 9995.92900348)
    'VSSG.B5R6.B': -0.299393,  # overlap range s=(16928.9680287, 16946.8245357)
    'VMAOA.A5L4.B': 6.65,  # overlap range s=(9989.29600348, 9995.92900348)
    'VSSG.B5L6.B': -0.1481,  # overlap range s=(16402.6498287, 16404.9125287)
    'BPMQBCZB.4L1.B1': -0.0337,  # overlap range s=(26507.2013752, 26507.2350752)
    'VSCSD.4L1.B': 0.0337,  # overlap range s=(26507.2013752, 26507.2350752)
    'BPMQBCZB.4R1.B1': 0.0337,  # overlap range s=(137.599566482, 152.037270158)
    'VSCSD.4R1.B': -0.0337,  # overlap range s=(137.599566482, 152.037270158)
    'BPMQBCZB.4L5.B1': -0.0337,  # overlap range s=(13177.7594884, 13177.7931884)
    'VSCSD.4L5.B': 0.0337,  # overlap range s=(13177.7594884, 13177.7931884)
    'VSCST.A4R5.B': -0.0151,  # overlap range s=(13466.7981248, 13481.0880287)
    'VCDED.6R4.B': 0.015,  # overlap range s=(10139.7335884, 10146.2585884)
    'VMAWC.A5R4.B': 0.01,  # overlap range s=(10041.5105035, 10041.5205035)
    'VFCDO.D6R1.B': -0.008,  # overlap range s=(210.801070158, 211.121070158)
    'VVGRT.I5L2.B': 0.007,  # overlap range s=(3169.98947016, 3170.35747016)
    'VMVIFAAA.A4R5.B': 0.001,  # overlap range s=(13497.3271287, 13500.5461287)
    'VCDW.5R4.B': -0.000632,  # overlap range s=(10057.4125121, 10063.7119004)
    'VMWIIAAA.B4R1.B': -0.0003,  # overlap range s=(163.539370158, 166.285070158)
    'VCBKKAAA.B4L1.B': -0.0003,  # overlap range s=(26504.3151752, 26506.0168752)
    'VMANA.A5L4.B': 0.000295,  # overlap range s=(9894.62244228, 9897.96714794)
    'BPMWI.A5L4.B1': -0.000201,  # overlap range s=(9938.16549662, 9938.75029596)
    'VSSB.6R5.B': -0.000106,  # overlap range s=(13554.8420347, 13562.2352287)
    'VSTB.B6R5.B': 0.000106,  # overlap range s=(13554.8420347, 13562.2352287)
    'VSSB.6R1.B': -0.000106,  # overlap range s=(224.792976158, 232.186170158)
    'VSTB.B6R1.B': 0.000106,  # overlap range s=(224.792976158, 232.186170158)
    'VSSB.5L5.B': -0.0001,  # overlap range s=(13119.4618884, 13126.8550884)
    'VSSB.5R5.B': -0.0001,  # overlap range s=(13533.4420287, 13540.8352287)
    'VSTB.B5L5.B': 0.0001,  # overlap range s=(13119.4618884, 13126.8550884)
    'VSTB.B5R5.B': 0.0001,  # overlap range s=(13533.4420287, 13540.8352287)
    # Random assumptions, sub-1mm
    # 'VSSB.6L5.B': -9.4000000899500006e-05,  # overlap range s=(13098.0618824, 13105.4550884)
    # 'VSTB.B6L5.B': 9.4000000899500006e-05,  # overlap range s=(13098.0618824, 13105.4550884)
    # 'VMAAB.B5R4.B': 4.4816981244400003e-05,  # overlap range s=(10056.0025104, 10056.3124872)
    # 'VMSIR.B6L2.B': 3.2243334317199998e-05,  # overlap range s=(3117.56246289, 3122.01243382)
    # 'VCSIF.6L2.B': -3.1669210329000012e-06,  # fits between VMSIR.B6L2.B and VMSIR.A6L2.B; overlap range s=(3122.01243699, 3126.16243994)
    # 'VMSIR.A6L2.B': 2.7051764845940002e-05,  # fits before VCSIE.6L2.B; overlap range s=(3126.16247218, 3130.61247016)
    # 'VMSIP.6L2.B': 2.0904661141699999e-05,  # overlap range s=(3130.91247055, 3135.36245281)
    # 'VCDWH.B5R4.B': 2.0737155864499999e-05,  # overlap range s=(10070.4625095, 10076.6124989)
    # 'BSRTM.5R4.B1': -1.8762046238400001e-05,  # overlap range s=(10069.8625275, 10070.8625095)
    # 'VCSIE.6L2.B': -8.3564964370499996e-06,  # overlap range s=(3126.46247556, 3130.91247016)
    # 'VMAPB.A6R8.B': -6.0745951486799998e-06,  # overlap range s=(23512.6014691, 23512.6014752)
    # 'VMAPB.C6R8.B': -4.85845885123e-06,  # overlap range s=(23521.5014704, 23521.5014752)
    # 'VMSDU.A4R6.B': 3.8907965063100004e-06,  # overlap range s=(16698.4775259, 16698.4775298)
    # 'BPMWB.4R8.B1': 2.5683111744e-06,  # overlap range s=(23429.8114659, 23430.2764636)
    # 'VMACA.A5R4.B': -0.000798,  # overlap range s=(10080.767198135, 10080.766400247)
    # Things that cannot be fixed by simple single-element shifts we just remove:
    'BQKV.6L4.B1': None,
    'VCBIIAAA.A6L4.B': None,
    'VMABD.4L8.B': None,
    'VMAIIAAH.4R1.B': None,
    'VVGST.B1L2.X': None,
    'BSRTRB.5L4.B1': None,
    # Sub-1mm overlaps that cannot be fixed by a simple shift
    # - VMABD.4L8.B/VCRLV.A4L8.B overlap range s = (23195.2253395, 23196.1203406)
    # - VVGST.B1L2.X/VMABD.B1L2.X overlap range s = (3313.16854079, 3313.16954079)
    # - VCBIIAAA.E4R1.B/VMAIIAAH.4R1.B overlap range s = (163.759070158, 166.584070158)
    # - VCBIIAAA.5R4.B/VMACA.A5R4.B overlap range s = (10079.2575391, 10080.9667412)
    # - VCBIIAAE.A4L5.B/VMWIIAAA.A4L5.B overlap range s = (13170.5601884, 13171.3997884)
    # - VMWIIAAA.A4R5.B/VCBIIAAE.A4R5.B overlap range s = (13487.4843287, 13488.3239287)
    # - VCBKKAAA.A4L5.B/VMBLKAAK.A4L5.B overlap range s = (13174.7182884, 13176.5749884)
    # - VMWIIAAA.C4L5.B/VCBIIAAA.A4L5.B overlap range s = (13165.8386884, 13165.8389884)
    # - VMBKIAAA.A4R5.B/VVGSC.D4R5.B overlap range s = (13484.1661287, 13484.4508287)
    # - VVGSC.E4R5.B/VMWIIAAA.4R5.B overlap range s = (13491.3574287, 13491.6621287)
    # - VCDAY.A5L4.B/VMAAB.B5L4.B overlap range s = (9898.53864766, 9901.92245093)
    # - VCTCC.4L2.X/VMJQQDDA.4L2.X overlap range s = (3240.74564428, 3246.62751607)
    # - VVGSW.D5R4.B/VMANC.A5R4.B overlap range s = (10055.3274991, 10055.7024376)
    # - VCSIM.A6R8.B/VMZBA.6R8.B overlap range s = (23508.1514629, 23508.1514752)
    # - VCTCW.6L2.B/VCSIG.6L2.B overlap range s = (3117.36247016, 3121.71246585)
    # - VCSIM.E6R8.B/VMZAF.6R8.B overlap range s = (23525.9514766, 23530.4014752)
    # - VCSIF.6L2.B/VMSIR.A6L2.B overlap range s = (3122.01246606, 3126.4624672)
}

TYPE_APERTURE_PLACEMENT_PATCHES = {
    # These are guesses to remove overlaps
    'HCVMAAQ': (0, -0.01),
}

# Profile examples for each shape:
# CIRCLE e.g. <Profile AP0493 CIRCLE, 0.178,0.178>
# ELLIPSE e.g. <Profile AP004 ELLIPSE, 0.052,0.03>
# OCTAGON e.g. <Profile AP249 OCTAGON, 0.0,0.0>
# RACETRACK e.g. <Profile AP033 RACETRACK, 0.128,0.052>
# RECTANGLE e.g. <Profile AP0651 RECTANGLE, 0.06,0.06>
# RECTELLIPSE e.g. <Profile AP076 RECTELLIPSE, 0.066,0.1>
# UNDEFINED e.g. <Profile AP016 UNDEFINED, 0.332,0.12>

ldb_model = layout.Machine.from_pickle("LHC.pickle")

# Load and straighten the LHC
lattice = xt.load('lhc.json')
lattice.vars.load('opt_6000.madx')

b1 = lattice.b1
b2 = lattice.b2

# Check that they can twiss
b1.twiss4d()
b2.twiss4d()

b1.regenerate_from_composer()
b2.regenerate_from_composer()

angle_before = lattice['a.mb']
ds_before = lattice['ds']

lattice.vars['a.mb'] = 0
lattice.vars['ds'] = 0

b1.end_compose()
b2.end_compose()

sv_b1_straight = b1.survey()
sv_b2_straight = b2.survey()


def beam_inner_outer(s, sv):
    x = np.interp(s, sv.s, sv.x)

    if x > 1e-10:
        return 'E'

    if x < -1e-10:
        return 'I'

    return None


plt.plot(sv_b1_straight.Z, sv_b1_straight.X, c='red', label='B1', marker='o')
plt.plot(sv_b2_straight.Z, sv_b2_straight.X, c='blue', label='B2', marker='o')

ldb_curv = ldb_model.get_ref_curve()
dcum = ldb_curv.dcum
plt.plot(dcum, np.zeros_like(dcum), c='gray', label='B0 (midbeam)', marker='o')

plt.legend()
plt.title('HL-LHC19 (straightened)')
plt.xlabel('$Z$ = $s_{B0}$ [m]')
plt.ylabel('$X$ = $x_{B0}$ [m]')

# Build the model
builder = ApertureBuilder(line=b1)

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
        raise ValueError(f'{profile_name} has invalid shape: {builder._profiles[profile_name].shape}')

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

    inner_outer = reduce(operator.or_, (set(ie) if ie else {None} for ie in aperture_straight.element_type_aperture), set())
    if 'E' not in inner_outer and None not in inner_outer:
        ignored_types.append(type_name)
        continue

    element_type_aperture = aperture_straight.element_type_aperture

    # Sanity check the element type apertures, and apply element type overrides if deemed as inconsistent
    if set(element_type_aperture) <= {'E', 'I', 'IE'}:
        # If the profiles are not all assigned to the same beam, deem the pipe beam specific
        beam_specific_type = len(set(element_type_aperture)) > 1
    elif set(element_type_aperture) == {None}:
        beam_specific_type = False
    else:
        message = (
            f'Do not know how to handle {type_name}, where only some profiles are assigned to the specific beam '
            f'(some profiles are marked with I/E/IE element_type_aperture, while others are None: either the former '
            f'or the latter is expected): {element_type_aperture!r}.'
        )
        if type_name not in ELEMENT_APERTURE_TYPE_OVERRIDES:
            raise ValueError(message)

        override = ELEMENT_APERTURE_TYPE_OVERRIDES[type_name]
        warn(message + f' Will override according to the prescription: {override!r}.', LDBConverterWarning)

        # Apply overrides
        if override == 'D':
            override = None  # No one knows what "D" stands for...

        element_type_aperture = [override] * len(aperture_straight.element_type_aperture)
        beam_specific_type = override is not None

    profiles = aperture_straight.aperture_alias
    offsets = (aperture_straight.offset_x, aperture_straight.offset_y, aperture_straight.offset_z)
    s_patches = TYPE_APERTURE_PLACEMENT_PATCHES.get(type_name, repeat(0, len(profiles)))

    if not beam_specific_type:
        pipes = [(None, builder.new_pipe(type_name, curvature=0))]
    else:
        pipes = [
            ('I', builder.new_pipe(type_name + '/I', curvature=0)),
            ('E', builder.new_pipe(type_name + '/E', curvature=0)),
        ]

    for profile, side, off_x, off_y, off_z, s_patch in zip(profiles, element_type_aperture, *offsets, s_patches):
        mad_off = layout.LDBPoint(x=off_x, y=off_y, z=off_z).to_madpoint()
        for side_marker, pipe in pipes:
            if side_marker is None or side is None or side_marker in side:
                pipe.place_profile(
                    profile,
                    shift_s=mad_off.z + s_patch,
                    shift_x=mad_off.x,
                    shift_y=mad_off.y,
                )

    for _, pipe in pipes:
        builder._pipes[pipe.name].positions = sorted(builder._pipes[pipe.name].positions, key=lambda p: p.shift_s)


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

excluded_pipe_positions = _pipe_positions_excluded_by_assembly_filter(
    [transform_name for transform_name, _ in pipes_loc],
    ldb_model.transformations,
    PIPE_OVERLAP_ASSEMBLY_FILTER,
)
if excluded_pipe_positions:
    pipes_loc = [
        (transform_name, loc)
        for transform_name, loc in pipes_loc
        if transform_name not in excluded_pipe_positions
    ]
    for transform_name in excluded_pipe_positions:
        pipes_loc_middles.pop(transform_name, None)


if ignored_transforms:
    warn(f'Ignored {len(ignored_transforms)} transforms without target types, whose target types are not valid '
         f'apertures, or which were excluded purposefully due to data errors: {_format_list(ignored_transforms)}',
         LDBConverterWarning)
if excluded_pipe_positions:
    warn(
        f'Excluded {len(excluded_pipe_positions)} pipe positions using assembly filter '
        f'{PIPE_OVERLAP_ASSEMBLY_FILTER!r}: {_format_list(sorted(excluded_pipe_positions))}',
        LDBConverterWarning,
    )


# Refer pipe locations to survey elements
# We use the following heuristic: the reference element is the one on which the middle of the pipe falls, and we take
# side='right' which means we default to thick elements as references (ensure main dipoles are all correct)
s_mid_positions = [pipes_loc_middles[transform_name].z for transform_name, _ in pipes_loc]
sv_indices = np.searchsorted(sv_b1_straight.Z, s_mid_positions, side='right') - 1
sv_indices = np.clip(sv_indices, 0, len(sv_b1_straight.Z) - 1)
for ii, sv_index in enumerate(sv_indices):
    while sv_index > 0 and sv_b1_straight.length[sv_index] == 0:
        sv_index -= 1
    while sv_index > 0 and sv_b1_straight.name[sv_index].startswith('||'):
        sv_index -= 1
    sv_indices[ii] = sv_index
sv_names = [sv_b1_straight.name[idx] for idx in sv_indices]
assert len(set(sv_b1_straight.name)) == len(sv_b1_straight.name), "There are non-unique survey names, which might be a problem"
transform_to_sv_point = dict(zip([transform_name for transform_name, _ in pipes_loc], sv_names))

# Place pipes in the model
for transform_name, mad_point in pipes_loc:
    if transform_name in excluded_pipe_positions:
        continue

    type_name = ldb_model.transformations[transform_name].target_type
    s_pos = mad_point.z
    pipe_to_place = None

    # Determine if the pipe is for B1, B2, or both, by doing a rough check: does the first profile of the pipe as
    # installed fit the beam? In addition, this check is rough because the interpolation ignores curved elements.

    # We check this only for pipes that are deemed to be not beam specific (None in the element_type_aperture field).
    if not type_name.endswith('/I') and not type_name.endswith('/E'):
        # Get profile info and the local offsets
        ldb_aper = ldb_model.types[type_name].aperture
        profile = ldb_aper.aperture_alias[0]
        offset = np.array([ldb_aper.offset_x[0], ldb_aper.offset_y[0], ldb_aper.offset_z[0]])

        # Get a rough polygon in the survey space
        poly_2d = builder._profiles[profile].build_polygon(8)
        poly_hom = poly2d_to_homogeneous(poly_2d)  # -> shape is (4, len_points)
        poly_hom[:3, :] += offset[:, np.newaxis]
        poly_hom = mad_point.matrix @ poly_hom

        s_profile = np.mean(poly_hom[2, :])

        # Get the rough survey position at s
        survey_x_at_s = np.interp(s_profile, sv_b1_straight.Z, sv_b1_straight.X)
        survey_y_at_s = np.interp(s_profile, sv_b1_straight.Z, sv_b1_straight.Y)

        # Check that the survey is inside the polygon
        poly = Polygon(vertices=poly_hom[(0, 1), :].T)

        if poly.is_point_inside_polygon(np.array([survey_x_at_s, survey_y_at_s])):
            pipe_to_place = type_name
    else:
        # For a beam specific pipe, place the correct variant (I or E) based on the current side of the beam
        side = 'E' if np.interp(s_pos, sv_b1_straight.Z, sv_b1_straight.X) > 0 else 'I'
        pipe_to_place = f'{type_name}/{side}'

    # Find the right reference point in the survey (closest) for this pipe
    survey_ref = transform_to_sv_point[transform_name]
    from_ip1_to_ref = survey_relative_transform(sv_b1_straight, 'ip1', survey_ref)
    from_ip1_to_here = mad_point.matrix
    from_ref_to_here = np.linalg.inv(from_ip1_to_ref) @ from_ip1_to_here

    if pipe_to_place:
        builder.place_pipe(transform_name, pipe_to_place, transformation=from_ref_to_here, at=survey_ref)

# Clip the pipe that crosses the ring boundary. This avoids placing the single-turn model
# past _end_point until the aperture model supports wrapped pipe spans.
last_profile_hcvc1ib = builder._pipes['HCVC1IB'].positions[1]
old_hcvc1ib_last_shift_s = last_profile_hcvc1ib.shift_s
vc1ib_1l1_start = dict(pipes_loc)['VC1IB.1L1.X'].z
boundary_margin = 0.001
last_profile_hcvc1ib.shift_s = min(old_hcvc1ib_last_shift_s, b1.get_length() - vc1ib_1l1_start - boundary_margin)

aperture_model = builder.build(context=context)
aperture_straight = Aperture(model=aperture_model, line=b1, s_tol=PIPE_OVERLAP_TOL, context=context)

p_tab = aperture_straight.get_pipe_table()
pipe_overlap_rows = _write_pipe_overlap_report(
    p_tab,
    b1.get_length(),
    'pipe_overlaps_summary.csv',
    s_tol=PIPE_OVERLAP_TOL,
)
print(f'Wrote {len(pipe_overlap_rows)} pipe overlap rows to pipe_overlaps_summary.csv')

ax = plt.gca()
aperture_straight.plot_floor_projection(ax=ax)
ax.set_aspect('auto')
plt.title('LHC LS3 Aperture and Survey Floor Plot (Straightened)')

XLIM = (0, 27000)
YLIM = (-0.5, 0.5)

ax.set_xlim(*XLIM)
ax.set_ylim(*YLIM)
ax.set_xbound(lower=XLIM[0], upper=XLIM[1])
ax.set_ybound(lower=YLIM[0], upper=YLIM[1])
plt.show()


# Find all the magnets that are controlled by the "a.mb" knob
main_dipoles = [
    ref._owner._key for ref in b1.vars['a.mb'].xdeps.controlled_targets
    if isinstance(ref, xd.refs.AttrRef) and ref._key == 'angle'
]
elements_b1 = set(b1.element_names)
main_dipoles_b1 = [name for name in main_dipoles if name in set(elements_b1)]

# Plot the element boxes vs the relevant pipe boxes to see how we match
sv_dipoles_b1 = sv_b1_straight.rows[main_dipoles_b1]
magnet_boxes = zip(sv_dipoles_b1.Z, sv_dipoles_b1.length, sv_dipoles_b1.name)
BOX_HEIGHT = 1

fig, ax = plt.subplots()

def _draw_boxes(boxes, height=1., label_rotation=0, label_y=0., label_size=6,
                label_va='center', **kwargs):
    for x, width, label in boxes:
        rect = Rectangle(
            (x, -height * BOX_HEIGHT),
            width,
            2 * height * BOX_HEIGHT,
            **kwargs,
        )
        ax.add_patch(rect)
        ax.text(
            x + width / 2,
            label_y * height * BOX_HEIGHT,
            label,
            ha='center',
            va=label_va,
            rotation=label_rotation,
            fontsize=label_size,
            clip_on=True,
        )

_draw_boxes(
    magnet_boxes,
    edgecolor='black',
    facecolor='skyblue',
    alpha=0.7,
    label_rotation=90,
    label_y=1.15,
    label_size=9,
    label_va='bottom',
)

# Also plot the relevant type boxes
pipe_labels = [f'{name}\nref: {ref}' for name, ref in zip(p_tab.name, p_tab.survey_reference)]
pipe_boxes = zip(p_tab.s_span_start, p_tab.span, pipe_labels)
_draw_boxes(
    pipe_boxes,
    edgecolor='black',
    facecolor='pink',
    alpha=0.7,
    height=0.5,
    label_rotation=90,
    label_size=8,
    label_y=-1.15,
    label_va='top',
)

# Set plot limits
ax.set_xlim(0, b1.get_length())
ax.set_ylim(-5 * BOX_HEIGHT - 1, 5 * BOX_HEIGHT + 1)

plt.show()


# Sanity checks for main dipoles
table_main_dipoles_pipe_model = p_tab.rows[np.isin(p_tab.survey_reference, main_dipoles)]
assert set(main_dipoles_b1) == set(table_main_dipoles_pipe_model.survey_reference)  # All dipoles have pipes assigned?

main_dipole_pipes = set(table_main_dipoles_pipe_model.pipe_name)
indices_with_dipole_pipes = np.isin(p_tab.pipe_name, list(main_dipole_pipes))
indices_not_main_dipoles = ~np.isin(p_tab.survey_reference, main_dipoles)
assert len(p_tab.rows[indices_with_dipole_pipes & indices_not_main_dipoles]) == 0  # Main dipole pipes not elsewhere?


# Unbend the model!
b1.regenerate_from_composer()
b2.regenerate_from_composer()

lattice.vars['a.mb'] = angle_before
lattice.vars['ds'] = ds_before

b1.end_compose()
b2.end_compose()

sv_b1 = b1.survey()
sv_b2 = b2.survey(theta0=np.pi)

for pipe_name in main_dipole_pipes:
    a_dipole_name = p_tab.rows[p_tab.pipe_name == pipe_name].survey_reference[0]
    pipe = aperture_straight.pipes[pipe_name]
    pipe.curvature = b1[a_dipole_name].angle / pipe.length

# UNDO THE EARLIER PATCH
# aperture_model.pipes[aperture_model.pipe_names.index('HCVC1IB')].shift_s = old_hcvc1ib_last_shift_s

# Build the curved model
aperture = Aperture(model=aperture_model, line=b1, s_tol=PIPE_OVERLAP_TOL, context=context)
aperture.to_json('b1_aperture.json')
