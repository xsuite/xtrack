import cernlayoutdb as layout
from functools import reduce
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import operator
import xdeps as xd
import xtrack as xt
from xtrack.aperture import Aperture, ApertureBuilder
from xtrack.aperture.structures import Polygon
from xtrack.aperture.transform import poly2d_to_homogeneous
from xtrack.survey import survey_relative_transform
from warnings import warn


class LDBConverterWarning(UserWarning):
    pass


def _format_list(lst):
    if len(lst) < 3:
        return ", ".join(lst)
    return ", ".join(lst[:3]) + ", ..."


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
    'AP205': ('Rectangle', 0.04293, 0.03843),  # Originally an octagon with no diagonal value...
    'AP207': ('Rectangle', 0.04978, 0.04978),  # ditto
    'AP163': ('RectEllipse', 0.068, 0.0328, 0.0101, 0.0101)  # Originally half_minor was zero...
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

b1 = lattice.b1
b2 = lattice.b2

b1.regenerate_from_composer()
b2.regenerate_from_composer()

angle_before = lattice.vars['a.mb']
ds_before = lattice.vars['ds']

lattice.vars['a.mb'] = 0
lattice.vars['ds'] = 0

b1.end_compose()
b2.end_compose()

sv_b1 = b1.survey()
sv_b2 = b2.survey()


def beam_inner_outer(s, sv):
    x = np.interp(s, sv.s, sv.x)

    if x > 1e-10:
        return 'E'

    if x < -1e-10:
        return 'I'

    return None


plt.plot(sv_b1.Z, sv_b1.X, c='red', label='B1', marker='o')
plt.plot(sv_b2.Z, sv_b2.X, c='blue', label='B2', marker='o')

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
for type_name, type in ldb_model.types.items():
    if (aperture := type.aperture) is None:
        ignored_types.append(type_name)
        continue

    inner_outer = reduce(operator.or_, (set(ie) if ie else {None} for ie in aperture.element_type_aperture), set())
    if 'E' not in inner_outer and None not in inner_outer:
        ignored_types.append(type_name)
        continue

    element_type_aperture = aperture.element_type_aperture

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

        element_type_aperture = [override] * len(aperture.element_type_aperture)
        beam_specific_type = override is not None

    profiles = aperture.aperture_alias
    offsets = (aperture.offset_x, aperture.offset_y, aperture.offset_z)

    if not beam_specific_type:
        pipes = [(None, builder.new_pipe(type_name, curvature=0))]
    else:
        pipes = [
            ('I', builder.new_pipe(type_name + '/I', curvature=0)),
            ('E', builder.new_pipe(type_name + '/E', curvature=0)),
        ]

    for profile, side, off_x, off_y, off_z in zip(profiles, element_type_aperture, *offsets):
        mad_off = layout.LDBPoint(x=off_x, y=off_y, z=off_z).to_madpoint()

        for side_marker, pipe in pipes:
            if side_marker is None or side is None or side_marker in side:
                pipe.place_profile(
                    profile,
                    shift_s=mad_off.z,
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

    loc = ldb_model.get_abs_points(transform_name)['MECHANICAL START']
    loc_middle = ldb_model.get_abs_points(transform_name)['MECHANICAL MIDDLE']
    pipes_loc[transform_name] = loc.to_madpoint()
    pipes_loc_middles[transform_name] = loc_middle.to_madpoint()

pipes_loc = sorted(pipes_loc.items(), key=lambda x: x[1].z)


if ignored_transforms:
    warn(f'Ignored {len(ignored_transforms)} transforms without target types, or whose target types are not valid '
         f'apertures: {_format_list(ignored_transforms)}', LDBConverterWarning)


# Refer pipe locations to survey elements
s_mid_positions = [mad_point.z for _, mad_point in pipes_loc]
sv_indices = np.searchsorted(sv_b1.Z + np.concat([np.diff(sv_b1.Z) / 2, [0]]), s_mid_positions, side='right')
sv_names = [sv_b1.name[idx] for idx in sv_indices]
assert len(set(sv_b1.name)) == len(sv_b1.name), "There are non-unique survey names, which might be a problem"
transform_to_sv_point = dict(zip([transform_name for transform_name, _ in pipes_loc], sv_names))

# Place pipes in the model
for transform_name, mad_point in pipes_loc:
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
        survey_x_at_s = np.interp(s_profile, sv_b1.Z, sv_b1.X)
        survey_y_at_s = np.interp(s_profile, sv_b1.Y, sv_b1.X)

        # Check that the survey is inside the polygon
        poly = Polygon(vertices=poly_hom[(0, 1), :].T)

        if poly.is_point_inside_polygon(np.array([survey_x_at_s, survey_y_at_s])):
            pipe_to_place = type_name
    else:
        # For a beam specific pipe, place the correct variant (I or E) based on the current side of the beam
        side = 'E' if np.interp(s_pos, sv_b1.Z, sv_b1.X) > 0 else 'I'
        pipe_to_place = f'{type_name}/{side}'

    # Find the right reference point in the survey (closest) for this pipe
    survey_ref = transform_to_sv_point[transform_name]
    from_ip1_to_ref = survey_relative_transform(sv_b1, 'ip1', survey_ref)
    from_ip1_to_here = mad_point.matrix
    from_ref_to_here = np.linalg.inv(from_ip1_to_ref) @ from_ip1_to_here

    if pipe_to_place:
        builder.place_pipe(transform_name, pipe_to_place, transformation=from_ref_to_here, survey_reference=survey_ref)

# TEMPORARILY PATCH THE LAST PIPE SO IT DOESN'T EXTEND PAST THE END OF THE MACHINE
# CAUSING CHECKS TO FAIL - THIS SHOULD BE UNDONE IN THE RING VERSION!
builder._pipes['HCVC1IB'].positions[1].shift_s = 3.6
# END OF PATCH

aperture_model = builder.build()
aperture = Aperture(model=aperture_model, line=b1, _skip_validity_check=False)

ax = plt.gca()
aperture.plot_floor_projection(ax=ax, len_points=32)
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
sv_dipoles_b1 = sv_b1.rows[main_dipoles_b1]
magnet_boxes = zip(sv_dipoles_b1.Z, sv_dipoles_b1.length, sv_dipoles_b1.name)
BOX_HEIGHT = 1

fig, ax = plt.subplots()

def _draw_boxes(boxes, height=1., label_rotation=0, label_y=0., label_size=6, **kwargs):
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
            va='center',
            rotation=label_rotation,
            fontsize=label_size,
            clip_on=True,
        )

_draw_boxes(magnet_boxes, edgecolor='black', facecolor='skyblue', alpha=0.7, label_rotation=90, label_y=0.35)

# Also plot the relevant type boxes
p_tab = aperture.get_pipe_table()
pipe_labels = [f'{name}\nref: {ref}' for name, ref in zip(p_tab.name, p_tab.survey_reference)]
pipe_boxes = zip(p_tab.s_start, p_tab.span, pipe_labels)
_draw_boxes(
    pipe_boxes,
    edgecolor='black',
    facecolor='pink',
    alpha=0.7,
    height=0.5,
    label_rotation=90,
    label_size=5,
    label_y=-0.35,
)

# Set plot limits
ax.set_xlim(0, b1.get_length())
ax.set_ylim(-5 * BOX_HEIGHT - 1, 5 * BOX_HEIGHT + 1)

plt.show()
