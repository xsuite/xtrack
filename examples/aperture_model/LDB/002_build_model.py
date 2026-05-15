import cernlayoutdb as layout
import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
from xtrack.aperture import Aperture, ApertureBuilder
from xtrack.aperture.structures import Polygon
from xtrack.aperture.transform import poly2d_to_homogeneous
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

lattice.vars['a.mb'] = 0
lattice.vars['ds'] = 0

b1.end_compose()
b2.end_compose()

sv_b1 = b1.survey()
sv_b2 = b2.survey()

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

ignored_profiles = []
for profile_name, profile in ldb_model.profiles.items():
    shape, param_names = LDB_SHAPE_TO_XS.get(profile.shape, (None, None))

    if not shape:
        ignored_profiles.append(profile_name)
        continue

    ldb_params = (getattr(profile, param_name) for param_name in LDB_PARAMS[:len(param_names)])
    shape_params = dict(zip(param_names, ldb_params))

    builder.new_profile(profile_name, shape, **shape_params)

if ignored_profiles:
    warn(
        f'Ignored {len(ignored_profiles)} profiles with no shapes: {_format_list(ignored_profiles)}',
        LDBConverterWarning,
    )

ignored_types = []
for type_name, type in ldb_model.types.items():
    if (aperture := type.aperture) is None:
        ignored_types.append(type_name)
        continue

    profiles = aperture.aperture_alias
    offsets = (aperture.offset_x, aperture.offset_y, aperture.offset_z)

    pipe_blueprint = builder.new_pipe(type_name, curvature=0)

    for profile, off_x, off_y, off_z in zip(profiles, *offsets):
        mad_off = layout.LDBPoint(x=off_x, y=off_y, z=off_z).to_madpoint()
        pipe_blueprint.place_profile(profile, shift_s=mad_off.z, shift_x=mad_off.x, shift_y=mad_off.y)

if ignored_types:
    warn(f'Ignored {len(ignored_types)} types without apertures: {_format_list(ignored_types)}', LDBConverterWarning)


pipes_loc = {}
ignored_transforms = []
for transform_name, transformation in ldb_model.transformations.items():
    if (target_type := transformation.target_type) is None:
        ignored_transforms.append(transform_name)
        continue

    if target_type not in builder._pipes:
        ignored_transforms.append(transform_name)
        continue

    loc = ldb_model.get_abs_points(transform_name)['MECHANICAL START']
    pipes_loc[transform_name] = loc.to_madpoint()


pipes_loc = sorted(pipes_loc.items(), key=lambda x: x[1].z)


if ignored_transforms:
    warn(f'Ignored {len(ignored_transforms)} transforms without target types, or whose target types are not valid '
         f'apertures: {_format_list(ignored_transforms)}', LDBConverterWarning)


for transform_name, mad_point in pipes_loc:
    type_name = ldb_model.transformations[transform_name].target_type
    s_pos = mad_point.z

    # Determine if the pipe is for B1, B2, or both, by doing a rough check: does the first profile of the pipe as
    # installed fit the beam? In addition, this check is rough because the interpolation ignores curved elements.

    # Get profile info and the local offsets
    ldb_aper = ldb_model.types[type_name].aperture
    profile = ldb_aper.aperture_alias[0]
    offset = np.array([ldb_aper.offset_x[0], ldb_aper.offset_y[0], ldb_aper.offset_z[0]])
    s_profile = s_pos + offset[2]

    # Get a rough polygon in the survey space
    poly_2d = builder._profiles[ldb_aper.aperture_alias].build_polygon(8)
    poly_hom = poly2d_to_homogeneous(poly_2d)
    poly_hom[:, :3] += offset
    poly_hom = mad_point.matrix @ poly_hom

    assert np.isclose(s_profile, poly_hom[:, 3], atol=1e-9, rtol=1e-9)

    # Get the rough survey position at s
    survey_x_at_s = np.interp(s_profile, sv_b1.Z, sv_b1.X)
    survey_y_at_s = np.interp(s_profile, sv_b1.Y, sv_b1.X)

    # Check that the survey is inside the polygon
    poly = Polygon(vertices=poly_hom[:, (0, 1)])

    if poly.is_point_inside_polygon(np.array([survey_x_at_s, survey_y_at_s])):
        builder.place_pipe(transform_name, type_name, transformation=mad_point.matrix, survey_reference='ip1')

aperture_model = builder.build()
aperture = Aperture(model=aperture_model, line=b1, _skip_validity_check=True)

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
