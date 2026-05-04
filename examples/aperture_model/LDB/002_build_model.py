import cernlayoutdb as layout
import xtrack as xt
from xtrack.aperture import ApertureBuilder

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

lattice = xt.load('https://acc-models.web.cern.ch/acc-models/lhc/hl19/xsuite/lhc.json')
b1 = lattice.b1

ldb_model = layout.Machine.from_pickle("LHC.pickle")

builder = ApertureBuilder(line=b1)

for profile_name, profile in ldb_model.profiles.items():
    shape, param_names = LDB_SHAPE_TO_XS.get(profile.shape, (None, None))

    if not shape:
        print(f'No shape found for profile {profile}. Ignoring.')
        continue

    ldb_params = (getattr(profile, param_name) for param_name in LDB_PARAMS[:len(param_names)])
    shape_params = dict(zip(param_names, ldb_params))

    builder.new_profile(profile_name, shape, **shape_params)


for pipe_name, pipe in ldb_model.pipes.items():
    if (aperture := pipe.aperture) is None:
        print(f'Pipe `{pipe_name}` has no apertures. Ignoring.')
        continue

    profiles = aperture.aperture_alias
    offsets = (aperture.offset_x, aperture.offset_y, aperture.offset_z)

    pipe_blueprint = builder.new_pipe(pipe_name, curvature=0)

    for profile, off_x, off_y, off_z in zip(profiles, *offsets):
        mad_off = layout.LDBPoint(off_x, off_y, off_z).to_madpoint()

        pipe_blueprint.place_profile(profile, shift_s=mad_off.z, shift_x=mad_off.x, shift_y=mad_off.y)
