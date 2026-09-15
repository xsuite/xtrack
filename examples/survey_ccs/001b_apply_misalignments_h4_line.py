import numpy as np
import pandas as pd

import xtrack as xt
from xtrack._temp import survey_utils as su

# Apply the misalignments computed by 001a_bump_to_misalignments_h4_line.py to
# the H4 line, then export the misaligned line and its survey.
#
# Many of the requested elements (instruments, collimators) are modelled as
# drifts, and a Drift carries no alignment attributes. They are replaced by
# Device, which tracks as a drift but can hold a misalignment, so that every
# requested element ends up with the position the report asks for.
#
# The script is in three parts: the computation, then the exports, then the
# checks that verify the whole thing. The checks come last here, unlike in
# 001a, because some of them read the exported files back.

MISALIGNMENT_COLUMNS = ['dtheta', 'dphi', 'dpsi', 'dx', 'dy', 'ds']

# #############################################################################
# Part 1 - computation
# #############################################################################

table = pd.read_csv('bump_misalignments.csv', index_col=0)
element_names = [nn.lower() for nn in table.index]

env = xt.load('survey-h4-post-ls3-cern-coords-v4.seq')
line = env['h4']

# Swap the drift-modelled elements for devices of the same length, keeping
# their names so that they stay matched to the report.
converted = []
for name in element_names:
    element = line[name]
    if not isinstance(element, xt.Drift):
        continue
    device = xt.Device(
        length=element.length,
        model=element.model,
        rot_s_rad=getattr(element, 'rot_s_rad', 0.),
    )
    # `extra` carries the layout database slot_id, which the legacy survey
    # export writes out, so it has to survive the conversion.
    if hasattr(element, 'extra'):
        device.extra = element.extra
    line.element_dict[name] = device
    converted.append(name)

# Starting point
f0_ccs = xt.CCSFrame(
    x=669.234140,
    y=4590.727900,
    z=2441.574200,
    theta_gon=7.4198200,
    phi=-0.000370000,
    psi=0,
)
f0 = xt.Frame.from_ccs(f0_ccs)


# Reference (nominal) survey, taken before anything is misaligned.
survey_nominal = line.survey(include_element_frames=True,
                             X0=f0.X,
                             Y0=f0.Y,
                             Z0=f0.Z,
                             theta0=f0.theta,
                             phi0=f0.phi,
                             psi0=f0.psi)

for name, row in zip(element_names, table.to_dict('records')):
    su.Misalignment(
        dtheta=row['dtheta'],
        dphi=row['dphi'],
        dpsi=row['dpsi'],
        shift_x=row['dx'],
        shift_y=row['dy'],
        shift_s=row['ds'],
    ).apply_to_element(line[name])

survey_aligned = line.survey(include_element_frames=True,
                             X0=f0.X,
                             Y0=f0.Y,
                             Z0=f0.Z,
                             theta0=f0.theta,
                             phi0=f0.phi,
                             psi0=f0.psi)

# #############################################################################
# Part 2 - export
# #############################################################################

line.to_json('h4_misaligned.json')
print('written to h4_misaligned.json')

su.write_legacy_survey_tfs(
    'h4_misaligned_survey.tfs',
    survey=survey_aligned,
    element_names=element_names,
    element_container=env,
)
print('written to h4_misaligned_survey.tfs '
      f'({2 * len(element_names)} points, one per element end)')

# #############################################################################
# Part 3 - checks
# #############################################################################

print('checks:')

print(f'  {len(element_names)} requested elements, '
      f'{len(converted)} drifts converted to devices')

# Every requested element must now be able to hold a misalignment.
for name in element_names:
    assert hasattr(line[name], 'rot_s_rad'), name
print('  all requested elements can hold a misalignment')

# The slot_id must survive the conversion: the legacy survey export needs it.
for name in element_names:
    assert 'slot_id' in line[name].extra, name
print(f'  slot_id preserved on all {len(element_names)} elements')

# The survey reference path is not perturbed by misalignments.
for ii in range(len(survey_aligned)):
    np.testing.assert_allclose(
        survey_aligned.XYZ_ref_start[ii], survey_nominal.XYZ_ref_start[ii],
        atol=1e-12, rtol=0)
print(f'  reference path unchanged over all {len(survey_aligned)} survey rows')

# Nothing outside the requested elements moved.
requested = set(element_names)
untouched = [nn for nn in survey_nominal.name if nn not in requested]
for name in untouched:
    np.testing.assert_allclose(
        survey_aligned['XYZ_elem_start', name],
        survey_nominal['XYZ_elem_start', name], atol=1e-12, rtol=0)
print(f'  {len(untouched)} other survey rows stayed where they were')

# Read the misalignment back out of the surveyed geometry. This goes through
# the line and its survey rather than through the formulas 001a already used,
# so it checks that the applied parameters really produce the requested shape.
recovered = {}
for name in element_names:
    element = line[name]
    misalignment = su.misalignment_from_absolute_position(
        XYZ_elem_start=survey_aligned['XYZ_elem_start', name],
        E_elem_start=survey_aligned['E_elem_start', name],
        XYZ_ref_start=survey_nominal['XYZ_ref_start', name],
        E_ref_start=survey_nominal['E_ref_start', name],
        rbend_angle=(element.angle if isinstance(element, xt.RBend) else None),
    )
    recovered[name.upper()] = [
        misalignment.dtheta, misalignment.dphi, misalignment.dpsi,
        misalignment.shift_x, misalignment.shift_y, misalignment.shift_s,
    ]

recovered = pd.DataFrame.from_dict(
    recovered, orient='index', columns=MISALIGNMENT_COLUMNS)
error = (recovered - table[MISALIGNMENT_COLUMNS]).abs().to_numpy().max()
assert error < 1e-10
print(f'  misalignment recovered from the survey of all {len(table)} '
      f'elements: max error = {error:.2e}')

# The misaligned line has to come back from the json as it went in: the
# devices must still be devices and still carry their misalignment.
reloaded = xt.Line.from_json('h4_misaligned.json')
for name in element_names:
    before, after = line[name], reloaded[name]
    assert type(after._xobject) is type(before._xobject), name
    np.testing.assert_allclose(
        [after.shift_x, after.shift_y, after.shift_s,
         after.rot_x_rad, after.rot_y_rad, after.rot_s_rad_no_frame],
        [before.shift_x, before.shift_y, before.shift_s,
         before.rot_x_rad, before.rot_y_rad, before.rot_s_rad_no_frame],
        atol=0, rtol=0)
print('  the json reloads with its devices and their misalignments')
