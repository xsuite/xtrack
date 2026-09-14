from types import SimpleNamespace

import numpy as np
import pandas as pd

import xtrack as xt
from xtrack._temp import survey_utils as su

# Convert the SU bump requests (RST displacements of the element end points)
# into MAD-X misalignments (dtheta, dphi, dpsi, dx, dy, ds).
#
# The report gives, for each element, the requested displacement of its
# entrance (`.E`) and exit (`.S`) points in the element's own RST frame, plus a
# roll about the chord. Together with the nominal chord length, that is exactly
# what `su.misalignment_from_rst_displacements` takes.
#
# The two displacements plus the roll are seven numbers, while a rigid body has
# only six degrees of freedom, so the request is over-determined: in general no
# rigid motion puts both end points exactly where asked. The element is kept
# rigid, which means its chord length is preserved, so the requested exit
# longitudinal (S) displacement is discarded -- for a rigid element it follows
# from the other two rather than being free. The entrance counterpart is kept:
# that one is the `ds` shift of the whole element. This script reports, element
# by element, how much of the request that drops.
#
# The script is in three parts: the computation, then the checks that verify
# it, then the report. The computation hands over `out`, a table with one row
# per element, and `rst_offsets`, the RST end-point positions of each displaced
# element.

# RST component order matches survey_utils: E_rst = column_stack((er, es, et)).
RST_COLUMNS = ['Radial (m)', 'Longitudinal (m)', 'Vertical (m)']
ROLL_COLUMN = 'Roll (rad)'


def report_value(row, column):
    """Read one cell; a blank means no bump requested, i.e. zero."""
    value = row[column]
    return 0. if pd.isna(value) else float(value)


# #############################################################################
# Part 1 - computation
# #############################################################################

# ------------------------------------------------------- read the bump report

df = pd.read_csv('Bumps_sp_report.csv')

# The accented character in 'Elément' was lost in the csv export
df.rename(columns={'El�ment': 'Element'}, inplace=True)

# Each row of the report is a *point*, not an element: the name ends in `.E`
# (entrée, the element start) or `.S` (sortie, the element end). Collect the
# requested displacement and roll of each point under its own name; the two
# points of an element are brought together in the loop below.
requests = {}
for _, row in df.iterrows():
    requests[row['Element']] = (
        np.array([report_value(row, cc) for cc in RST_COLUMNS]),
        report_value(row, ROLL_COLUMN),
    )

# --------------------------------------------------------------- load lattice

env = xt.load('survey-h4-post-ls3-cern-coords-v4.seq')
line = env['h4']

# ------------------------------------------- convert, one element at a time

# Walk the lattice and pick out the elements that carry a bump request, so
# that the results come out in machine order.
results = []
rst_offsets = {}
for element_name in line.get_table().name:
    name = element_name.upper()
    if f'{name}.E' not in requests:
        continue

    displ_start, roll = requests[f'{name}.E']

    # The thin instruments (XSCI, XDWC) are reported on their entrance point
    # only. One point cannot define a rotation, so the request is a rigid
    # translation: the exit moves with the entrance.
    single_point = f'{name}.S' not in requests
    displ_end = displ_start if single_point else requests[f'{name}.S'][0]

    element = line[element_name]

    # Chord length: the RBends keep the arc in `length` and the chord in
    # `length_straight`. The elements modelled as drifts (instruments,
    # collimators) carry neither a tilt nor a bending angle.
    length = getattr(element, 'length_straight', None) or element.length
    tilt = getattr(element, 'rot_s_rad', 0.) or 0.
    angle = getattr(element, 'angle', 0.) or 0.

    mis = su.misalignment_from_rst_displacements(
        displ_start, displ_end, length, bgamma=roll, tilt=tilt, angle=angle)

    # Chord that the rigid motion gives the element, to compare the
    # longitudinal exit displacement it produces with the requested one that
    # had to be dropped. The nominal element runs from (0, 0, 0) to
    # (0, length, 0) in its own RST frame.
    chord_rst = su.rst_rigid_chord(displ_start, displ_end, length)
    rst_offsets[name] = (displ_start, displ_start + chord_rst)

    ds_exit_rigid = displ_start[1] + chord_rst[1] - length

    results.append({
        'name': name,
        'element_type': type(element._xobject).__name__.replace('Data', ''),
        'length_chord': length,
        'tilt': tilt,
        'angle': angle,
        'dtheta': mis.dtheta,
        'dphi': mis.dphi,
        # `dpsi` from survey_utils includes the design tilt; the bump-induced
        # part alone is what MAD-X receives on top of the nominal tilt.
        'dpsi': mis.dpsi,
        'dpsi_no_tilt': mis.dpsi - tilt,
        'dx': mis.shift_x,
        'dy': mis.shift_y,
        'ds': mis.shift_s,
        'single_point': single_point,
        'ds_exit_requested': displ_end[1],
        'ds_exit_rigid': ds_exit_rigid,
        'ds_exit_dropped': displ_end[1] - ds_exit_rigid,
    })

out = pd.DataFrame(results).set_index('name')

# #############################################################################
# Part 2 - checks
# #############################################################################

print('checks:')

# The element name and point suffix parsed out of the `Elément` column must
# agree with the `Nom Layout` column, and every point must be an end point.
for _, row in df.iterrows():
    name, point = row['Element'].rsplit('.', 1)
    assert name == row['Nom Layout']
    assert point in ('E', 'S')
print(f'  {len(df)} report points, names consistent with Nom Layout')

# The roll is stored redundantly on both points of an element.
for point_name, (_, roll) in requests.items():
    name, point = point_name.rsplit('.', 1)
    if point == 'S':
        assert roll == requests[f'{name}.E'][1]
print('  roll consistent on both points of every element')

# Walking the lattice skips whatever carries no request, so a report element
# missing from the lattice would go unnoticed.
assert len(out) == len({nn.rsplit('.', 1)[0] for nn in requests})
print(f'  all {len(out)} report elements found in the lattice')

# The element must come out rigid: the chord of the displaced element has to
# keep its nominal length.
length_error = np.array([
    np.linalg.norm(offset_end - offset_start) - out.loc[name, 'length_chord']
    for name, (offset_start, offset_end) in rst_offsets.items()])
assert np.abs(length_error).max() < 1e-12
print('  chord length preserved: max error = '
      f'{np.abs(length_error).max():.2e} m')

# Push each misalignment back through the forward transformation, at the
# nominal chord length, and check the RST end points are recovered. A stand-in
# object is used so that the check also covers the elements modelled as drifts,
# which have no misalignment attributes, and so that the line is untouched.
round_trip_error = {}
for name, (offset_start, offset_end) in rst_offsets.items():
    row = out.loc[name]
    probe = SimpleNamespace(
        angle=row['angle'],
        rot_s_rad=row['tilt'],
        rot_y_rad=row['dtheta'],
        rot_x_rad=row['dphi'],
        rot_s_rad_no_frame=row['dpsi_no_tilt'],
        shift_x=row['dx'],
        shift_y=row['dy'],
        shift_s=row['ds'],
    )
    back_start, back_end = su.rst_start_end_offsets_from_parameters(
        probe, row['length_chord'])
    round_trip_error[name] = max(np.max(np.abs(back_start - offset_start)),
                                 np.max(np.abs(back_end - offset_end)))

round_trip_error = pd.Series(round_trip_error)
assert round_trip_error.max() < 1e-12
print(f'  round trip over {len(out)} elements: max error = '
      f'{round_trip_error.max():.2e} m')

# Closed-form checks of the RST -> MAD-X mapping, on a straight untilted
# element of unit chord: R = -x, S = +s, T = +y, and the SU roll is minus the
# MAD-X dpsi. Everything is expressed in the element's own tilted chord frame,
# so a non-zero design tilt rotates R and T into dx and dy (as seen e.g. on the
# vertical benders MBNV, which have tilt = pi/2).
def check(displ_e, displ_s, bgamma=0.):
    return su.misalignment_from_rst_displacements(
        displ_e, displ_s, length=1., bgamma=bgamma)


dd = 1e-3
mm = check([dd, 0, 0], [dd, 0, 0])                  # radial translation
assert np.allclose([mm.shift_x, mm.shift_y, mm.shift_s], [-dd, 0, 0])
mm = check([0, dd, 0], [0, dd, 0])                  # longitudinal translation
assert np.allclose([mm.shift_x, mm.shift_y, mm.shift_s], [0, 0, dd])
mm = check([0, 0, dd], [0, 0, dd])                  # vertical translation
assert np.allclose([mm.shift_x, mm.shift_y, mm.shift_s], [0, dd, 0])
mm = check([0, 0, 0], [0, 0, 0], bgamma=dd)         # roll
assert np.isclose(mm.dpsi, -dd)
mm = check([dd, 0, 0], [-dd, 0, 0])                 # radial crab
assert np.isclose(mm.dtheta, np.arctan(2 * dd / np.sqrt(1 - (2 * dd)**2)))
mm = check([0, 0, dd], [0, 0, -dd])                 # vertical crab
assert np.isclose(mm.dphi, -np.arctan(2 * dd / np.sqrt(1 - (2 * dd)**2)))
print('  RST -> MAD-X conventions: 6 closed-form cases')

# #############################################################################
# Part 3 - report
# #############################################################################

misalignment_columns = ['dtheta', 'dphi', 'dpsi_no_tilt', 'dx', 'dy', 'ds']
moved = ~np.isclose(out[misalignment_columns], 0.).all(axis=1)
print(f'\n{moved.sum()} of {len(out)} elements are actually misaligned\n')

with pd.option_context('display.width', 200, 'display.max_rows', None):
    print(out.loc[moved, ['element_type', 'length_chord'] +
                  misalignment_columns].to_string(
        float_format=lambda vv: f'{vv: .6e}' if abs(vv) > 0 else f'{0.: .6e}'))

print('\n=== requested exit S displacement dropped to keep the element rigid '
      '===')
dropped = out['ds_exit_dropped'].abs() > 1e-6
print(f'{dropped.sum()} elements differ by more than 1 um '
      f'(max {out["ds_exit_dropped"].abs().max() * 1e3:.3f} mm)')
if dropped.any():
    print(out.loc[dropped, ['length_chord', 'ds_exit_requested',
                            'ds_exit_rigid', 'ds_exit_dropped']].sort_values(
        'ds_exit_dropped', key=abs, ascending=False).to_string())

print('\n=== magnitudes ===')
for column in misalignment_columns:
    unit = 'm' if column in ('dx', 'dy', 'ds') else 'rad'
    print(f'{column:14s} max|.| = {out[column].abs().max():.6e} {unit}   '
          f'n non-zero = {(~np.isclose(out[column], 0.)).sum()}')

out.to_csv('bump_misalignments.csv')
print('\nwritten to bump_misalignments.csv')
