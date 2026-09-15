from types import SimpleNamespace

import numpy as np
import pandas as pd

import xtrack as xt
from xtrack._temp import survey_utils as su

from bumps_report import read_bumps_report

# Convert the SU bump requests (RST displacements of the element end points)
# into MAD-X misalignments (dtheta, dphi, dpsi, dx, dy, ds).
#
# `bumps_report` reads the report, checks it over and gives one row per
# element: the requested displacement of the entrance and the exit points in
# the element's own RST frame, plus a roll about the chord. Together with the
# nominal chord length, that is exactly what
# `su.misalignment_from_rst_displacements` takes.
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

ENTRY_COLUMNS = ['r_entry', 's_entry', 't_entry']
EXIT_COLUMNS = ['r_exit', 's_exit', 't_exit']

# #############################################################################
# Part 1 - computation
# #############################################################################

requests = read_bumps_report('Bumps_sp_report.csv')

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
    if name not in requests.index:
        continue

    request = requests.loc[name]
    roll = request['roll']
    displ_start = request[ENTRY_COLUMNS].to_numpy(dtype=float)
    displ_end = request[EXIT_COLUMNS].to_numpy(dtype=float)

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

    # The thin instruments (XSCI, XDWC) are reported on their entrance point
    # only; the reader gives their exit the same displacement, which comes out
    # here as a rigid translation with no rotation about x or y. See
    # `read_bumps_report` for why that is the reading. That filling-in is not
    # a request, so there is nothing it could have dropped, and the requested
    # exit displacement is left empty for those elements.
    single_point = bool(request['single_point'])
    ds_exit_requested = np.nan if single_point else request['s_exit']

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
        'ds_exit_requested': ds_exit_requested,
        'ds_exit_rigid': ds_exit_rigid,
        'ds_exit_dropped': ds_exit_requested - ds_exit_rigid,
    })

out = pd.DataFrame(results).set_index('name')

# #############################################################################
# Part 2 - checks
# #############################################################################

print('checks:')

# The report itself is checked by `read_bumps_report`: the column totals
# against their four sources, the names against `Nom Layout`, the roll against
# its duplicate on the other point.
print(f'  {len(requests)} elements read from the report')

# Walking the lattice skips whatever carries no request, so a report element
# missing from the lattice would go unnoticed.
assert len(out) == len(requests)
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
      f'(max {out["ds_exit_dropped"].abs().max() * 1e3:.3f} mm); '
      f'{out["ds_exit_requested"].isna().sum()} single-point elements '
      f'requested nothing at the exit')
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
