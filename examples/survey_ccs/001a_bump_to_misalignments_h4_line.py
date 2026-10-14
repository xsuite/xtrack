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
# the element's own RST frame, plus the GEODE roll. Together with the
# nominal chord length, that is exactly what
# `su.misalignment_from_geode_displacements` takes.
# The reader has already negated the report's radial deviations to obtain
# geometric R displacements; no further sign conversion is needed here.
#
# The two displacements plus the roll are seven numbers, while a rigid body has
# only six degrees of freedom, so the request is over-determined: in general no
# rigid motion puts both end points exactly where asked. The element is kept
# rigid, which means its chord length is preserved, so the requested exit
# longitudinal (S) displacement is discarded -- for a rigid element it follows
# from the other two rather than being free. The entrance counterpart is kept.
# This determines the crab in the tilted chord frame. The additional GEODE
# roll is composed about the entrance reference tangent, so it can further
# move a bend's exit. The report separates the rigidity adjustment from the
# exit displacement introduced by roll.
#
# The script is in three parts: the computation, then the checks that verify
# it, then the report. The computation hands over `out`, a table with one row
# per element; the checks rebuild from it whatever geometry they need.

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
    # `length_straight`. Drift-modelled instruments can carry a design tilt.
    length = getattr(element, 'length_straight', None) or element.length
    tilt = getattr(element, 'rot_s_rad', 0.) or 0.
    angle = getattr(element, 'angle', 0.) or 0.

    mis = su.misalignment_from_geode_displacements(
        displ_start, displ_end, length, bgamma=roll, tilt=tilt, angle=angle)

    # Longitudinal adjustment from rigidity, before the additional roll.
    chord_rst = su.rst_rigid_chord(displ_start, displ_end, length)
    ds_exit_rigid = displ_start[1] + chord_rst[1] - length

    # For entrance-only requests, the reader defaults exit R and T to zero
    # to match GEODE, so the instrument can crab. Exit S follows from rigidity;
    # no exit S was requested, hence nothing can be reported as discarded.
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
        # The additional MAD-X rotation includes crab/bend coupling as well
        # as the requested roll. It is generally not just minus that roll.
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

# Independently construct the expected GEODE chord: roll the nominal chord
# about the entrance tangent, then apply the crab inferred without roll.
# A nominal element runs from (0, 0, 0) to (0, length, 0) in its RST frame.
rst_offsets = {}
rst_basis = np.array([[-1., 0, 0], [0, 0, 1], [0, 1, 0]])
for name, row in out.iterrows():
    request = requests.loc[name]
    displ_start = request[ENTRY_COLUMNS].to_numpy(dtype=float)
    displ_end = request[EXIT_COLUMNS].to_numpy(dtype=float)
    chord_rst = su.rst_rigid_chord(displ_start, displ_end, row['length_chord'])
    nominal = xt.Frame().rotate_s(row['tilt']).rotate_y(-row['angle']/2)
    axis_rst = rst_basis.T @ nominal.E_matrix.T @ np.array([0., 0., 1.])
    nominal_chord = np.array([0., row['length_chord'], 0.])
    gamma = -request['roll']
    rolled_chord = (nominal_chord*np.cos(gamma)
                    + np.cross(axis_rst, nominal_chord)*np.sin(gamma)
                    + axis_rst*np.dot(axis_rst, nominal_chord)*(1-np.cos(gamma)))
    crab_theta = np.arctan2(-chord_rst[0], chord_rst[1])
    crab_phi = np.arcsin(chord_rst[2]/row['length_chord'])
    crab = xt.Frame().rotate_y(crab_theta).rotate_x(-crab_phi)
    final_chord = rst_basis.T @ crab.E_matrix @ rst_basis @ rolled_chord
    rst_offsets[name] = (displ_start, displ_start + final_chord)
    out.loc[name, 'ds_exit_final'] = displ_start[1] + final_chord[1] - row['length_chord']
    for component, delta in zip('rst', final_chord-chord_rst):
        out.loc[name, f'd{component}_exit_from_roll'] = delta

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

# An unreported exit stays at nominal transverse coordinates, including for
# instruments whose only socket is at their centre rather than their entrance.
single_point_names = out.index[out['single_point']]
for name in single_point_names:
    np.testing.assert_allclose(rst_offsets[name][1][[0, 2]], [0., 0.],
                               atol=1e-12, rtol=0)
print(f'  {len(single_point_names)} unreported exits have zero R and T')

# Closed-form checks of the RST -> MAD-X mapping, on a straight untilted
# element of unit chord: R = -x, S = +s, T = +y, and the SU roll is minus the
# MAD-X dpsi. Everything is expressed in the element's own tilted chord frame,
# so a non-zero design tilt rotates R and T into dx and dy (as seen e.g. on the
# vertical benders MBNV, which have tilt = pi/2).
def check(displ_e, displ_s, bgamma=0.):
    return su.misalignment_from_geode_displacements(
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

print('\n=== requested exit S displacement dropped for rigidity, before roll ===')
dropped = out['ds_exit_dropped'].abs() > 1e-6
print(f'{dropped.sum()} elements differ by more than 1 um '
      f'(max {out["ds_exit_dropped"].abs().max() * 1e3:.3f} mm); '
      f'{out["ds_exit_requested"].isna().sum()} single-point elements '
      f'requested nothing at the exit')
if dropped.any():
    print(out.loc[dropped, ['length_chord', 'ds_exit_requested',
                            'ds_exit_rigid', 'ds_exit_dropped']].sort_values(
        'ds_exit_dropped', key=abs, ascending=False).to_string())

print('\n=== additional exit displacement from entrance-tangent roll ===')
roll_columns = ['dr_exit_from_roll', 'ds_exit_from_roll', 'dt_exit_from_roll']
rolled = out[roll_columns].abs().max(axis=1) > 1e-6
print(f'{rolled.sum()} elements move by more than 1 um in at least one component')
if rolled.any():
    print(out.loc[rolled, roll_columns + ['ds_exit_final']].to_string())

print('\n=== magnitudes ===')
for column in misalignment_columns:
    unit = 'm' if column in ('dx', 'dy', 'ds') else 'rad'
    print(f'{column:14s} max|.| = {out[column].abs().max():.6e} {unit}   '
          f'n non-zero = {(~np.isclose(out[column], 0.)).sum()}')

out.to_csv('bump_misalignments.csv')
print('\nwritten to bump_misalignments.csv')
