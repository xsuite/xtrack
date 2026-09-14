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
# roll about the chord. `su.misalignment_from_rst_offsets` wants the *absolute*
# RST positions of the two end points, both measured from the nominal entrance:
# a nominal element runs from (0, 0, 0) to (0, L_chord, 0) in that frame,
# independently of its tilt and bending angle (verified at the end of this
# script).
#
# The two end-point displacements plus the roll are seven numbers, while a rigid
# body has only six degrees of freedom, so the request is over-determined: in
# general no rigid motion puts both end points exactly where asked. The element
# is kept rigid here, which means its chord length is preserved. The transverse
# (R, T) part of the requested exit displacement fixes the chord direction, and
# the exit longitudinal (S) coordinate then follows from the fixed length, so
# the requested exit S displacement is discarded. Its entrance counterpart is
# kept: that one is the `ds` shift of the whole element.

# RST component order matches survey_utils: E_rst = column_stack((er, es, et)).
RST_COLUMNS = ['Radial (m)', 'Longitudinal (m)', 'Vertical (m)']
ROLL_COLUMN = 'Roll (rad)'


def report_value(row, column):
    """Read one cell; a blank means no bump requested, i.e. zero."""
    value = row[column]
    return 0. if pd.isna(value) else float(value)


# ------------------------------------------------------- read the bump report

df = pd.read_csv('Bumps_sp_report.csv')

# The accented character in 'Elément' was lost in the csv export
df.rename(columns={'El�ment': 'Element'}, inplace=True)

# Each row of the report is a *point*, not an element: the name ends in `.E`
# (entrée, the element start) or `.S` (sortie, the element end). Collect the
# requested displacement of each point, plus the roll, per element.
requests = {}
for _, row in df.iterrows():
    name, point = row['Element'].rsplit('.', 1)
    assert name == row['Nom Layout']
    assert point in ('E', 'S')

    request = requests.setdefault(name, {})
    request[point] = np.array([report_value(row, cc) for cc in RST_COLUMNS])

    # The roll is stored redundantly on both points; check the two agree.
    roll = report_value(row, ROLL_COLUMN)
    assert request.setdefault('roll', roll) == roll
    request['roll'] = roll

element_names = sorted(requests)

# --------------------------------------------------------------- load lattice

env = xt.load('survey-h4-post-ls3-cern-coords-v4.seq')
line = env['h4']

# ------------------------------------------- convert, one element at a time

results = []
for name in element_names:
    request = requests[name]
    roll = request['roll']
    displ_start = request['E']

    # The thin instruments (XSCI, XDWC) are reported on their entrance point
    # only. One point cannot define a rotation, so the request is a rigid
    # translation: the exit moves with the entrance.
    single_point = 'S' not in request
    displ_end = displ_start if single_point else request['S']

    element = line[name.lower()]

    # Chord length: the RBends keep the arc in `length` and the chord in
    # `length_straight`. The elements modelled as drifts (instruments,
    # collimators) carry neither a tilt nor a bending angle.
    length = getattr(element, 'length_straight', None) or element.length
    tilt = getattr(element, 'rot_s_rad', 0.) or 0.
    angle = getattr(element, 'angle', 0.) or 0.

    # Chord of the displaced element. Its R and T components are set by the
    # requested end-point displacements (the nominal end points both have
    # R = T = 0), while its S component is the one that keeps the chord length
    # equal to the nominal one, i.e. that keeps the element rigid.
    chord_r = displ_end[0] - displ_start[0]
    chord_t = displ_end[2] - displ_start[2]
    transverse_sq = chord_r**2 + chord_t**2
    assert transverse_sq < length**2, \
        f'{name}: transverse bump larger than the element chord'
    chord_s = np.sqrt(length**2 - transverse_sq)

    # Nominal entrance at (0, 0, 0), nominal exit at (0, length, 0).
    offset_start_rst = displ_start
    offset_end_rst = displ_start + np.array([chord_r, chord_s, chord_t])

    mis = su.misalignment_from_rst_offsets(
        offset_start_rst, offset_end_rst, bgamma=roll, tilt=tilt, angle=angle)

    # Longitudinal exit displacement that the rigid motion produces, against
    # the one that was requested and had to be dropped.
    ds_exit_rigid = offset_end_rst[1] - length
    ds_exit_dropped = displ_end[1] - ds_exit_rigid

    # Push the misalignment back through the forward transformation, at the
    # nominal chord length, and check the RST end points are recovered. A
    # stand-in object is used so that the check also covers the elements
    # modelled as drifts, which have no misalignment attributes, and so that
    # the loaded line is left untouched.
    probe = SimpleNamespace(
        angle=angle,
        rot_s_rad=tilt,
        rot_y_rad=mis.dtheta,
        rot_x_rad=mis.dphi,
        rot_s_rad_no_frame=mis.dpsi - tilt,
        shift_x=mis.shift_x,
        shift_y=mis.shift_y,
        shift_s=mis.shift_s,
    )
    back_start, back_end = su.rst_start_end_offsets_from_parameters(
        probe, length)
    round_trip_error = max(np.max(np.abs(back_start - offset_start_rst)),
                           np.max(np.abs(back_end - offset_end_rst)))
    assert round_trip_error < 1e-12, f'{name}: round trip failed'

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
        'ds_exit_dropped': ds_exit_dropped,
        'round_trip_error': round_trip_error,
    })

out = pd.DataFrame(results).set_index('name')

# ------------------------------------------------------- convention checks

# Closed-form checks of the RST -> MAD-X mapping, on a straight untilted
# element of unit chord: R = -x, S = +s, T = +y, and the SU roll is minus the
# MAD-X dpsi. Everything is expressed in the element's own tilted chord frame,
# so a non-zero design tilt rotates R and T into dx and dy (as seen e.g. on the
# vertical benders MBNV, which have tilt = pi/2).
def check(displ_e, displ_s, bgamma=0.):
    return su.misalignment_from_rst_offsets(
        np.array(displ_e, dtype=float),
        np.array([0., 1., 0.]) + np.array(displ_s, dtype=float),
        bgamma=bgamma)


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
assert np.isclose(mm.dtheta, np.arctan(2 * dd))
mm = check([0, 0, dd], [0, 0, -dd])                 # vertical crab
assert np.isclose(mm.dphi, -np.arctan(2 * dd))

# ---------------------------------------------------------------------- report

print(f'round trip: max error over {len(out)} elements = '
      f'{out["round_trip_error"].max():.2e} m')

misalignment_columns = ['dtheta', 'dphi', 'dpsi_no_tilt', 'dx', 'dy', 'ds']
moved = ~np.isclose(out[misalignment_columns], 0.).all(axis=1)
print(f'{moved.sum()} of {len(out)} elements are actually misaligned\n')

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

out.drop(columns='round_trip_error').to_csv('bump_misalignments.csv')
print('\nwritten to bump_misalignments.csv')
