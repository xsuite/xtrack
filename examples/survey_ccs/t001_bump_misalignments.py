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
RST = ['Radial (m)', 'Longitudinal (m)', 'Vertical (m)']
ROLL = 'Roll (rad)'

# ----------------------------------------------------------------- bump report

df = pd.read_csv('Bumps_sp_report.csv')

# The accented character in 'Elément' was lost in the csv export
df.rename(columns={'El�ment': 'Element'}, inplace=True)

df['layout'] = df['Element'].str.rsplit('.', n=1).str[0]
df['point'] = df['Element'].str.rsplit('.', n=1).str[1]
assert (df['layout'] == df['Nom Layout']).all()

piv = {kk: df.pivot_table(index='layout', columns='point', values=kk,
                          dropna=False).reindex(columns=['E', 'S'])
       for kk in RST + [ROLL]}
elements = piv[ROLL].index

# A blank cell means "no bump requested on this axis", i.e. zero displacement.
displ_start = np.column_stack([piv[kk]['E'].fillna(0.).values for kk in RST])
displ_end = np.column_stack([piv[kk]['S'].fillna(0.).values for kk in RST])

# The roll is stored redundantly on both points; check the two agree, then use
# it as the single per-element `bgamma`.
both = piv[ROLL].notna().all(axis=1)
assert np.array_equal(piv[ROLL]['E'][both].values, piv[ROLL]['S'][both].values)
roll = piv[ROLL]['E'].fillna(0.).values

# The thin instruments (XSCI, XDWC) are reported on their entrance point only.
# One point cannot define a rotation, so the request is a rigid translation.
points = df.groupby('layout')['point'].apply(lambda ss: ''.join(sorted(ss)))
single_point = (points.reindex(elements) == 'E').values
displ_end[single_point] = displ_start[single_point]

# --------------------------------------------------------------------- lattice

env = xt.load('survey-h4-post-ls3-cern-coords-v4.seq')
line = env['h4']

# Chord length, design tilt and bending angle of each element. The elements
# modelled as drifts (instruments, collimators) carry none of these attributes.
length = np.array([getattr(line[nn.lower()], 'length_straight', None)
                   or line[nn.lower()].length for nn in elements])
tilt = np.array([getattr(line[nn.lower()], 'rot_s_rad', 0.) or 0.
                 for nn in elements])
angle = np.array([getattr(line[nn.lower()], 'angle', 0.) or 0.
                  for nn in elements])
el_type = [type(line[nn.lower()]._xobject).__name__.replace('Data', '')
           for nn in elements]

# ---------------------------------------------------------------- conversion

offset_start_rst = displ_start

# Chord of the displaced element. Its R and T components are set by the
# requested end-point displacements (the nominal end points have R = T = 0),
# while its S component is what keeps the chord length equal to the nominal one.
chord_rst = np.zeros_like(displ_start)
chord_rst[:, 0] = displ_end[:, 0] - displ_start[:, 0]
chord_rst[:, 2] = displ_end[:, 2] - displ_start[:, 2]
transverse_sq = chord_rst[:, 0]**2 + chord_rst[:, 2]**2
assert (transverse_sq < length**2).all(), \
    'transverse bump larger than the element chord'
chord_rst[:, 1] = np.sqrt(length**2 - transverse_sq)

offset_end_rst = offset_start_rst + chord_rst

# Longitudinal exit displacement that the rigid motion produces, against the
# one that was requested and had to be dropped.
displ_end_s_rigid = offset_end_rst[:, 1] - length
displ_end_s_dropped = displ_end[:, 1] - displ_end_s_rigid

mis = [su.misalignment_from_rst_offsets(
           offset_start_rst[ii], offset_end_rst[ii], bgamma=roll[ii],
           tilt=tilt[ii], angle=angle[ii])
       for ii in range(len(elements))]

out = pd.DataFrame({
    'element_type': el_type,
    'length_chord': length,
    'tilt': tilt,
    'angle': angle,
    'dtheta': [mm.dtheta for mm in mis],
    'dphi': [mm.dphi for mm in mis],
    'dpsi': [mm.dpsi for mm in mis],
    'dx': [mm.shift_x for mm in mis],
    'dy': [mm.shift_y for mm in mis],
    'ds': [mm.shift_s for mm in mis],
    'single_point': single_point,
}, index=elements)
out.index.name = 'name'

# `dpsi` returned by survey_utils includes the design tilt; the bump-induced
# part alone is what MAD-X would receive on top of the nominal tilt.
out['dpsi_no_tilt'] = out['dpsi'] - out['tilt']

# ------------------------------------------------------- convention checks

# Closed-form checks of the RST -> MAD-X mapping, on a straight untilted
# element of unit chord: R = -x, S = +s, T = +y, and the SU roll is minus the
# MAD-X dpsi. Everything is expressed in the element's own tilted chord frame,
# so a non-zero design tilt rotates R and T into dx and dy (as seen e.g. on the
# vertical benders MBNV, which have tilt = pi/2).
def _check(displ_e, displ_s, bgamma=0.):
    return su.misalignment_from_rst_offsets(
        np.array(displ_e, dtype=float),
        np.array([0., 1., 0.]) + np.array(displ_s, dtype=float),
        bgamma=bgamma)

dd = 1e-3
mm = _check([dd, 0, 0], [dd, 0, 0])                  # radial translation
assert np.allclose([mm.shift_x, mm.shift_y, mm.shift_s], [-dd, 0, 0])
mm = _check([0, dd, 0], [0, dd, 0])                  # longitudinal translation
assert np.allclose([mm.shift_x, mm.shift_y, mm.shift_s], [0, 0, dd])
mm = _check([0, 0, dd], [0, 0, dd])                  # vertical translation
assert np.allclose([mm.shift_x, mm.shift_y, mm.shift_s], [0, dd, 0])
mm = _check([0, 0, 0], [0, 0, 0], bgamma=dd)         # roll
assert np.isclose(mm.dpsi, -dd)
mm = _check([dd, 0, 0], [-dd, 0, 0])                 # radial crab
assert np.isclose(mm.dtheta, np.arctan(2 * dd))
mm = _check([0, 0, dd], [0, 0, -dd])                 # vertical crab
assert np.isclose(mm.dphi, -np.arctan(2 * dd))

# ------------------------------------------------------- round-trip validation

# The chord built above is rigid by construction, so the length that
# `misalignment_from_rst_offsets` infers from it is the nominal one.
assert np.allclose(np.linalg.norm(offset_end_rst - offset_start_rst, axis=1),
                   length, rtol=0, atol=1e-12)

# Push the misalignments back through the forward transformation, at the
# nominal chord length, and check the RST end points are recovered. A stand-in
# object is used so that the check also covers the elements modelled as drifts,
# which have no misalignment attributes, and so that the line is left untouched.
err = np.zeros(len(elements))
for ii, nn in enumerate(elements):
    probe = SimpleNamespace(
        angle=angle[ii],
        rot_s_rad=tilt[ii],
        rot_y_rad=mis[ii].dtheta,
        rot_x_rad=mis[ii].dphi,
        rot_s_rad_no_frame=mis[ii].dpsi - tilt[ii],
        shift_x=mis[ii].shift_x,
        shift_y=mis[ii].shift_y,
        shift_s=mis[ii].shift_s,
    )
    back_start, back_end = su.rst_start_end_offsets_from_parameters(
        probe, length[ii])
    err[ii] = max(np.max(np.abs(back_start - offset_start_rst[ii])),
                  np.max(np.abs(back_end - offset_end_rst[ii])))

print(f'round trip: max error over {len(elements)} elements = {err.max():.2e} m'
      f'  (worst: {elements[err.argmax()]})')
assert err.max() < 1e-12

# ---------------------------------------------------------------------- report

moved = ~np.isclose(out[['dtheta', 'dphi', 'dpsi_no_tilt', 'dx', 'dy', 'ds']],
                    0.).all(axis=1)
print(f'{moved.sum()} of {len(out)} elements are actually misaligned\n')

cols = ['element_type', 'length_chord', 'dtheta', 'dphi', 'dpsi_no_tilt',
        'dx', 'dy', 'ds']
with pd.option_context('display.width', 200, 'display.max_rows', None):
    print(out.loc[moved, cols].to_string(
        float_format=lambda vv: f'{vv: .6e}' if abs(vv) > 0 else f'{0.: .6e}'))

print('\n=== requested exit S displacement dropped to keep the element rigid '
      '===')
out['ds_exit_requested'] = displ_end[:, 1]
out['ds_exit_rigid'] = displ_end_s_rigid
out['ds_exit_dropped'] = displ_end_s_dropped
dropped = out['ds_exit_dropped'].abs() > 1e-6
print(f'{dropped.sum()} elements differ by more than 1 um '
      f'(max {out["ds_exit_dropped"].abs().max()*1e3:.3f} mm)')
if dropped.any():
    print(out.loc[dropped, ['length_chord', 'ds_exit_requested',
                            'ds_exit_rigid', 'ds_exit_dropped']].sort_values(
        'ds_exit_dropped', key=abs, ascending=False).to_string())

print('\n=== magnitudes ===')
for cc in ['dtheta', 'dphi', 'dpsi_no_tilt', 'dx', 'dy', 'ds']:
    unit = 'rad' if cc.startswith(('dt', 'dph', 'dps')) else 'm'
    print(f'{cc:14s} max|.| = {out[cc].abs().max():.6e} {unit}   '
          f'n non-zero = {(~np.isclose(out[cc], 0.)).sum()}')

out.to_csv('bump_misalignments.csv')
print('\nwritten to bump_misalignments.csv')
