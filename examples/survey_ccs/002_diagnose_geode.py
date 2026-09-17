"""Compare the supplied GEODE snapshots without overwriting their inputs.

Run with XSUITE_ALLOW_KERNEL_COMPILATION=1
/Users/giadarol/miniforge3/envs/py313/bin/python 002_diagnose_geode.py.
All vector comparisons use CERN X,Y,Z (Z is elevation), in metres internally.
Results for the radial-sign-corrected import go into second_diagnostics/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tfs
import xtrack as xt
from xtrack._temp import survey_utils as su

from bumps_report import read_bumps_report

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'second_diagnostics'
# These inputs describe the H4TEST3 import, even after a subsequent regeneration.
IMPORT_INPUTS = OUTPUT
XYZ = ['X (m)', 'Y (m)', 'Z (m)']
C = np.array([[-1., 0, 0], [0, 0, 1], [0, 1, 0]])


def read_geode(name):
    return pd.read_csv(ROOT / name, encoding='cp1252').set_index('Elément')


def survey(line):
    if line.tracker is None:
        line.build_tracker(compile=False)
    f = xt.Frame.from_ccs(xt.CCSFrame(
        x=669.234140, y=4590.727900, z=2441.574200,
        theta_gon=7.4198200, phi=-0.000370000))
    return line.survey(include_element_frames=True, X0=f.X, Y0=f.Y, Z0=f.Z,
                       theta0=f.theta, phi0=f.phi, psi0=f.psi)


def main():
    OUTPUT.mkdir(exist_ok=True)
    bumps = read_bumps_report(ROOT / 'Bumps_sp_report.csv')
    misalignments = pd.read_csv(IMPORT_INPUTS / 'bump_misalignments.csv', index_col=0)
    old_beam = read_geode('geode_beam_points_from_mad_and_bumps.csv')
    new_beam = read_geode('geode_beam_points_xsuite_no_bumps.csv')
    old_socket = read_geode('geode_socket_points_mad_and_bumps.csv')
    new_socket = read_geode('geode_socket_points_xsuite_no_bumps.csv')
    for kind, before, after in [('beam', old_beam, new_beam),
                                ('socket', old_socket, new_socket)]:
        assert before.index.is_unique and after.index.is_unique, kind
        print(f'{kind}: {len(before.index.intersection(after.index))} matched; '
              f'legacy only: {before.index.difference(after.index).tolist()}; '
              f'new only: {after.index.difference(before.index).tolist()}')
    env = xt.load(ROOT / 'survey-h4-post-ls3-cern-coords-v4.seq')
    nominal = survey(env['h4'])
    # The legacy CSV starts at 7.419824 gon; the H4TEST3 import used 7.419820 gon.
    seed = dict(x=669.234140, y=4590.727900, z=2441.574200, phi=-0.000370000)
    baseline_transform = (xt.Frame.from_ccs(xt.CCSFrame(**seed, theta_gon=7.419824))
                          @ xt.Frame.from_ccs(xt.CCSFrame(**seed, theta_gon=7.419820)).inverse())
    aligned_line = xt.Line.from_json(IMPORT_INPUTS / 'h4_misaligned.json')
    aligned = survey(aligned_line)
    # The H4TEST3 import includes the radial-sign fix but copied entrance bumps
    # to unreported exits. The reader now defaults those exit R/T bumps to zero.
    for name, request in bumps.iterrows():
        row = misalignments.loc[name]
        start = request[['r_entry', 's_entry', 't_entry']].to_numpy(float)
        end = request[['r_exit', 's_exit', 't_exit']].to_numpy(float)
        su.misalignment_from_rst_displacements(
            start, end, row.length_chord, bgamma=request['roll'],
            tilt=row.tilt, angle=row.angle).apply_to_element(aligned_line[name.lower()])
    corrected = survey(aligned_line)
    # Separate diagnostic hypothesis: infer the crab before adding roll.
    # On a bend, rolling about the entrance tangent then also moves the chord.
    for name, request in bumps.iterrows():
        if not request['roll']:
            continue
        row = misalignments.loc[name]
        start = request[['r_entry', 's_entry', 't_entry']].to_numpy(float)
        end = request[['r_exit', 's_exit', 't_exit']].to_numpy(float)
        mis = su.misalignment_from_rst_displacements(
            start, end, row.length_chord, bgamma=0., tilt=row.tilt, angle=row.angle)
        mis.dpsi -= request['roll']
        mis.apply_to_element(aligned_line[name.lower()])
    tangent_roll = survey(aligned_line)
    exported = tfs.read(IMPORT_INPUTS / 'h4_survey_comp_vbend_output.tfs')
    exported_points = {}
    for i in range(1, len(exported), 2):
        for j, suffix in [(i-1, '.E'), (i, '.S')]:
            exported_points[exported.iloc[i].NAME + suffix] = (
                C @ exported.iloc[j][['X', 'Y', 'Z']].to_numpy(float))

    rows = []
    for point in old_beam.index.intersection(new_beam.index):
        name, suffix = point.rsplit('.', 1)
        place = 'start' if suffix == 'E' else 'end'
        p0 = C @ nominal[f'XYZ_elem_{place}', name.lower()]
        p1 = C @ aligned[f'XYZ_elem_{place}', name.lower()]
        old = old_beam.loc[point, XYZ].to_numpy(float)
        new = new_beam.loc[point, XYZ].to_numpy(float)
        row = {'point': point, 'name': name}
        for label, delta in [('new_minus_old', new-old),
                             ('old_minus_nominal', old-p0),
                             ('old_minus_nominal_legacy_seed', old-C @ (
                                 baseline_transform.XYZ + baseline_transform.E_matrix @ (C.T @ p0))),
                             ('new_minus_aligned', new-p1),
                             ('new_minus_tfs', new-exported_points[point])]:
            for axis, value in zip('xyz', delta):
                row[f'{label}_{axis}_mm'] = value*1000
            row[f'{label}_norm_mm'] = np.linalg.norm(delta)*1000
        rows.append(row)
    beam = pd.DataFrame(rows).set_index('point')
    beam.to_csv(OUTPUT / 'geode_diagnostic_beam.csv')
    print('\nBeam residual norms (mm):')
    print(beam.filter(like='_norm_mm').agg(['max', 'median']).to_string())

    rows = []
    for point in old_socket.index.intersection(new_socket.index):
        a, b = old_socket.loc[point], new_socket.loc[point]
        name = a['Nom Layout']
        nn = name.lower()
        element = env['h4'][nn]
        length = getattr(element, 'length_straight', None) or element.length
        tilt = (misalignments.loc[name, 'tilt'] if name in misalignments.index
                else getattr(element, 'rot_s_rad', 0.))
        _, rst = su.rst_from_reference_start(
            nominal['XYZ_ref_start', nn], nominal['E_ref_start', nn],
            tilt, getattr(element, 'angle', 0.))
        basis = C @ rst
        delta = b[XYZ].to_numpy(float) - a[XYZ].to_numpy(float)
        req = bumps.loc[name] if name in bumps.index else None
        # Socket S is measured from Entry or Centre, not from the beam exit.
        s = a['S (m)'] + (length/2 if a['Origine'] == 'Centre' else 0.)
        fraction = s/length
        request = np.zeros(3) if req is None else np.array([
            (1-fraction)*req[f'{axis}_entry'] + fraction*req[f'{axis}_exit']
            for axis in 'rst'])
        old_frame = aligned.get_frame(nn, which='elem_start')
        new_frame = corrected.get_frame(nn, which='elem_start')
        local_socket = old_frame.E_matrix.T @ (C.T @ b[XYZ].to_numpy(float) - old_frame.XYZ)
        predicted = C @ (new_frame.XYZ + new_frame.E_matrix @ local_socket)
        corrected_residual = predicted-a[XYZ].to_numpy(float)
        predicted_legacy_seed = C @ (baseline_transform.XYZ
                                     + baseline_transform.E_matrix @ (C.T @ predicted))
        final_residual = predicted_legacy_seed-a[XYZ].to_numpy(float)
        tangent_frame = tangent_roll.get_frame(nn, which='elem_start')
        tangent_socket = tangent_frame.XYZ + tangent_frame.E_matrix @ local_socket
        tangent_residual = C @ (baseline_transform.XYZ + baseline_transform.E_matrix @ tangent_socket) - a[XYZ].to_numpy(float)
        row = {'point': point, 'name': name,
               'norm_mm': np.linalg.norm(delta)*1000,
               'after_missing_exit_norm_mm': np.linalg.norm(corrected_residual)*1000,
               'after_seed_correction_norm_mm': np.linalg.norm(final_residual)*1000,
               'tangent_roll_hypothesis_norm_mm': np.linalg.norm(tangent_residual)*1000,
               'geometric_r_at_socket_mm': request[0]*1000,
               'single_point': bool(req.single_point) if req is not None else False,
               'origin': a['Origine'],
               'roll_difference_rad': b['Roll Elément (rad)']-a['Roll Elément (rad)'],
               'socket_rst_changed': not np.allclose(
                   a[['R (m)', 'S (m)', 'T (m)']].to_numpy(float),
                   b[['R (m)', 'S (m)', 'T (m)']].to_numpy(float), atol=0, rtol=0)}
        for axis, value in zip('rst', basis.T @ delta):
            row[f'delta_{axis}_mm'] = value*1000
        for axis, value in zip('xyz', delta):
            row[f'delta_{axis}_mm'] = value*1000
        for axis, value in zip('xyz', corrected_residual):
            row[f'after_missing_exit_{axis}_mm'] = value*1000
        for axis, value in zip('xyz', final_residual):
            row[f'after_seed_correction_{axis}_mm'] = value*1000
        for axis, value in zip('xyz', tangent_residual):
            row[f'tangent_roll_hypothesis_{axis}_mm'] = value*1000
        rows.append(row)
    sockets = pd.DataFrame(rows).set_index('point')
    sockets.to_csv(OUTPUT / 'geode_diagnostic_socket.csv')
    # These roll-convention changes are intentional in the supplied exercise.
    comparable = sockets.loc[~sockets.name.str.startswith(('MCXCA.', 'MDSV.'))]
    cols = ['name', 'norm_mm', 'after_missing_exit_norm_mm', 'after_seed_correction_norm_mm',
            'tangent_roll_hypothesis_norm_mm']
    print('\nLargest socket differences, excluding intentional corrector roll cases:')
    print(comparable.sort_values('norm_mm', ascending=False)[cols].head(12).to_string())
    print('\nLargest residuals after missing-exit and seed corrections:')
    print(comparable.sort_values('after_seed_correction_norm_mm', ascending=False)
          [cols].head(16).to_string())
    print('\nSummary (mm):')
    print(comparable[cols[1:]].agg(['max', 'median']).to_string())
    print('Sockets below 5 um under tangent-roll hypothesis: '
          f'{(comparable.tangent_roll_hypothesis_norm_mm < .005).sum()}/{len(comparable)}')
    previous_path = ROOT / 'first_diagnostics' / 'geode_diagnostic_socket.csv'
    if previous_path.exists():
        previous = pd.read_csv(previous_path, index_col='point')
        comparison = comparable[['name', 'norm_mm']].rename(columns={'norm_mm': 'current_norm_mm'})
        comparison['previous_norm_mm'] = previous['norm_mm']
        comparison.to_csv(OUTPUT / 'geode_import_comparison.csv')
        print('\nLargest discrepancies in the previous import, compared with this import:')
        print(comparison.sort_values('previous_norm_mm', ascending=False).head(16).to_string())


if __name__ == '__main__':
    main()
