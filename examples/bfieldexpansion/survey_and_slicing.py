"""Survey and lab-frame tracking through two dipoles, before and after slicing.

Run with::

    python -m examples.bfieldexpansion.survey_and_slicing
    python -m examples.bfieldexpansion.survey_and_slicing --no-plot --save-plot /tmp/survey.png

Each bend has a total length of 1 m, including both fringes, and an angle of
30 degrees. The field-free gap between the magnets is 50 cm. Each 30 cm
fringe has two cubic spline segments, one straight and one curved.
Thus each magnet consists of 15 cm straight entrance fringe, 15 cm curved
entrance fringe, 40 cm flat body, 15 cm curved exit fringe, and 15 cm straight
exit fringe. The curved length is 70 cm, so h = (30 degrees) / (0.7 m).

Slicing shares each parent's polynomial; it does not refit the fringe.
The RK4 step counts are chosen to keep the integration grid the same.
Unsliced particle positions are shown only at the original boundaries;
the sliced line supplies the additional trajectory samples between them.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicHermiteSpline

import xtrack as xt


BEND_LENGTH = 1.0
BEND_ANGLE = np.deg2rad(30.)
GAP = 0.5
FRINGE_LENGTH = 0.3
BODY_LENGTH = BEND_LENGTH - 2 * FRINGE_LENGTH
CURVED_LENGTH = BEND_LENGTH - FRINGE_LENGTH
H = BEND_ANGLE / CURVED_LENGTH


def make_expansion(length, h, coefficients, num_integration_steps):
    coefficients = np.asarray(coefficients)
    return xt.BFieldExpansion(
        length=length, h=h, knc=coefficients[None, :],
        ksc=np.zeros((1, len(coefficients))), ksol=np.zeros_like(coefficients),
        num_phi=5, num_integration_steps=num_integration_steps, pkin_const=False)


def make_line(num_integration_steps):
    # These two Hermite cubics reproduce H * (3*t**2 - 2*t**3), t=s/F.
    # The on-axis field and its first derivative are continuous at every join.
    knots = np.array([0., FRINGE_LENGTH / 2, FRINGE_LENGTH])
    entrance = CubicHermiteSpline(knots, [0., H / 2, H],
                                 [0., 1.5 * H / FRINGE_LENGTH, 0.])
    exit_ = CubicHermiteSpline(knots, [H, H / 2, 0.],
                             [0., -1.5 * H / FRINGE_LENGTH, 0.])
    elements = {}
    for bend in ('b1', 'b2'):
        # scipy uses descending powers of s-knots[i]; BFieldExpansion uses
        # ascending powers. Each parent segment starts at local s_start=0.
        for i, geometry_h in enumerate((0., H)):
            elements[f'{bend}_entry_{i}'] = make_expansion(
                FRINGE_LENGTH / 2, geometry_h, entrance.c[::-1, i], num_integration_steps)
        elements[f'{bend}_body'] = make_expansion(
            BODY_LENGTH, H, [H], num_integration_steps)
        for i, geometry_h in enumerate((H, 0.)):
            elements[f'{bend}_exit_{i}'] = make_expansion(
                FRINGE_LENGTH / 2, geometry_h, exit_.c[::-1, i], num_integration_steps)
        if bend == 'b1':
            elements['gap'] = xt.Drift(length=GAP)
    line = xt.Line(elements=elements)
    line.configure_drift_model(model='exact')
    return line


def track_in_lab(line, initial):
    survey = line.survey()
    particles = initial.copy()
    line.track(particles, turn_by_turn_monitor='ONE_TURN_EBE',
               _force_no_end_turn_actions=True)
    monitor = line.record_last_track
    assert np.all(monitor.state[0] > 0)
    np.testing.assert_allclose(monitor.s[0], survey.s, rtol=0, atol=1e-13)

    # Each record is at its element's entrance plane (including the endpoint).
    # zeta describes timing; it is not a displacement along the surveyed axis.
    trajectory = (survey.XYZ + monitor.x[0, :, None] * survey.ex
                  + monitor.y[0, :, None] * survey.ey)
    return survey, monitor, trajectory


def matching_rows(source_s, target_s):
    """Find the same observation planes, allowing duplicate marker rows."""
    indices = np.abs(source_s[:, None] - target_s[None, :]).argmin(axis=1)
    np.testing.assert_allclose(source_s, target_s[indices], rtol=0, atol=1e-13)
    return indices


def plot_comparison(line, survey, trajectory, sliced_survey, sliced_trajectory,
                    survey_error, trajectory_error):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    field_ax, horizontal_ax, error_ax, vertical_ax = axes.ravel()
    profile_s, profile_by, profile_h, smooth_survey = [], [], [], []
    for i, name in enumerate(line.element_names):
        element = line[name]
        local_s = np.linspace(0., element.length, 101)
        h = getattr(element, 'h', 0.)
        by = (element.get_field(x=0., y=0., s_local=local_s)['By']
              if isinstance(element, xt.BFieldExpansion) else np.zeros_like(local_s))
        profile_s.extend(survey.s[i] + local_s)
        profile_by.extend(by)
        profile_h.extend(np.full_like(local_s, h))
        # Draw the unsliced survey as actual arcs, not chords between its rows.
        frame = survey.get_frame(name)
        smooth_survey.extend(frame.copy().arc_x(length=ss, angle=h * ss).XYZ
                             for ss in local_s)
    smooth_survey = np.asarray(smooth_survey)

    field_ax.plot(profile_s, profile_by, label=r'On-axis $B_y/(B\rho)$')
    field_ax.plot(profile_s, profile_h, '--', label=r'Reference curvature $h$')
    for i in range(2):
        edge = i * (BEND_LENGTH + GAP)
        field_ax.axvspan(edge, edge + BEND_LENGTH, color='C0', alpha=0.07)
        field_ax.text(edge + BEND_LENGTH / 2, 0.95 * H, f'Bend {i + 1}',
                      ha='center', va='top')
    field_ax.set(xlabel='Reference s [m]', ylabel=r'[$\mathrm{m}^{-1}$]',
                 title='Two 1 m / 30° bends including fringes, 50 cm gap')
    field_ax.legend(loc='lower center')

    for ax, coordinate, scale, label in ((horizontal_ax, 0, 1., 'X [m]'),
                                         (vertical_ax, 1, 1e3, 'Y [mm]')):
        ax.plot(smooth_survey[:, 2], scale * smooth_survey[:, coordinate],
                color='0.25', ls='--', label='Survey: unsliced arcs')
        ax.plot(sliced_survey.Z, scale * sliced_survey.XYZ[:, coordinate],
                '.', color='0.55', ms=3, label='Survey: sliced boundaries')
        ax.plot(sliced_trajectory[:, 2], scale * sliced_trajectory[:, coordinate],
                color='C0', label='Particle: sliced')
        ax.plot(trajectory[:, 2], scale * trajectory[:, coordinate],
                'o', color='C1', mfc='none', ms=6, label='Particle: unsliced boundaries')
        ax.set(xlabel='Z [m]', ylabel=label)
    horizontal_ax.set_title('Horizontal lab-frame trajectory')
    horizontal_ax.set_aspect('equal', adjustable='datalim')
    horizontal_ax.legend(fontsize=8)
    vertical_ax.set_title('Vertical lab-frame trajectory')

    error_ax.plot(survey.s, survey_error, '.-', label='Survey position')
    error_ax.plot(survey.s, trajectory_error, 'o-', mfc='none', label='Particle position')
    error_ax.set(xlabel='Reference s [m]', ylabel=r'$\|\Delta(X,Y,Z)\|$ [m]',
                 title='Sliced − unsliced at the original boundaries')
    error_ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    error_ax.legend()
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.suptitle('BFieldExpansion: survey and particle tracking through spline fringes')
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--slices', type=int, default=20,
                        help='Thick slices per field segment (default: 20)')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--save-plot', type=Path)
    args = parser.parse_args()
    if args.slices < 1:
        parser.error('--slices must be positive')

    # 20 RK4 steps per slice, with the same total steps in the unsliced parent.
    line = make_line(num_integration_steps=20 * args.slices)
    sliced = line.copy()
    sliced.slice_thick_elements([
        xt.Strategy(xt.Uniform(args.slices, mode='thick'),
                    element_type=xt.BFieldExpansion)])
    initial = xt.Particles(p0c=7e9, x=3e-3, px=2e-4,
                           y=2e-3, py=1e-4, delta=0.01)
    survey, monitor, trajectory = track_in_lab(line, initial)
    sliced_survey, sliced_monitor, sliced_trajectory = track_in_lab(sliced, initial)
    indices = matching_rows(survey.s, sliced_survey.s)

    np.testing.assert_allclose(sliced_survey.XYZ[indices], survey.XYZ, rtol=0, atol=1e-13)
    np.testing.assert_allclose(sliced_survey.E_matrix[indices], survey.E_matrix,
                               rtol=0, atol=1e-13)
    np.testing.assert_allclose(sliced_trajectory[indices], trajectory, rtol=0, atol=1e-11)
    for coordinate in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
        np.testing.assert_allclose(getattr(sliced_monitor, coordinate)[0, indices],
                                   getattr(monitor, coordinate)[0], rtol=0, atol=1e-11)

    # Independent geometry check: two ordinary sector bends at nominal edges.
    nominal = xt.Line(elements=[
        xt.Drift(length=FRINGE_LENGTH / 2),
        xt.Bend(length=CURVED_LENGTH, angle=BEND_ANGLE),
        xt.Drift(length=GAP + FRINGE_LENGTH),
        xt.Bend(length=CURVED_LENGTH, angle=BEND_ANGLE),
        xt.Drift(length=FRINGE_LENGTH / 2),
    ]).survey()
    nominal_indices = matching_rows(nominal.s, survey.s)
    np.testing.assert_allclose(survey.XYZ[nominal_indices], nominal.XYZ, rtol=0, atol=1e-13)
    np.testing.assert_allclose(survey.E_matrix[nominal_indices], nominal.E_matrix,
                               rtol=0, atol=1e-13)
    np.testing.assert_allclose(survey.theta[-1], -2 * BEND_ANGLE, rtol=0, atol=1e-13)
    for bend in ('b1', 'b2'):
        elements = [line[name] for name in line.element_names if name.startswith(bend)]
        np.testing.assert_allclose(sum(el.length for el in elements), BEND_LENGTH,
                                   rtol=0, atol=1e-14)
        np.testing.assert_allclose(sum(el.angle for el in elements), BEND_ANGLE,
                                   rtol=0, atol=1e-14)
        np.testing.assert_allclose(sum(el.get_total_knl_ksl()[0][0] for el in elements), BEND_ANGLE,
                                   rtol=0, atol=1e-14)
    np.testing.assert_allclose(survey.s[-1], 2 * BEND_LENGTH + GAP, rtol=0, atol=1e-14)

    survey_error = np.linalg.norm(sliced_survey.XYZ[indices] - survey.XYZ, axis=1)
    trajectory_error = np.linalg.norm(sliced_trajectory[indices] - trajectory, axis=1)
    print(f'Total survey deflection: {-np.rad2deg(survey.theta[-1]):.12g} deg')
    print(f'Total length per bend including fringes: {BEND_LENGTH:g} m; gap: {GAP:g} m')
    print(f'Flat body: {BODY_LENGTH:g} m; curved length: {CURVED_LENGTH:g} m; h: {H:g} 1/m')
    print(f'Fringe length per edge: {FRINGE_LENGTH:g} m; line length: {survey.s[-1]:g} m')
    print('Particle: x0=3 mm, y0=2 mm, px0=2e-4, py0=1e-4, delta=1%')
    print(f'Field slices: {10 * args.slices}; RK4 steps per slice: 20')
    print(f'Maximum survey position difference: {survey_error.max():.3e} m')
    print(f'Maximum survey frame difference: '
          f'{np.max(np.abs(sliced_survey.E_matrix[indices] - survey.E_matrix)):.3e}')
    print(f'Maximum particle lab-position difference: {trajectory_error.max():.3e} m')
    print('Unsliced and sliced surveys also match the two nominal sector bends.')

    if not args.no_plot or args.save_plot:
        fig = plot_comparison(line, survey, trajectory, sliced_survey,
                              sliced_trajectory, survey_error, trajectory_error)
        if args.save_plot:
            fig.savefig(args.save_plot, dpi=160)
            print(f'Saved {args.save_plot}')
        if not args.no_plot:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == '__main__':
    main()
