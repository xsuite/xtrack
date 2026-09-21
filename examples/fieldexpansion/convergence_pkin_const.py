"""Investigate FieldExpansion convergence with and without pkin_const.

Run from the repository root::

    python -m examples.fieldexpansion.convergence_pkin_const
    python -m examples.fieldexpansion.convergence_pkin_const --no-plot

One smooth normal dipole has b(s) of degree six, a=bs=0, and length 1 m.
Its exact off-axis field is a finite Maxwell expansion (ny=7 is sufficient).
Compare an unsplit FieldExpansion and lists of C1 cubic Hermite segments
against an independently integrated DOP853 Lorentz-force reference. Cubics
match b and b' at each boundary and use LOCAL longitudinal coordinates.
Double the RK4 steps per segment to distinguish integration error from
the error caused by replacing the smooth magnet with cubic segments.

Initial conditions always describe the SAME kinetic momenta. For False,
convert them to the first element's canonical momenta at the entrance;
compare kin_px/kin_py at the exit. Keep the requested boundary handling
inside the list: no manual momentum corrections between elements.

The result need not be convergence to the full field at fixed nonzero y!
Writing b^(j) for the j-th s derivative, the smooth dipole has

    By = b - b'' y^2/2 + b^(4) y^4/24 - b^(6) y^6/720,
    Bs = b' y - b''' y^3/6 + b^(5) y^5/120,
    Ax = -b' y^2/2 + b''' y^4/24 - b^(5) y^6/720, Ay=0, As=-b*x.

Cubic segments omit derivatives >=4 inside each interval. With True,
their limiting regular field has By through y^2 and Bs through y^3.
With False, the jumps of Ax also act on kinetic momentum. Their limiting
potential is Ax=-b' y^2/2+b''' y^4/24, As=-b*x, whose field additionally
contains b^(4) y^4/24 in By. Both still omit part of the full sixth-order
field. DOP853 solutions of these two limits expose the resulting floors.
Agreement of b(s) alone therefore does not establish off-axis convergence.

False preserves canonical momentum at interfaces (the symplectic boundary
choice). Both modes currently use classical RK4 inside FieldExpansion;
neither finite-step numerical map is exactly symplectic.

Also measure ||M.T S M - S||_2 on a grid of initial y and segment counts.
Here M is the full 6D Jacobian in canonical (x/L, px, y/L, py, tau/L, ptau),
tau=zeta/beta0. Canonical momenta refer to the first/last segment's gauge:
remove the external kinetic-momentum conversions from BOTH modes before
testing symplecticity, while retaining their different internal interfaces.
Using kinetic momenta or (zeta,delta) with standard S would give a false
defect. The reported norm depends on the stated coordinate scaling.

The dipole-specific helper differentiates the SAME RK4 stages and interface
maps analytically (variational equations). This retains RK4's numerical
defect and avoids finite-difference cancellation. Compare trajectories to
native tracking at every grid point, and independently audit Jacobians
with fourth-order Richardson finite differences at three perturbation
widths for N=1,8,max. Repeat with twice as many RK4 steps. The extra figure
shows both modes on the same color scale, plus y/N cuts and the finite-
difference noise. Defects below 1e-14 are clipped for display, not claimed
to be zero. If --save-plot is used, also save <stem>_symplectic<suffix>.
"""

import argparse
from math import factorial
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline

import xobjects as xo
import xtrack as xt

from ._pkin_const_symplecticity import (
    defect, finite_difference_check, native_map, scaled_jacobian, tangent_map,
)


def dipole_field(derivatives, y, s, by_terms=4, bs_terms=3):
    """Normalized fields B/(B rho); exact for the degree-six dipole."""
    by = sum((-1)**k * y**(2*k) / factorial(2*k) * derivatives[2*k](s)
             for k in range(by_terms))
    bs = sum((-1)**k * y**(2*k+1) / factorial(2*k+1) * derivatives[2*k+1](s)
             for k in range(bs_terms))
    return by, bs


def coordinates(particles):
    values = np.array([getattr(particles, name) for name in
                       ('x', 'kin_px', 'y', 'kin_py', 'zeta', 'delta')])
    if not np.all(particles.state > 0) or not np.all(np.isfinite(values)):
        raise RuntimeError('Tracking produced lost particles or non-finite coordinates.')
    return values


def error_per_particle(values, reference, length):
    # Absolute dimensionless error, maximized over six exit coordinates.
    scales = np.array([length, 1, length, 1, length, 1])[:, None]
    return np.max(np.abs((values - reference) / scales), axis=0)


def solve_reference(derivatives, initial, length, by_terms=4, bs_terms=3,
                    refined=True):
    start = coordinates(initial)
    beta0, ptau = np.array(initial.beta0), np.array(initial.ptau)

    def rhs(s, flattened):
        x, px, y, py, zeta, delta = flattened.reshape(start.shape)
        ps = np.sqrt((1 + delta)**2 - px**2 - py**2)
        by, bs = dipole_field(derivatives, y, s, by_terms, bs_terms)
        return np.array([
            px / ps, py / ps * bs - by,
            py / ps, -px / ps * bs,
            1 - (1 + beta0 * ptau) / ps, np.zeros_like(delta),
        ]).ravel()

    solution = solve_ivp(
        rhs, (0, length), start.ravel(), method='DOP853',
        rtol=3e-14 if refined else 2e-12,
        atol=3e-16 if refined else 2e-14,
        max_step=length / (64 if refined else 32), t_eval=[length],
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y[:, -1].reshape(start.shape)


def make_element(coefficients, length, context):
    coefficients = np.asarray(coefficients)
    return xt.FieldExpansion(
        _context=context, length=length, sstart=0, nstep=1,
        a=np.zeros((1, len(coefficients))), b=coefficients[None, :],
        bs=np.zeros(len(coefficients)), ny=7,
    )


def track_segments(elements, initial, pkin_const, steps):
    particles = initial.copy()
    if not pkin_const:
        entrance = elements[0].get_field(particles.x, particles.y, s=0)
        particles.px += entrance['Ax']
        particles.py += entrance['Ay']
        particles.ax = entrance['Ax']
        particles.ay = entrance['Ay']
    for element in elements:
        element.pkin_const = pkin_const
        element.nstep = steps
        element.ds = element.length / steps  # Stored separately from nstep.
        element.track(particles)
    np.testing.assert_allclose(
        particles.s, sum(el.length for el in elements), rtol=0, atol=1e-11)
    return coordinates(particles)


def orders(errors, counts):
    """Local observed order; can be zero/negative near floors or cancellation."""
    return np.log(errors[:-1] / errors[1:]) / np.log(counts[1:] / counts[:-1])


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max-segments', type=int, default=1024,
                        help='Power of two, at least 8 (default: 1024).')
    parser.add_argument('--steps-per-segment', type=int, default=8,
                        help='Compare this RK4 step count with twice as many (default: 8).')
    parser.add_argument('--symplectic-y-points', type=int, default=21,
                        help='Number of offsets in [-50, 50] mm (default: 21).')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--save-plot', help='Save the figure, e.g. /tmp/convergence.png.')
    args = parser.parse_args()
    if (args.max_segments < 8 or args.max_segments & (args.max_segments - 1)
            or args.steps_per_segment < 1 or args.symplectic_y_points < 3):
        parser.error('Use a power-of-two max-segments >= 8, positive steps-per-segment,'
                     ' and symplectic-y-points >= 3.')

    context = xo.ContextCpu(omp_num_threads=0)
    context.allow_kernel_compilation = True
    length = 1.0
    # b(s) = 0.05 + 0.10 * 64*u^3*(1-u)^3 [1/m], u=s/length.
    # Only b[0] is nonzero; it is the normal dipole component B_y(x,0,s)/(B rho).
    b = Polynomial([0.05, 0, 0, 6.4, -19.2, 19.2, -6.4])(Polynomial([0, 1/length]))
    derivatives = [b.deriv(k) for k in range(7)]
    y_initial = np.array([0.002, 0.01, 0.05])
    initial = xt.Particles(
        _context=context, p0c=1e9, q0=1, x=0.003, px=0.004,
        y=y_initial, py=0.1*y_initial, zeta=0, delta=0.02,
    )
    smooth = make_element(b.coef, length, context)

    # Check that the independent analytical field is the field actually used.
    y_grid, s_grid = np.meshgrid(np.linspace(-0.08, 0.08, 7),
                                  np.linspace(0, length, 31), indexing='ij')
    field = smooth.get_field(x=0.003, y=y_grid, s=s_grid)
    by, bs = dipole_field(derivatives, y_grid, s_grid)
    np.testing.assert_allclose(field['Bx'], 0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(field['By'], by, rtol=0, atol=2e-13)
    np.testing.assert_allclose(field['Bs'], bs, rtol=0, atol=2e-13)

    reference = solve_reference(derivatives, initial, length)
    reference_check = solve_reference(derivatives, initial, length, refined=False)
    reference_error = np.max(error_per_particle(reference, reference_check, length))
    print(f'DOP853 reference refinement difference: {reference_error:.3e}')
    if reference_error > 1e-13:
        raise RuntimeError('The reference needs tighter integration settings.')

    # First isolate the integration error with the exact degree-six element.
    integration_counts = 2**np.arange(1, 10)
    integration_errors = {}
    print('\nUnsplit degree-six segment: maximum error over all particles')
    print(' steps       False       order          True       order')
    for mode in (False, True):
        integration_errors[mode] = np.array([
            np.max(error_per_particle(
                track_segments([smooth], initial, mode, int(n)), reference, length))
            for n in integration_counts
        ])
    for i, n in enumerate(integration_counts):
        entries = []
        for mode in (False, True):
            rate = '    -'
            if i:
                rate = f'{orders(integration_errors[mode], integration_counts)[i-1]:5.2f}'
            entries.append(f'{integration_errors[mode][i]:.3e}   {rate}')
        print(f'{n:6d}   ' + '     '.join(entries))

    # Control: splitting the EXACT polynomial introduces no model change.
    exact_segments = [
        make_element(b(Polynomial([left, 1])).coef, length/8, context)
        for left in np.linspace(0, length, 9)[:-1]
    ]
    for mode in (False, True):
        control = track_segments(exact_segments, initial, mode, 64)
        control_error = np.max(error_per_particle(control, reference, length))
        print(f'Exact degree-six split control, pkin_const={mode}: {control_error:.3e}')
        np.testing.assert_allclose(control, reference, rtol=0, atol=2e-11)

    # Integrate the analytically predicted limits of the cubic sequences.
    limits = {
        False: solve_reference(derivatives, initial, length, by_terms=3, bs_terms=2),
        True: solve_reference(derivatives, initial, length, by_terms=2, bs_terms=2),
    }
    floors = {mode: error_per_particle(limit, reference, length)
              for mode, limit in limits.items()}
    counts = 2**np.arange(int(np.log2(args.max_segments)) + 1)
    errors = {mode: [] for mode in (False, True)}
    limit_errors = {mode: [] for mode in (False, True)}
    integration_checks = {mode: [] for mode in (False, True)}
    fit_errors = []
    field_errors = []
    interpolation_examples = {}
    symplectic_y = np.linspace(-0.05, 0.05, args.symplectic_y_points)
    symplectic_initial = np.zeros((len(symplectic_y), 6))
    symplectic_initial[:, 0] = initial.x[0]
    symplectic_initial[:, 1] = initial.kin_px[0]
    symplectic_initial[:, 2] = symplectic_y
    symplectic_initial[:, 3] = 0.1*symplectic_y
    symplectic_initial[:, 5] = initial.ptau[0]
    symplectic = {mode: {'coarse': [], 'fine': [], 'det': []} for mode in (False, True)}
    fd_checks = {mode: {} for mode in (False, True)}
    max_native_difference = 0.
    print('\nComputing analytic tangent maps and native tracking checks...')

    for count in counts:
        boundaries = np.linspace(0, length, count + 1)
        spline = CubicHermiteSpline(boundaries, b(boundaries), derivatives[1](boundaries))
        # scipy stores descending powers of s-boundaries[i]. FieldExpansion
        # takes ascending powers, evaluated from sstart=0 in EACH element.
        cubic_segments = [make_element(c, right-left, context)
                          for c, left, right in zip(
                              spline.c[::-1].T, boundaries[:-1], boundaries[1:])]
        samples = (boundaries[:-1, None]
                   + np.linspace(0, 1, 9)[None, :] * length/count).ravel()
        fit_errors.append(np.max(np.abs(spline(samples) - b(samples))))
        cubic_by = spline(samples) - spline(samples, 2)*y_initial[-1]**2/2
        cubic_bs = spline(samples, 1)*y_initial[-1] - spline(samples, 3)*y_initial[-1]**3/6
        exact_by, exact_bs = dipole_field(derivatives, y_initial[-1], samples)
        field_errors.append(max(np.max(np.abs(cubic_by - exact_by)),
                                np.max(np.abs(cubic_bs - exact_bs))))
        if count in (2, 4, 8):
            interpolation_examples[count] = spline
        # Same physical central trajectories for both modes. Perturbations
        # used for Jacobians are then independent CANONICAL perturbations.
        canonical_initial = symplectic_initial.copy()
        canonical_initial[:, 1] += cubic_segments[0].get_field(
            canonical_initial[:, 0], canonical_initial[:, 2], s=0)['Ax']
        for mode in (False, True):
            coarse = track_segments(cubic_segments, initial, mode, args.steps_per_segment)
            fine = track_segments(cubic_segments, initial, mode, 2*args.steps_per_segment)
            errors[mode].append(error_per_particle(fine, reference, length))
            limit_errors[mode].append(error_per_particle(fine, limits[mode], length))
            integration_checks[mode].append(np.max(error_per_particle(fine, coarse, length)))
            for label, nsteps in [('coarse', args.steps_per_segment),
                                  ('fine', 2*args.steps_per_segment)]:
                tangent_exit, matrix = tangent_map(
                    spline.c[::-1].T, np.diff(boundaries), canonical_initial,
                    initial.beta0[0], nsteps, mode)
                matrix = scaled_jacobian(matrix, length)
                symplectic[mode][label].append(defect(matrix))
            symplectic[mode]['det'].append(np.abs(np.linalg.det(matrix) - 1))
            native_exit = native_map(cubic_segments, canonical_initial, initial,
                                     2*args.steps_per_segment, mode)
            max_native_difference = max(max_native_difference,
                                         np.max(np.abs(native_exit - tangent_exit)))
            np.testing.assert_allclose(tangent_exit, native_exit, rtol=0, atol=2e-12)
            if count in (1, 8, args.max_segments):
                fd_matrix, sensitivity = finite_difference_check(
                    cubic_segments, canonical_initial, initial,
                    2*args.steps_per_segment, mode, length)
                difference = np.linalg.norm(matrix - fd_matrix, ord=2, axis=(-2, -1))
                if np.any(difference > np.maximum(2e-11, 8*sensitivity)):
                    raise RuntimeError('Analytic tangent map failed the native Jacobian check.')
                fd_checks[mode][count] = dict(
                    defect=defect(fd_matrix), sensitivity=sensitivity, difference=difference,
                )
        print(f'  N={count:4d}: max symplectic defect (refined RK4) '
              f'False={np.max(symplectic[False]["fine"][-1]):.3e}, '
              f'True={np.max(symplectic[True]["fine"][-1]):.3e}', flush=True)

    print('\nCubic segments: errors below use the refined RK4 step count')
    print(' N    max|b_fit-b|    max|B_fit-B|    err False   order'
          '    err True    order   RK4 check')
    for mode in (False, True):
        errors[mode] = np.array(errors[mode])
        limit_errors[mode] = np.array(limit_errors[mode])
    for i, count in enumerate(counts):
        entries = []
        for mode in (False, True):
            maximum = errors[mode].max(axis=1)
            rate = f'{orders(maximum, counts)[i-1]:5.2f}' if i else '    -'
            entries.append(f'{maximum[i]:.3e}  {rate}')
        check = max(integration_checks[mode][i] for mode in (False, True))
        print(f'{count:4d}   {fit_errors[i]:.3e}      {field_errors[i]:.3e}     '
              + '   '.join(entries) + f'   {check:.2e}')
    print('Field errors are sampled in B/(B rho) [1/m] at y=50 mm,')
    print('using the regular field inside cubics (excluding boundary impulses).')
    print(f'Last observed order of the on-axis b(s) fit: '
          f'{orders(np.array(fit_errors), counts)[-1]:.2f}')
    print('Exit errors: max |difference| in (x/L, kin_px, y/L, kin_py, zeta/L, delta).')
    print('RK4 check: maximum change after doubling steps per cubic segment.')
    relative_integration_change = max(
        np.max(np.array(integration_checks[mode]) / errors[mode].max(axis=1))
        for mode in (False, True)
    )
    print(f'Largest RK4 check / exit error over the scan: {relative_integration_change:.3e}')
    print('\n y0 [mm]   pkin_const   exit error at max N    predicted error floor')
    for j, y in enumerate(y_initial):
        for mode in (False, True):
            print(f'{y*1e3:8.1f}      {str(mode):5s}          '
                  f'{errors[mode][-1, j]:.3e}             {floors[mode][j]:.3e}')
    print('\nLast observed orders towards each cubic sequence\'s OWN limit (y0=50 mm):')
    for mode in (False, True):
        print(f'  pkin_const={mode}: {orders(limit_errors[mode][:, -1], counts)[-1]:.2f}')
    print('These limits differ from the full smooth field; small exit errors near a floor')
    print('can result from cancellation. Consult the per-offset curves and RK4 check.')

    print('\nSymplectic defect ||M.T S M-S||_2 in (x/L,px,y/L,py,tau/L,ptau)')
    print(f'Maximum over {len(symplectic_y)} y offsets in [-50,50] mm; '
          f'coarse/fine = {args.steps_per_segment}/{2*args.steps_per_segment} RK4 steps/segment')
    print(' N       False coarse    False fine      True coarse     True fine      |det M-1| True')
    for mode in (False, True):
        for key in symplectic[mode]:
            symplectic[mode][key] = np.array(symplectic[mode][key])
    for i, count in enumerate(counts):
        values = [np.max(symplectic[mode][key][i])
                  for mode in (False, True) for key in ('coarse', 'fine')]
        print(f'{count:4d}     ' + '      '.join(f'{value:.3e}' for value in values)
              + f'      {np.max(symplectic[True]["det"][i]):.3e}')
    print(f'Maximum tangent/native exit-coordinate difference: {max_native_difference:.3e}')
    for mode in (False, True):
        difference = max(np.max(check['difference']) for check in fd_checks[mode].values())
        sensitivity = max(np.max(check['sensitivity']) for check in fd_checks[mode].values())
        print(f'pkin_const={mode}: max native FD Jacobian difference={difference:.3e}; '
              f'FD step-size sensitivity={sensitivity:.3e}')
    print('FD audit: Richardson central differences at h=1e-3, h/2, h/4; N=1,8,max.')
    print('Near-roundoff defects are unresolved; the determinant alone misses interface defects.')

    if not args.no_plot or args.save_plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        s_plot = np.linspace(0, length, 501)
        axes[0, 0].plot(s_plot, b(s_plot), 'k', label='Smooth degree six')
        for count, spline in interpolation_examples.items():
            axes[0, 0].plot(s_plot, spline(s_plot), '--', label=f'{count} cubics')
        axes[0, 0].set(xlabel='s [m]', ylabel='b(s) [1/m]', title='On-axis dipole profile')
        axes[0, 0].legend()
        for mode, marker in ((False, 'o-'), (True, 'x--')):
            axes[0, 1].loglog(integration_counts, integration_errors[mode], marker,
                              label=f'pkin_const={mode}')
        axes[0, 1].loglog(integration_counts,
                          integration_errors[False][3]*(integration_counts/16.0)**-4,
                          ':', color='0.5', label='Fourth order')
        axes[0, 1].set(xlabel='RK4 steps', ylabel='Maximum normalized exit error',
                       title='Unsplit degree-six segment')
        axes[0, 1].legend()
        for column, mode in enumerate((False, True)):
            ax = axes[1, column]
            for j, y in enumerate(y_initial):
                ax.loglog(counts, errors[mode][:, j], '.-', color=f'C{j}',
                          label=f'y0={y*1e3:g} mm')
                ax.axhline(floors[mode][j], color=f'C{j}', linestyle='--', alpha=0.7)
            ax.plot([], [], 'k--', label='Predicted floors')
            ax.set(xlabel='Number of cubic segments', ylabel='Normalized exit error',
                   title=f'Cubic approximation, pkin_const={mode}')
            ax.legend()
        # Keep the same error scale for the two boundary treatments.
        axes[1, 1].sharey(axes[1, 0])
        for ax in axes.flat:
            ax.grid(True, which='both', alpha=0.25)
        fig.tight_layout()
        if args.save_plot:
            fig.savefig(args.save_plot, dpi=150)
        plot_symplecticity(counts, symplectic_y, symplectic, fd_checks, args)
        if not args.no_plot:
            plt.show()
    return dict(counts=counts, errors=errors, floors=floors,
                limit_errors=limit_errors, integration_checks=integration_checks,
                symplectic_y=symplectic_y, symplectic=symplectic, fd_checks=fd_checks)


def plot_symplecticity(counts, y, results, fd_checks, args):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    display_floor = 1e-14
    vmax = max(np.max(results[mode]['fine']) for mode in (False, True))
    norm = LogNorm(display_floor, max(vmax, 10*display_floor))
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    for column, mode in enumerate((False, True)):
        mesh = axes[0, column].pcolormesh(
            counts, 1e3*y, np.maximum(results[mode]['fine'].T, display_floor),
            shading='nearest', norm=norm, cmap='viridis')
        axes[0, column].set(xscale='log', xlabel='Number of cubic segments',
                            ylabel='Initial y [mm]', title=f'pkin_const={mode}')
    fig.colorbar(mesh, ax=list(axes[0]), label=r'$\|M^T S M-S\|_2$')
    for column, mode in enumerate((False, True)):
        ax = axes[1, 0]
        for key, style in [('coarse', '--'), ('fine', '-')]:
            ax.semilogy(y*1e3, np.maximum(results[mode][key][-1], display_floor),
                         style, color=f'C{column}', label=f'{mode}, {key} RK4')
        ax.semilogy(y*1e3, np.maximum(fd_checks[mode][counts[-1]]['defect'], display_floor),
                     'x', color=f'C{column}', label=f'{mode}, native finite differences')
    axes[1, 0].set(xlabel='Initial y [mm]', ylabel=r'$\|M^T S M-S\|_2$',
                   title=f'{counts[-1]} segments: tangent map vs finite differences')
    axes[1, 0].legend(fontsize=8)
    for j, target_y in enumerate((0., 0.01, 0.05)):
        index = np.argmin(np.abs(y - target_y))
        for mode, style in ((False, '--'), (True, '-')):
            axes[1, 1].loglog(
                counts, np.maximum(results[mode]['fine'][:, index], display_floor),
                style, marker='.', color=f'C{j}', label=f'{mode}, y={y[index]*1e3:g} mm')
    axes[1, 1].set(xlabel='Number of cubic segments', ylabel=r'$\|M^T S M-S\|_2$',
                   title='Refined RK4: selected vertical offsets')
    axes[1, 1].legend(fontsize=8)
    for ax in axes[1]:
        ax.axhline(display_floor, color='0.5', linestyle=':')
        ax.grid(True, which='both', alpha=0.25)
    fig.suptitle('Canonical 6D symplecticity; values below 1e-14 clipped for display')
    if args.save_plot:
        path = Path(args.save_plot)
        fig.savefig(path.with_name(path.stem + '_symplectic' + path.suffix), dpi=150)


if __name__ == '__main__':
    main()
