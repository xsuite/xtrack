"""Shared convergence study for the solenoid and curved-dipole examples.

Fit every nonzero a_i(s), b_i(s), bs(s) independently with C1 cubic Hermite
pieces. Pad the input arrays with one zero coefficient: FieldExpansion
must store the integral of bs, including when bs itself is cubic.

Use DOP853 on the Lorentz equations in the same Frenet frame as tracking,
including q=1+h*x and the h*p_s curvature term. The reference uses the
smooth element's get_field, so it checks integration and segmentation, not
the field-construction implementation independently. Check refinement of
both the reference integration and the transverse expansion order ny.
The straight-solenoid entry point additionally checks its analytical field.

False preserves canonical momentum at internal interfaces; True preserves
kinetic momentum. Both use classical RK4 inside each element, so even False
has a finite-step integration defect that should decrease on refinement.

Measure native canonical 6D Jacobians with fourth-order Richardson finite
differences at h_fd, h_fd/2, h_fd/4. The difference between the two estimates
gives a differentiation-resolution indicator (not a rigorous error bound).
Propagate it to the symplectic defect using 2*||M||*||dM||+||dM||^2, and
repeat with twice as many integration steps. Unlike the dipole-specific
analytic tangent map, this applies to all three geometries without a second
implementation of their tracking equations. Unresolved defects are marked
explicitly; a small determinant error alone does not establish symplecticity.

Do not transfer the straight dipole's analytically predicted limiting fields
to these cases. Report errors against each case's own smooth-field reference.
"""

import argparse
from dataclasses import dataclass
from math import factorial
from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicHermiteSpline

import xobjects as xo
import xtrack as xt

from ._pkin_const_symplecticity import defect, finite_difference_check
from .convergence_pkin_const import coordinates, error_per_particle, orders, track_segments


@dataclass
class FieldCase:
    name: str
    length: float
    h: float
    ny: int
    a: list
    b: list
    bs: Polynomial
    profile: Polynomial
    profile_label: str
    field_check: object = None

    @property
    def profiles(self):
        return self.a + self.b + [self.bs]


def round_solenoid_seeds(profile):
    """Mid-plane data of the finite, axisymmetric degree-six solenoid.

    B_r = sum_k (-1)^(k+1) r^(2k+1) bs^(2k+1)/(2^(2k+1) k! (k+1)!).
    a_i is the i-th x derivative of Bx at x=y=0, hence the factorial.
    In a curved frame these seeds instead define a curved Maxwell expansion;
    they do not assert exact axisymmetry or specify a particular coil design.
    """
    a = [Polynomial([0.]) for _ in range(6)]
    for k in range(3):
        a[2*k+1] = ((-1)**(k+1)*factorial(2*k+1)
                    / (2**(2*k+1)*factorial(k)*factorial(k+1)) * profile.deriv(2*k+1))
    return a, [Polynomial([0.])], profile


def make_element(case, profiles, length, context, ny=None):
    # Include one EXTRA storage power; never discard the highest bs term.
    size = max(len(p.coef) for p in profiles) + 1
    coefficients = np.array([np.pad(p.coef, (0, size-len(p.coef))) for p in profiles])
    na, nb = len(case.a), len(case.b)
    return xt.FieldExpansion(
        _context=context, length=length, h=case.h, sstart=0, nstep=1,
        a=coefficients[:na], b=coefficients[na:na+nb], bs=coefficients[-1],
        ny=case.ny if ny is None else ny,
    )


def reference_solution(element, initial, refined=True):
    """Kinetic Lorentz equations, parametrized by design arc length s."""
    start = coordinates(initial)
    beta0, ptau = np.array(initial.beta0), np.array(initial.ptau)

    def rhs(s, flattened):
        x, px, y, py, zeta, delta = flattened.reshape(start.shape)
        q = 1 + element.h*x
        ps = np.sqrt((1 + delta)**2 - px**2 - py**2)
        field = element.get_field(x, y, s)
        bx, by, bs = (field[name] for name in ('Bx', 'By', 'Bs'))
        return np.array([
            q*px/ps, element.h*ps + q*(py/ps*bs - by),
            q*py/ps, q*(bx - px/ps*bs),
            1 - q*(1 + beta0*ptau)/ps, np.zeros_like(delta),
        ]).ravel()

    solution = solve_ivp(
        rhs, (0, element.length), start.ravel(), method='DOP853',
        rtol=3e-12 if refined else 3e-11, atol=3e-14 if refined else 3e-13,
        max_step=element.length/(64 if refined else 32), t_eval=[element.length],
    )
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError(f'Reference integration failed: {solution.message}')
    return solution.y[:, -1].reshape(start.shape)


def measure_symplecticity(elements, central, initial, length, steps, mode, fd_step):
    matrix, sensitivity = finite_difference_check(
        elements, central, initial, steps, mode, length, h=fd_step)
    resolution = (2*np.linalg.norm(matrix, ord=2, axis=(-2, -1))*sensitivity
                  + sensitivity**2)
    return defect(matrix), resolution, np.abs(np.linalg.det(matrix)-1)


def run_case(case, description):
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--max-segments', type=int, default=128)
    parser.add_argument('--steps-per-segment', type=int, default=4)
    parser.add_argument('--symplectic-y-points', type=int, default=11)
    parser.add_argument('--fd-step', type=float, default=1e-3,
                        help='Canonical perturbation in scaled coordinates (default: 1e-3).')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--save-plot', help='Also save a companion <stem>_symplectic figure.')
    args = parser.parse_args()
    if (args.max_segments < 8 or args.max_segments & (args.max_segments-1)
            or args.steps_per_segment < 1 or args.symplectic_y_points < 3
            or not np.isfinite(args.fd_step) or args.fd_step <= 0):
        parser.error('Use a power-of-two max-segments >= 8, positive steps and fd-step,'
                     ' and symplectic-y-points >= 3.')

    context = xo.ContextCpu(omp_num_threads=0)
    context.allow_kernel_compilation = True
    initial_y = np.array([0.002, 0.01, 0.05])
    initial = xt.Particles(
        _context=context, p0c=1e9, q0=1, x=0.003, px=0.004,
        y=initial_y, py=0.1*initial_y, zeta=0, delta=0.02,
    )
    smooth = make_element(case, case.profiles, case.length, context)
    higher_ny = make_element(case, case.profiles, case.length, context, ny=case.ny+2)
    print(f'{case.name}: L={case.length:g} m, h={case.h:g} 1/m, ny={case.ny}', flush=True)
    if case.field_check:
        case.field_check(smooth)
        print('Analytical field check passed.')
    reference = reference_solution(smooth, initial)
    refinement = np.max(error_per_particle(
        reference, reference_solution(smooth, initial, refined=False), case.length))
    transverse_check = np.max(error_per_particle(
        reference, reference_solution(higher_ny, initial), case.length))
    print(f'DOP853 refinement difference: {refinement:.3e}')
    print(f'Reference change for ny={case.ny+2}: {transverse_check:.3e}')
    print('Reference is numerical and uses the declared finite transverse expansion.')
    reference_scale = max(refinement, transverse_check, 1e-14)

    integration_counts = 2**np.arange(1, 10)
    integration_errors = {}
    for mode in (False, True):
        integration_errors[mode] = np.array([
            np.max(error_per_particle(track_segments([smooth], initial, mode, int(n)),
                                      reference, case.length)) for n in integration_counts
        ])
    print('\nSmooth segment integration convergence')
    print(' steps      False error  order       True error   order')
    for i, n in enumerate(integration_counts):
        entries = []
        for mode in (False, True):
            rate = f'{orders(integration_errors[mode], integration_counts)[i-1]:5.2f}' if i else '    -'
            entries.append(f'{integration_errors[mode][i]:.3e}  {rate}')
        print(f'{n:6d}    ' + '     '.join(entries))
    # Same polynomial, different element boundaries: no cubic fitting error.
    exact = [make_element(case, [p(Polynomial([s, 1])) for p in case.profiles],
                          case.length/8, context)
             for s in np.linspace(0, case.length, 9)[:-1]]
    for mode in (False, True):
        control = track_segments(exact, initial, mode, 64)
        err = np.max(error_per_particle(control, reference, case.length))
        print(f'Exact polynomial split control, pkin_const={mode}: {err:.3e}')
        if err > max(1e-9, 100*reference_scale):
            raise RuntimeError('Exact split disagrees with the smooth-field reference.')

    counts = 2**np.arange(int(np.log2(args.max_segments))+1)
    errors = {mode: [] for mode in (False, True)}
    rk_checks = {mode: [] for mode in (False, True)}
    symplectic = {mode: {key: [] for key in ('coarse', 'fine', 'resolution', 'det')}
                  for mode in (False, True)}
    symplectic_y = np.linspace(-0.05, 0.05, args.symplectic_y_points)
    centers = np.zeros((len(symplectic_y), 6))
    centers[:, 0], centers[:, 1] = initial.x[0], initial.kin_px[0]
    centers[:, 2], centers[:, 3] = symplectic_y, 0.1*symplectic_y
    centers[:, 5] = initial.ptau[0]
    profile_index = next(i for i, p in enumerate(case.profiles) if p is case.profile)
    fitted_profiles = {}
    fit_errors = []

    print('\nCubic refinement: maximum exit error and symplectic defect over y')
    print(' N      err False    err True      sympl False  resolution   sympl True   resolution')
    for count in counts:
        boundaries = np.linspace(0, case.length, count+1)
        splines = [CubicHermiteSpline(boundaries, p(boundaries), p.deriv()(boundaries))
                   for p in case.profiles]
        segments = [make_element(
            case, [Polynomial(spline.c[::-1, i]) for spline in splines], right-left, context)
            for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:]))]
        sample_s = (boundaries[:-1, None]
                    + np.linspace(0, 1, 9)[None, :]*case.length/count).ravel()
        fit_errors.append(np.max(np.abs(splines[profile_index](sample_s)-case.profile(sample_s))))
        if count in (2, 4, 8):
            fitted_profiles[count] = splines[profile_index]
        canonical = centers.copy()
        entrance = segments[0].get_field(canonical[:, 0], canonical[:, 2], 0)
        canonical[:, 1] += entrance['Ax']
        canonical[:, 3] += entrance['Ay']
        for mode in (False, True):
            coarse = track_segments(segments, initial, mode, args.steps_per_segment)
            fine = track_segments(segments, initial, mode, 2*args.steps_per_segment)
            errors[mode].append(error_per_particle(fine, reference, case.length))
            rk_checks[mode].append(np.max(error_per_particle(fine, coarse, case.length)))
            for label, steps in (('coarse', args.steps_per_segment),
                                 ('fine', 2*args.steps_per_segment)):
                value, resolution, determinant = measure_symplecticity(
                    segments, canonical, initial, case.length, steps, mode, args.fd_step)
                symplectic[mode][label].append(value)
            symplectic[mode]['resolution'].append(resolution)
            symplectic[mode]['det'].append(determinant)
        print(f'{count:4d}   {np.max(errors[False][-1]):.3e}   {np.max(errors[True][-1]):.3e}'
              + ''.join(f'   {np.max(symplectic[mode][key][-1]):.3e}'
                        for mode in (False, True) for key in ('fine', 'resolution')), flush=True)

    for mode in (False, True):
        errors[mode] = np.array(errors[mode])
        for key in symplectic[mode]:
            symplectic[mode][key] = np.array(symplectic[mode][key])
    print('\nObserved cubic-refinement orders against the smooth reference')
    print(' N       False        True')
    for i, n in enumerate(counts[1:]):
        print(f'{n:4d}    {orders(errors[False].max(axis=1), counts)[i]:8.3f}'
              f'    {orders(errors[True].max(axis=1), counts)[i]:8.3f}')
    print(f'Last on-axis profile fit order: {orders(np.array(fit_errors), counts)[-1]:.2f}')
    for mode in (False, True):
        print(f'pkin_const={mode}: largest RK4 refinement change '
              f'{max(rk_checks[mode]):.3e}; finest-grid max |det M-1| '
              f'{np.max(symplectic[mode]["det"][-1]):.3e}')
    print('Exit error: max difference in (x/L,kin_px,y/L,kin_py,zeta/L,delta).')
    print('Symplectic defect: ||M.T S M-S||_2 in canonical (x/L,px,y/L,py,tau/L,ptau).')
    print('Resolution is a finite-difference sensitivity indicator, not a rigorous bound.')
    print('Plateaus and negative local orders can reflect field approximation, truncation,')
    print('roundoff or cancellation; cubic profile accuracy alone does not ensure field accuracy.')

    data = dict(counts=counts, errors=errors, rk_checks=rk_checks, reference_scale=reference_scale,
                integration_counts=integration_counts, integration_errors=integration_errors,
                initial_y=initial_y, symplectic_y=symplectic_y, symplectic=symplectic,
                fitted_profiles=fitted_profiles)
    if not args.no_plot or args.save_plot:
        plot_case(case, data, args)
    return data


def plot_case(case, data, args):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    s = np.linspace(0, case.length, 501)
    axes[0, 0].plot(s, case.profile(s), 'k', label='Degree six')
    for n, spline in data['fitted_profiles'].items():
        axes[0, 0].plot(s, spline(s), '--', label=f'{n} cubics')
    axes[0, 0].set(xlabel='s [m]', ylabel=case.profile_label, title=case.name)
    axes[0, 0].legend()
    for mode, style in ((False, 'o-'), (True, 'x--')):
        axes[0, 1].loglog(data['integration_counts'], data['integration_errors'][mode],
                          style, label=f'pkin_const={mode}')
    axes[0, 1].axhline(data['reference_scale'], color='0.5', linestyle=':',
                       label='Reference / ny refinement scale')
    axes[0, 1].set(xlabel='RK4 steps', ylabel='Maximum normalized exit error',
                   title='Smooth degree-six field')
    axes[0, 1].legend(fontsize=8)
    for column, mode in enumerate((False, True)):
        for j, y in enumerate(data['initial_y']):
            axes[1, column].loglog(data['counts'], data['errors'][mode][:, j], '.-',
                                   label=f'y0={y*1e3:g} mm')
        axes[1, column].set(xlabel='Number of cubic segments', ylabel='Normalized exit error',
                            title=f'pkin_const={mode}')
        axes[1, column].legend()
    axes[1, 1].sharey(axes[1, 0])
    for ax in axes.flat:
        ax.grid(True, which='both', alpha=0.25)

    sym_fig, sym_axes = plt.subplots(2, 2, figsize=(12, 9), layout='constrained')
    sym = data['symplectic']
    vmax = max(np.max(sym[mode]['fine']) for mode in (False, True))
    norm = LogNorm(1e-12, max(vmax, 1e-11))
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad('0.85')
    for column, mode in enumerate((False, True)):
        values = np.ma.masked_where(sym[mode]['fine'] <= sym[mode]['resolution'], sym[mode]['fine'])
        mesh = sym_axes[0, column].pcolormesh(
            data['counts'], data['symplectic_y']*1e3, values.T,
            shading='nearest', norm=norm, cmap=cmap)
        sym_axes[0, column].set(xscale='log', xlabel='Number of cubic segments',
                                ylabel='Initial y [mm]', title=f'pkin_const={mode}')
        ax = sym_axes[1, column]
        for key, style in [('coarse', '--'), ('fine', '-')]:
            ax.semilogy(data['symplectic_y']*1e3, np.maximum(sym[mode][key][-1], 1e-15),
                         style, label=f'{key} RK4')
        ax.semilogy(data['symplectic_y']*1e3, np.maximum(sym[mode]['resolution'][-1], 1e-15),
                     ':', color='k', label='FD resolution indicator')
        ax.set(xlabel='Initial y [mm]', ylabel=r'$\|M^T S M-S\|_2$',
               title=f'{data["counts"][-1]} cubic segments')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.25)
    sym_fig.colorbar(mesh, ax=list(sym_axes[0]), label=r'$\|M^T S M-S\|_2$')
    sym_fig.suptitle(case.name + ': canonical 6D symplecticity\nGrey cells: below FD resolution')
    if args.save_plot:
        path = Path(args.save_plot)
        fig.savefig(path, dpi=150)
        sym_fig.savefig(path.with_name(path.stem + '_symplectic' + path.suffix), dpi=150)
    if not args.no_plot:
        plt.show()
