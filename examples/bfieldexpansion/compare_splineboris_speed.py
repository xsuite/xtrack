"""Compare BFieldExpansion and SplineBoris speed at the same exit accuracy.

Run from the repository root, for example::

    python -m examples.bfieldexpansion.compare_splineboris_speed
    python -m examples.bfieldexpansion.compare_splineboris_speed --no-plot

The straight magnet has normal/skew dipole, quadrupole and sextupole
components varying polynomially in local s, plus a longitudinal field.
BFieldExpansion uses RK4; SplineBoris uses a second-order Boris scheme.
Each step count is selected against the same DOP853 Lorentz-force reference,
using the maximum error over a reproducible ensemble in
(x/length, kin_px, y/length, kin_py, zeta/length, delta).
This is an absolute, dimensionless exit-error criterion, not a relative error
in each coordinate or a test of long-term symplecticity.

Only warmed-up, single-pass element tracking on a serial CPU is timed.
Construction, compilation, reference integration and particle copies are
excluded. The timed bunch repeats the calibration ensemble, so the same
accuracy criterion applies to all timed particles.
Measure runtime at every scanned step count to plot error versus tracking
time as well as error versus steps, and highlight the matched-accuracy points.
"""

import argparse
from time import perf_counter

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp

import xobjects as xo
import xtrack as xt


def as_spline(coefficients, length, rigidity):
    """Convert ascending powers of s in normalized units to Spline4 in SI."""
    polynomial = Polynomial(coefficients) * rigidity
    derivative = polynomial.deriv()
    integral = polynomial.integ()
    return xt.Spline4(
        val_start=polynomial(0), der_start=derivative(0),
        val_end=polynomial(length), der_end=derivative(length),
        mean=(integral(length) - integral(0)) / length,
    )


def make_elements(context, length, rigidity):
    # At y=0: Bx/(B rho) = sum_i ksc[i](s) x**i/i!, and similarly
    # By/(B rho) = sum_i knc[i](s) x**i/i!. Both APIs use transverse
    # derivatives: ksc -> bx, knc -> by, with no sign or factorial conversion.
    # Column k multiplies s**k (s in meters, not s/length).
    ksc = np.array([
        [0.04, 0.10, 0.08, -0.06, 0.02],
        [0.12, -0.08, 0.05, 0.02, -0.01],
        [0.30, 0.10, -0.20, 0.05, 0.02],
    ])
    knc = np.array([
        [0.05, 0.04, 0.07, -0.03, 0.01],
        [0.40, -0.10, 0.08, 0.03, -0.02],
        [0.80, 0.20, -0.10, 0.04, 0.03],
    ])
    # Keep one trailing zero: BFieldExpansion stores the integral of ksol in
    # the scalar potential, which needs one more longitudinal power.
    ksol = np.array([0.10, 0.02, -0.03, 0.01, 0.0])
    expansion = xt.BFieldExpansion(
        _context=context, length=length, ksc=ksc, knc=knc, ksol=ksol,
        # ny=7 includes the full potential for this sextupole/quartic case.
        ny=7, nstep=1, sstart=0,
        # Preserve kinetic momentum across the entrance/exit gauge changes.
        # With this option, both elements accept and return kinetic px, py.
        pkin_const=True,
    )
    boris = xt.SplineBoris(
        _context=context, length=length, n_steps=1,
        bx=tuple(as_spline(row, length, rigidity) for row in ksc),
        by=tuple(as_spline(row, length, rigidity) for row in knc),
        bs=as_spline(ksol, length, rigidity), radiation_flag=0,
    )
    return {'BFieldExpansion': expansion, 'SplineBoris': boris}


def coordinates(particles):
    if not np.all(particles.state > 0):
        raise RuntimeError('A particle was lost.')
    values = np.array([getattr(particles, name) for name in
                       ('x', 'kin_px', 'y', 'kin_py', 'zeta', 'delta')])
    if not np.all(np.isfinite(values)):
        raise RuntimeError('Non-finite particle coordinates.')
    return values


def error(values, reference, length):
    scales = np.array([length, 1, length, 1, length, 1])[:, None]
    return np.max(np.abs((values - reference) / scales))


def reference_solution(boris, particles, rigidity, tolerance):
    initial = coordinates(particles)
    beta0 = particles.beta0[0]
    ptau = np.array(particles.ptau)

    def rhs(s, flattened):
        x, px, y, py, zeta, delta = flattened.reshape(initial.shape)
        ps = np.sqrt((1 + delta)**2 - px**2 - py**2)
        bx, by, ksol = np.asarray(boris.get_field(x, y, s)) / rigidity
        return np.array([
            px / ps, py / ps * ksol - by,
            py / ps, bx - px / ps * ksol,
            1 - (1 + beta0 * ptau) / ps, np.zeros_like(delta),
        ]).ravel()

    references = []
    for rtol, atol, max_step in [
        (2e-12, 2e-14, boris.length / 8),
        (3e-14, 3e-16, boris.length / 32),
    ]:
        solution = solve_ivp(
            rhs, (0, boris.length), initial.ravel(), method='DOP853',
            rtol=rtol, atol=atol, max_step=max_step, t_eval=[boris.length],
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        references.append(solution.y[:, -1].reshape(initial.shape))
    difference = error(*references, boris.length)
    print(f'Reference refinement difference: {difference:.3e}')
    if difference > tolerance / 100:
        raise RuntimeError('Reference is insufficiently resolved for this tolerance.')
    return references[-1]


def set_steps(element, steps):
    if isinstance(element, xt.SplineBoris):
        element.n_steps = steps
    else:
        element.nstep = steps


def calibrate(element, particles, reference, tolerance, max_steps):
    history = {}

    def measure(steps):
        if steps not in history:
            set_steps(element, steps)
            tracked = particles.copy()
            element.track(tracked)
            np.testing.assert_allclose(tracked.s, element.length, rtol=0, atol=1e-11)
            history[steps] = error(coordinates(tracked), reference, element.length)
        return history[steps]

    # Bracket the tolerance by doubling. Keep these points for the order plot.
    low, high = 0, 1
    scan = []
    while True:
        measured = measure(high)
        scan.append((high, measured))
        if measured <= tolerance:
            break
        if high == max_steps:
            raise RuntimeError(f'Tolerance not reached with {max_steps} steps.')
        low, high = high, min(2 * high, max_steps)

    # Refine to the first integer step count meeting the common tolerance in
    # the bracket (assuming the smooth convergence observed for this field).
    while high - low > 1:
        middle = (low + high) // 2
        if measure(middle) <= tolerance:
            high = middle
        else:
            low = middle
    set_steps(element, high)
    return high, measure(high), np.array(scan)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tolerance', type=float, default=1e-8)
    parser.add_argument('--particles', type=int, default=10000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--max-steps', type=int, default=65536)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    if (not np.isfinite(args.tolerance) or args.tolerance < 1e-12
            or min(args.particles, args.repeats, args.max_steps) < 1):
        parser.error('Use tolerance >= 1e-12 and positive particle/repeat/step counts.')

    # Serial execution also avoids shared BFieldExpansion tracking work arrays
    # being accessed concurrently by different particles.
    context = xo.ContextCpu(omp_num_threads=0)
    context.allow_kernel_compilation = True
    length = 0.8
    rng = np.random.default_rng(12345)
    initial = dict(zip(
        ('x', 'px', 'y', 'py', 'zeta', 'delta'),
        rng.uniform(-1, 1, (6, 24))
        * np.array([0.015, 0.02, 0.015, 0.02, 0.002, 0.03])[:, None],
    ))
    probes = xt.Particles(_context=context, p0c=1e9, q0=1, **initial)
    rigidity = probes.rigidity0[0]  # Signed reference B rho [T m].
    elements = make_elements(context, length, rigidity)

    # Check all three field components off axis, including fringe terms.
    xx, yy, ss = np.meshgrid(
        np.linspace(-0.15, 0.15, 5), np.linspace(-0.15, 0.15, 5),
        np.linspace(0, length, 9), indexing='ij',
    )
    field = elements['BFieldExpansion'].get_field(xx, yy, ss)
    normalized = np.array([field[name] for name in ('Bx', 'By', 'Bs')])
    physical = np.array(elements['SplineBoris'].get_field(xx, yy, ss))
    np.testing.assert_allclose(normalized * rigidity, physical, rtol=2e-12, atol=2e-13)
    print(f'Field agreement: max |difference| = '
          f'{np.max(np.abs(normalized * rigidity - physical)):.3e} T')
    reference = reference_solution(
        elements['SplineBoris'], probes, rigidity, args.tolerance)

    results = {}
    for name, element in elements.items():
        steps, achieved, scan = calibrate(
            element, probes, reference, args.tolerance, args.max_steps)
        results[name] = dict(steps=steps, error=achieved, scan=scan, times=[])
        print(f'\n{name}:')
        print('  steps       error       observed order')
        for i, (n, err) in enumerate(scan):
            order_text = '-'
            if i:
                order = np.log(scan[i - 1, 1] / err) / np.log(n / scan[i - 1, 0])
                order_text = f'{order:.2f}'
            print(f'  {int(n):5d}   {err:.3e}       {order_text:>5s}')

    bunch = xt.Particles(
        _context=context, p0c=1e9, q0=1,
        **{name: np.resize(values, args.particles) for name, values in initial.items()},
    )
    bunch_reference = reference[:, np.arange(args.particles) % reference.shape[1]]
    jobs = []
    for name, result in results.items():
        # Time the convergence scan and the selected integer step count.
        # Reuse a measurement if the selected count is already in the scan.
        counts = sorted(set(result['scan'][:, 0].astype(int)) | {result['steps']})
        result['times_by_steps'] = {int(n): [] for n in counts}
        jobs.extend((name, int(n)) for n in counts)
    print(f'\nTiming {len(jobs)} step-count settings with {args.particles} particles...')
    for name, steps in jobs:
        set_steps(elements[name], steps)
        elements[name].track(bunch.copy())  # Warm up with the full bunch.
    # Alternate order between repeats to reduce timing drift bias.
    for repeat in range(args.repeats):
        for name, steps in jobs[::1 if repeat % 2 == 0 else -1]:
            set_steps(elements[name], steps)
            tracked = bunch.copy()
            start = perf_counter()
            elements[name].track(tracked)
            elapsed = perf_counter() - start  # CPU calls are synchronous.
            # Validate every timed particle, outside the timed region.
            measured_error = error(coordinates(tracked), bunch_reference, length)
            if steps == results[name]['steps'] and measured_error > args.tolerance:
                raise RuntimeError(f'{name}: timed bunch failed the accuracy target.')
            results[name]['times_by_steps'][steps].append(elapsed)

    print(f'\nSerial CPU; {args.particles} particles; median of {args.repeats} passes')
    print(f'Common normalized exit-error tolerance: {args.tolerance:.3e}')
    print('Element            steps     error       time [ms]    ns/particle')
    for name, result in results.items():
        set_steps(elements[name], result['steps'])
        result['times'] = result['times_by_steps'][result['steps']]
        result['seconds'] = np.median(result['times'])
        result['scan_seconds'] = np.array([
            np.median(result['times_by_steps'][int(n)]) for n in result['scan'][:, 0]
        ])
        print(f'{name:18s} {result["steps"]:5d}   {result["error"]:.3e}'
              f'   {result["seconds"] * 1e3:10.3f}'
              f'   {result["seconds"] / args.particles * 1e9:12.1f}')
    ratio = results['BFieldExpansion']['seconds'] / results['SplineBoris']['seconds']
    print(f'Time ratio BFieldExpansion / SplineBoris: {ratio:.3f}')

    if not args.no_plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        for i, (name, result) in enumerate(results.items()):
            axes[0].loglog(*result['scan'].T, '.-', color=f'C{i}', label=name)
            axes[0].plot(result['steps'], result['error'], 'o', color=f'C{i}')
            axes[1].loglog(result['scan_seconds'] / args.particles * 1e6,
                           result['scan'][:, 1], '.-', color=f'C{i}', label=name)
            axes[1].plot(result['seconds'] / args.particles * 1e6,
                         result['error'], 'o', color=f'C{i}')
        for ax in axes[:2]:
            ax.axhline(args.tolerance, color='k', linestyle='--', label='Tolerance')
            ax.set(ylabel='Maximum normalized exit error')
            ax.legend()
            ax.grid(True, which='both', alpha=0.3)
        axes[0].set(xlabel='Integration steps', title='Convergence vs steps')
        axes[1].set(xlabel='Tracking time per particle [µs]', title='Convergence vs runtime')
        axes[2].bar(
            [f'{name}\n{r["steps"]} steps' for name, r in results.items()],
            [r['seconds'] * 1e3 for r in results.values()], color=['C0', 'C1'],
        )
        axes[2].set(ylabel=f'Time for {args.particles} particles [ms]',
                    title=f'Equal accuracy: tolerance {args.tolerance:g}')
        fig.tight_layout()
        plt.show()
    return results


if __name__ == '__main__':
    main()
