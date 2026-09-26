"""Twiss, survey, and 100 particles tracked for 100 turns in the ELENA model.

Run from the repository root with::

    %run examples/bfieldexpansion/elena_twiss_survey_tracking.py
    python examples/bfieldexpansion/elena_twiss_survey_tracking.py --no-plot --save-prefix /tmp/elena

The lattice and fitted dipole coefficients are all in elena.py. The beam is
an on-momentum, matched Gaussian distribution with geometric rms emittances
of 1 mm mrad in both planes, at 100 MeV/c. Tracking uses pkin_const=False.
RF and apertures are omitted; the survival plot is not an acceptance estimate.
"""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import xtrack as xt

from elena import CIRCUMFERENCE, DIPOLE_SEGMENTS, make_line


NUM_PARTICLES = 100
NUM_TURNS = 100
GEOMETRIC_EMITTANCE = 1e-6


def make_plots(line, twiss, survey, monitor, survived):
    import matplotlib.pyplot as plt

    figures = {}
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8), layout='constrained')
    axes[0].plot(twiss.s, twiss.betx, label=r'$\beta_x$')
    axes[0].plot(twiss.s, twiss.bety, label=r'$\beta_y$')
    axes[0].set(ylabel=r'$\beta$ [m]',
                title=f'ELENA field-map optics: Qx={twiss.qx:.5f}, Qy={twiss.qy:.5f}')
    axes[1].plot(twiss.s, twiss.dx, label=r'$D_x$')
    axes[1].plot(twiss.s, twiss.dy, label=r'$D_y$')
    axes[1].set_ylabel('Dispersion [m]')
    axes[2].plot(twiss.s, 1e3 * twiss.x, label='x')
    axes[2].plot(twiss.s, 1e3 * twiss.y, label='y')
    axes[2].set(xlabel='s [m]', ylabel='Closed orbit [mm]')
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    figures['twiss'] = fig

    fig, ax = plt.subplots(figsize=(8, 7), layout='constrained')
    ax.plot(survey.Z, survey.X, color='0.55', label='Survey reference path')
    for i, name in enumerate(line.element_names):
        if isinstance(line[name], xt.BFieldExpansion):
            ax.plot(survey.Z[i:i+2], survey.X[i:i+2], color='C0', lw=3)
        elif isinstance(line[name], xt.Quadrupole):
            ax.plot(survey.Z[i:i+2], survey.X[i:i+2], color='C1', lw=5)
    # Proxy artists avoid repeating a legend entry for each segment.
    ax.plot([], [], color='C0', lw=3, label='Dipole field-map segments')
    ax.plot([], [], color='C1', lw=5, label='Quadrupoles')
    # Twiss and survey refer to the same entrance planes. Rotate each local
    # transverse offset into the lab frame before adding it to the reference path.
    np.testing.assert_array_equal(twiss.name, survey.name)
    closed_orbit = (survey.XYZ + twiss.x[:, None] * survey.ex
                    + twiss.y[:, None] * survey.ey)
    ax.plot(closed_orbit[:, 2], closed_orbit[:, 0], color='C3', ls='--',
            lw=1.5, label='Closed orbit')
    ax.plot(survey.Z[0], survey.X[0], 'ko', ms=5, label='Tracking observation point')
    ax.quiver(survey.Z[0], survey.X[0], survey.es[0, 2], survey.es[0, 0],
              angles='xy', scale_units='xy', scale=1.5, color='k', width=0.004)
    ax.set(xlabel='Z [m]', ylabel='X [m]',
           title=f'ELENA survey: circumference {CIRCUMFERENCE:.6f} m')
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
    ax.legend(loc='best')
    figures['survey'] = fig

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    active = monitor.state > 0
    colours = np.broadcast_to(np.arange(NUM_PARTICLES)[:, None], active.shape)
    for ax, position, momentum, label in (
            (axes[0, 0], 'x', 'px', 'x'), (axes[0, 1], 'y', 'py', 'y')):
        position_values = 1e3 * (getattr(monitor, position) - twiss[position][0])
        momentum_values = 1e3 * (getattr(monitor, momentum) - twiss[momentum][0])
        ax.scatter(position_values[active], momentum_values[active],
                   c=colours[active], cmap='turbo', s=2, alpha=0.5, rasterized=True)
        ax.set(xlabel=f'{label} − closed orbit [mm]',
               ylabel=rf'$10^3 (p_{label} - p_{{{label},co}})$',
               title=f'{label} phase space at the observation point')
    turns = np.arange(NUM_TURNS)
    for coordinate in ('x', 'y'):
        values = np.where(active, getattr(monitor, coordinate), np.nan)
        axes[1, 0].plot(turns, 1e3 * np.nanstd(values, axis=0), label=coordinate)
    axes[1, 0].set(xlabel='Turn', ylabel='Beam rms size [mm]')
    axes[1, 0].legend()
    # The turn monitor records entrances; append the state after turn 100.
    counts = np.r_[np.sum(active, axis=0), survived]
    axes[1, 1].step(np.arange(NUM_TURNS + 1), counts, where='post')
    axes[1, 1].set(xlabel='Completed turns', ylabel='Surviving particles',
                   ylim=(0, NUM_PARTICLES * 1.05))
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
    fig.suptitle('100 particles × 100 turns; geometric rms emittance 1 mm mrad')
    figures['tracking'] = fig
    return figures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-integration-steps', type=int, default=10,
                        help='RK4 steps per dipole spline segment (default: 10)')
    parser.add_argument('--no-plot', action='store_true')
    parser.add_argument('--save-prefix', type=Path)
    args = parser.parse_args()
    if args.num_integration_steps < 1:
        parser.error('--num-integration-steps must be positive')

    line = make_line(num_integration_steps=args.num_integration_steps)
    # More observation points in the long straights make the drift parabolas
    # visible in the Twiss plot. The dipole spline segments are left intact.
    line.slice_thick_elements([
        xt.Strategy(None),
        xt.Strategy(xt.Uniform(20, mode='thick'), element_type=xt.Drift)])
    survey = line.survey()
    np.testing.assert_allclose(line.get_length(), CIRCUMFERENCE, rtol=0, atol=1e-12)
    np.testing.assert_allclose(survey.XYZ[-1], survey.XYZ[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(survey.E_matrix[-1], survey.E_matrix[0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(survey.theta[-1] - survey.theta[0], -2*np.pi,
                               rtol=0, atol=1e-12)
    twiss = line.twiss4d()
    assert np.all(np.isfinite(twiss.betx)) and np.all(twiss.betx > 0)
    assert np.all(np.isfinite(twiss.bety)) and np.all(twiss.bety > 0)
    np.testing.assert_allclose(twiss.x[-1], twiss.x[0], rtol=0, atol=1e-9)
    print(f'Circumference: {line.get_length():.9f} m', flush=True)
    print(f'Dipoles: 6 × {len(DIPOLE_SEGMENTS)} BFieldExpansion segments; '
          f'{args.num_integration_steps} RK4 steps per segment', flush=True)
    print(f'Tunes: Qx={twiss.qx:.8f}, Qy={twiss.qy:.8f}', flush=True)
    print(f'Maximum |closed orbit x|: {1e3*np.max(np.abs(twiss.x)):.6f} mm', flush=True)
    print(f'Survey closure: {np.linalg.norm(survey.XYZ[-1]-survey.XYZ[0]):.3e} m', flush=True)

    rng = np.random.default_rng(2026)
    beta_gamma = line.particle_ref.beta0[0] * line.particle_ref.gamma0[0]
    nemitt = GEOMETRIC_EMITTANCE * beta_gamma
    particles = line.build_particles(
        particle_on_co=twiss.particle_on_co, W_matrix=twiss.W_matrix[0],
        x_norm=rng.normal(size=NUM_PARTICLES), px_norm=rng.normal(size=NUM_PARTICLES),
        y_norm=rng.normal(size=NUM_PARTICLES), py_norm=rng.normal(size=NUM_PARTICLES),
        nemitt_x=nemitt, nemitt_y=nemitt, delta=0., zeta=0., method='4d')
    print(f'Tracking {NUM_PARTICLES} particles for {NUM_TURNS} turns...', flush=True)
    start = perf_counter()
    line.track(particles, num_turns=NUM_TURNS, turn_by_turn_monitor=True)
    elapsed = perf_counter() - start
    monitor = line.record_last_track
    survived = np.count_nonzero(particles.state > 0)
    print(f'Survived: {survived}/{NUM_PARTICLES}; tracking time: {elapsed:.3f} s', flush=True)
    assert monitor.x.shape == (NUM_PARTICLES, NUM_TURNS)
    assert np.all(monitor.state > 0) and survived == NUM_PARTICLES
    assert np.all(particles.at_turn == NUM_TURNS)
    for coordinate in ('x', 'px', 'y', 'py', 'zeta', 'delta'):
        assert np.all(np.isfinite(getattr(monitor, coordinate)))
        assert np.all(np.isfinite(getattr(particles, coordinate)))

    if not args.no_plot or args.save_prefix:
        figures = make_plots(line, twiss, survey, monitor, survived)
        if args.save_prefix:
            args.save_prefix.parent.mkdir(parents=True, exist_ok=True)
            for name, fig in figures.items():
                path = args.save_prefix.with_name(args.save_prefix.name + f'_{name}.png')
                fig.savefig(path, dpi=160)
                print(f'Saved {path}')
        if not args.no_plot:
            import matplotlib.pyplot as plt
            plt.show()


if __name__ == '__main__':
    main()
