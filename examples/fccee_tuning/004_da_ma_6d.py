"""Team-style DA and amplitude-versus-initial-delta survival maps at the septum.

6D tracking, active RF, independent mean-radiation tapering for clean/corrected
machines, followed by the selected radiation model. No correction is run.
MA here is a survival map, not a local Touschek momentum acceptance or lifetime.
Run in IPython with %matplotlib followed by %run 004_da_ma_6d.py.
"""
from pathlib import Path
from time import perf_counter
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
import xobjects as xo

HERE = Path(__file__).resolve().parent
INPUT = HERE / 'response_api_class' / 'machine.json'
SUPPRESS_ARC_DIPOLE_B4 = False       # Installed systematic + random b4 errors only.
OUTPUT_DIR = HERE / 'analysis' / ('da_ma_6d_b4_off' if SUPPRESS_ARC_DIPOLE_B4 else 'da_ma_6d')
LAUNCH_AT = 'septum'
NUM_TURNS = 300                    # Preliminary comparison; team benchmark: 2500.
TRACKING_THREADS = 8               # BLAS can stay single-threaded.
GEMITT_X = 0.71e-9                 # Geometric emittances [m].
GEMITT_Y = 1.4e-12
RADIATION_MODEL = 'mean'          # Choose 'mean' or 'quantum' for tracking.
DA_RADII = np.linspace(0., 20., 11)
DA_ANGLES = np.linspace(0., 2*np.pi, 16, endpoint=False)
MA_DELTAS = np.linspace(-0.012, 0.012, 13)
MA_AMPLITUDES = np.linspace(-20., 20., 11)
# MA recipe inspired by K. André, S. Liuzzo, S. White, P. Raimondi (ESRF),
# "LCC Optics Tuning Updates", 2 July 2026, slide 5: eight phases and equal
# normalized transverse amplitudes.
# https://indico.cern.ch/event/1704042/contributions/7166876/attachments/3307462/5918024/LCC_tuning02072026.pdf
# The identical phase in both modes below is our interpretation; the slide
# does not specify the exact phase pairing or longitudinal initialization.
# Four phases for each amplitude sign = eight directions at nonzero |A|.
# The two transverse modes use the SAME phase; not a 4 x 4 phase product.
MA_PHASES = np.arange(4)*np.pi/4
RNG_SEED = 281047                  # One quantum realization per launch.
PILOT_TURNS = 64
PILOT_PARTICLES = 128
RUN_SCANS = True                  # False: preparation and runtime pilot only.
SHOW_PLOTS = True
CHECK_AMPLITUDE_DETUNING = False   # Optional conservative diagnostic before tapering.
DETUNING_TURNS = 512
DETUNING_MAX_SECONDS_PER_CASE = 120.
DETUNING_AMPLITUDES = np.array([[.5, .5], [1., .5], [2., .5], [3., .5],
    [.5, 1.], [.5, 2.], [.5, 3.], [1., 1.], [2., 2.], [3., 3.]])
CO_SEARCH_AT = 'ipd'
STEPS_R_MATRIX = dict(dx=1e-8, dpx=1e-9, dy=1e-8, dpy=1e-9,
                      dzeta=1e-7, ddelta=1e-8)


def detuning_slopes(tunes, maximum):
    """Vary each action separately; keep the other amplitude fixed at 0.5 sigma."""
    slopes = np.empty((2, 2))  # rows: tune mode; columns: geometric action.
    for mode in range(2):
        selected = (DETUNING_AMPLITUDES[:, 1-mode] == .5) & (DETUNING_AMPLITUDES[:, mode] <= maximum)
        action_in_emittances = DETUNING_AMPLITUDES[selected, mode]**2/2
        slopes[:, mode] = np.polynomial.polynomial.polyfit(
            action_in_emittances, tunes[selected]-tunes[0], 1)[1]/[GEMITT_X, GEMITT_Y][mode]
    return slopes


def track_amplitude_detuning(line, tw, turns, label):
    import nafflib

    if line.element_names[0] != LAUNCH_AT:
        raise ValueError('Cycle to septum before the amplitude diagnostic')
    particles = line.build_particles(particle_on_co=tw.particle_on_co,
        W_matrix=tw.W_matrix[0], method='4d',
        x_norm=DETUNING_AMPLITUDES[:, 0], px_norm=0., y_norm=DETUNING_AMPLITUDES[:, 1], py_norm=0.,
        gemitt_x=GEMITT_X, gemitt_y=GEMITT_Y, zeta=0., delta=0.)
    flags = line.tracker.track_flags
    previous = flags.XS_FLAG_KILL_CAVITY_KICK
    started = perf_counter()
    try:
        flags.XS_FLAG_KILL_CAVITY_KICK = True
        line.track(particles, num_turns=turns, turn_by_turn_monitor=True)
    finally:
        flags.XS_FLAG_KILL_CAVITY_KICK = previous
    elapsed = perf_counter()-started
    mon = line.record_last_track
    order = np.argsort(np.asarray(particles.particle_id))
    state = np.asarray(particles.state)[order]
    if np.any(state <= 0):
        raise RuntimeError(f'{label}: amplitude diagnostic lost particles: {state}')
    np.testing.assert_allclose(mon.delta, 0., atol=1e-14, rtol=0.)
    physical = np.array([np.asarray(getattr(mon, key))-tw[key][0]
                         for key in ('x', 'px', 'y', 'py')])
    normal = np.einsum('ij,jpt->ipt', np.linalg.inv(tw.W_matrix[0, :4, :4]), physical)
    q0 = np.array([tw.qx, tw.qy])
    results = {}
    for n in sorted(set([turns//2, turns])):
        tunes = np.empty((len(DETUNING_AMPLITUDES), 2))
        for mode in range(2):
            signals = normal[2*mode, :, :n]-1j*normal[2*mode+1, :, :n]
            for i, signal in enumerate(signals):
                q = nafflib.get_tune(signal-signal.mean())
                tunes[i, mode] = q + np.round(q0[mode]-q)
        results[str(n)] = dict(tunes=tunes.tolist(), slopes={str(a): detuning_slopes(tunes, a).tolist()
                                                   for a in (1., 2., 3.)})
    np.savez_compressed(OUTPUT_DIR / f'{label}_detuning_tracking.npz',
        normalized=normal, q0=q0, amplitudes=DETUNING_AMPLITUDES, emittances=[GEMITT_X, GEMITT_Y],
        state=state, delta=np.asarray(mon.delta), zeta=np.asarray(mon.zeta))
    print(label, 'tracking seconds', elapsed, 'detuning', results[str(turns)]['slopes']['1.0'], flush=True)
    return dict(seconds=elapsed, turns=turns, estimates=results)


def prepare_machine(clean):
    line = xt.load(INPUT, with_progress=False)
    line.twiss_default.clear()
    line.configure_radiation(model=None)
    if any(getattr(el, 'delta_taper', 0.) != 0.
           for el in line.element_dict.values()):
        raise ValueError('Input must be the untapered standard corrected machine')
    if clean:
        circuits = line.metadata['correction_knobs']
        values = {name: 0. for name in circuits['switches'].values()}
        values.update(on_nonlinear=1., on_chromaticity_corr=0.)
        values.update(circuits['tunes']['nominal_values'])
        line.vars.update(values)
    b4_before = float(line['on_error_arc_dipole_b4'])
    if SUPPRESS_ARC_DIPOLE_B4:
        # Keep the saved correction settings and nominal octupole strengths.
        # This switch multiplies both systematic and random arc-dipole b4 errors.
        line['on_error_arc_dipole_b4'] = 0.
    line.cycle(name_first_element=LAUNCH_AT, inplace=True)
    detuning = None
    if CHECK_AMPLITUDE_DETUNING:
        label = 'clean' if clean else 'corrected'
        line.build_tracker(_context=xo.ContextCpu(omp_num_threads=TRACKING_THREADS))
        tw4d = line.twiss(method='4d', co_search_at=CO_SEARCH_AT, chrom=False,
                         steps_R_matrix=STEPS_R_MATRIX)
        pilot = track_amplitude_detuning(line, tw4d, 64, label+'_pilot')
        estimate = pilot['seconds']*DETUNING_TURNS/64
        print(label, 'estimated detuning tracking seconds:', estimate, flush=True)
        if estimate > DETUNING_MAX_SECONDS_PER_CASE:
            raise RuntimeError('Amplitude-detuning pilot exceeds configured runtime limit')
        detuning = track_amplitude_detuning(line, tw4d, DETUNING_TURNS, label)
        # Radiation Twiss requires a serial context; restore parallelism below.
        line.build_tracker(_context=xo.ContextCpu())
    line.configure_radiation(model='mean')
    line.tracker.track_flags.XS_FLAG_KILL_CAVITY_KICK = False
    # Keep the mean reference energy fixed, sharing the loss among active RF.
    line.compensate_radiation_energy_loss(delta0='zero_mean', co_search_at=CO_SEARCH_AT)
    tw = line.twiss(method='6d', co_search_at=CO_SEARCH_AT, chrom=False,
                    steps_R_matrix=STEPS_R_MATRIX)
    # Independent 6D closure test, with physical radiation and RF kicks enabled.
    co = tw.particle_on_co.copy()
    keys = ('x', 'px', 'y', 'py', 'zeta', 'delta')
    before = np.array([getattr(co, key)[0] for key in keys])
    line.track(co, num_turns=1)
    after = np.array([getattr(co, key)[0] for key in keys])
    if co.state[0] <= 0 or np.any(np.abs(after-before) >
                                 np.array([1e-7, 1e-8, 1e-7, 1e-8, 1e-6, 1e-8])):
        raise ValueError(f'6D closed orbit does not close: {after-before}')
    tt = line.get_table()
    info = dict(amplitude_detuning=detuning, tunes=[float(tw.qx), float(tw.qy), float(tw.qs)],
                arc_dipole_b4_switch_before=b4_before,
                arc_dipole_b4_switch=float(line['on_error_arc_dipole_b4']),
                closed_orbit=before.tolist(), closure_error=(after-before).tolist(),
                betx=float(tw.betx[0]), bety=float(tw.bety[0]),
                aperture_elements=tt.name[np.char.startswith(
                    np.asarray(tt.element_type, dtype=str), 'Limit')].tolist(),
                global_xy_limit=float(line.config.XTRACK_GLOBAL_XY_LIMIT))
    print(('Clean' if clean else 'Corrected'), '6D optics:', info, flush=True)
    if TRACKING_THREADS:
        line.build_tracker(_context=xo.ContextCpu(omp_num_threads=TRACKING_THREADS))
    return line, tw, info


def launch_grids():
    angle, radius = np.meshgrid(DA_ANGLES, DA_RADII, indexing='ij')
    da = dict(x_norm=(radius*np.cos(angle)).ravel(),
              y_norm=(radius*np.sin(angle)).ravel(),
              px_norm=np.zeros(radius.size), py_norm=np.zeros(radius.size),
              delta_offset=np.zeros(radius.size))
    amplitude, delta, phase = np.meshgrid(MA_AMPLITUDES, MA_DELTAS,
                                         MA_PHASES, indexing='ij')
    ma = dict(x_norm=(amplitude*np.cos(phase)).ravel(),
              y_norm=(amplitude*np.cos(phase)).ravel(),
              px_norm=(amplitude*np.sin(phase)).ravel(),
              py_norm=(amplitude*np.sin(phase)).ravel(),
              delta_offset=delta.ravel())
    return da, ma


def make_particles(line, tw, grid):
    coordinates = {key: grid[key] for key in ('x_norm', 'px_norm', 'y_norm', 'py_norm')}
    # Six constraints: four transverse normal coordinates plus physical zeta and
    # delta. The W matrix supplies the linear dispersive offset at initial delta;
    # this is NOT a fixed-delta nonlinear closed orbit. zeta starts at the 6D CO.
    return line.build_particles(particle_on_co=tw.particle_on_co,
        W_matrix=tw.W_matrix[0], gemitt_x=GEMITT_X, gemitt_y=GEMITT_Y,
        zeta=tw.zeta[0], delta=tw.delta[0]+grid['delta_offset'], **coordinates)


def ordered_result(particles, turns):
    # Tracking can reorder lost particles. Every artifact uses launch order.
    order = np.argsort(np.asarray(particles.particle_id))
    state = np.asarray(particles.state)[order].copy()
    at_turn = np.asarray(particles.at_turn)[order].copy()
    survived = state > 0
    return dict(state=state, at_turn=at_turn,
                at_element=np.asarray(particles.at_element)[order].copy(),
                survived=survived, completed_turns=np.where(survived, turns, at_turn))


def plot_results(results):
    for scan in ('da', 'ma'):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained',
                                 num=f'FCC-ee 6D {scan.upper()}', clear=True)
        for col, machine in enumerate(('clean', 'corrected')):
            ax = axes[col]
            key = f'{machine}_{RADIATION_MODEL}_{scan}'
            label = machine.capitalize()
            if machine == 'corrected' and SUPPRESS_ARC_DIPOLE_B4:
                label += ' (arc-dipole b₄ off)'
            ax.set_title(f'{label} — {RADIATION_MODEL} radiation')
            if key not in results:
                ax.text(.5, .5, 'Not yet tracked', ha='center', transform=ax.transAxes)
                continue
            data = results[key]
            if scan == 'da':
                color = ax.scatter(data['x_norm'], data['y_norm'],
                    c=data['completed_turns'], s=15, vmin=0, vmax=NUM_TURNS,
                    cmap='viridis')
                ax.set(xlabel=r'$x_N$ [$\sigma_x$]', ylabel=r'$y_N$ [$\sigma_y$]',
                       aspect='equal')
            else:
                shape = (len(MA_AMPLITUDES), len(MA_DELTAS), len(MA_PHASES))
                mean_turns = data['completed_turns'].reshape(shape).mean(axis=2)
                color = ax.pcolormesh(100*MA_DELTAS, MA_AMPLITUDES, mean_turns,
                    shading='nearest', vmin=0, vmax=NUM_TURNS, cmap='viridis')
                fraction = data['survived'].reshape(shape).mean(axis=2)
                if np.any(fraction == 1) and np.any(fraction < 1):
                    # Boundary between cells where all four phases survive and
                    # cells with at least one loss. Not an interpolated MA value.
                    ax.contour(100*MA_DELTAS, MA_AMPLITUDES,
                        (fraction == 1).astype(float), levels=[.5], colors='white', linewidths=.8)
                ax.set(xlabel='Initial Δδ relative to closed orbit [%]',
                       ylabel='Signed equal transverse amplitude A [σ]',
                       xlim=(100*MA_DELTAS[0], 100*MA_DELTAS[-1]))
            ax.grid(alpha=.15)
            fig.colorbar(color, ax=ax, label=('Completed turns' if scan == 'da'
                         else 'Mean completed turns over four phases per sign'))
        fig.suptitle(f'{scan.upper()} at {LAUNCH_AT}; 6D RF + tapering; {NUM_TURNS} turns\n'
                     + ('Initial Δδ = 0; normalized momenta = 0' if scan == 'da' else
                        'Eight phase directions across both signs; white: all-phase survival boundary'))
        fig.savefig(OUTPUT_DIR / f'{scan}_comparison.png', dpi=180)
        fig.savefig(OUTPUT_DIR / f'{scan}_comparison.pdf')
        if not SHOW_PLOTS:
            plt.close(fig)


def plot_amplitude_detuning(machines):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    for label, info in machines.items():
        estimates = info['amplitude_detuning']['estimates'][str(DETUNING_TURNS)]['slopes']
        matrices = np.array([estimates[str(a)] for a in (1., 2., 3.)])
        for ax, (i, j), title in zip(axes, [(0, 0), (0, 1), (1, 1)],
                ['axx', 'axy (solid), ayx (dotted)', 'ayy']):
            ln, = ax.plot([1, 2, 3], matrices[:, i, j]/1e5, '.-', label=label)
            if i != j:
                ax.plot([1, 2, 3], matrices[:, j, i]/1e5, ':', color=ln.get_color())
            ax.set(xlabel='Maximum amplitude in separate action fit [σ]',
                   ylabel='Detuning coefficient [10⁵ m⁻¹]', title=title, xticks=[1, 2, 3])
            ax.grid(alpha=.25)
    axes[0].legend()
    fig.suptitle('Amplitude detuning at '+LAUNCH_AT+'; radiation/tapering off; cavity kicks off')
    for suffix in ('png', 'pdf'):
        fig.savefig(OUTPUT_DIR / f'amplitude_detuning.{suffix}', dpi=180)
    if not SHOW_PLOTS:
        plt.close(fig)


def main():
    if RADIATION_MODEL not in ('mean', 'quantum'):
        raise ValueError("RADIATION_MODEL must be 'mean' or 'quantum'")
    started = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    da, ma = launch_grids()
    count = len(da['x_norm']) + len(ma['x_norm'])
    summary = dict(input=str(INPUT), input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        launch_at=LAUNCH_AT, turns=NUM_TURNS, tracking_threads=TRACKING_THREADS,
        gemitt_x=GEMITT_X, gemitt_y=GEMITT_Y,
        suppress_arc_dipole_b4=SUPPRESS_ARC_DIPOLE_B4,
        amplitude_detuning_settings=dict(enabled=CHECK_AMPLITUDE_DETUNING,
            turns=DETUNING_TURNS, amplitudes=DETUNING_AMPLITUDES.tolist(),
            radiation=False, tapering=False, rf='XS_FLAG_KILL_CAVITY_KICK',
            definition='dQ_i/dJ_j; vary each action separately, other mode at 0.5 sigma'),
        correction_settings='saved values; no re-correction',
        radiation_model=RADIATION_MODEL, tapering='independent zero-mean-energy tapering',
        longitudinal_launch='zeta=closed orbit; delta=closed orbit + requested offset',
        transverse_launch='each machine 6D W matrix, identical geometric emittances',
        ma_definition='initial-delta versus equal-amplitude survival; same phase in both modes',
        da_radii=DA_RADII.tolist(), da_angles=DA_ANGLES.tolist(),
        ma_deltas=MA_DELTAS.tolist(), ma_amplitudes=MA_AMPLITUDES.tolist(),
        ma_phases=MA_PHASES.tolist(), rng_seed=RNG_SEED,
        beam_beam=False, beamstrahlung=False, seed_count=1,
        aperture_policy='retain input apertures and global tracking limit; no added chamber model',
        machines={}, pilots=[], cases={}, status='running')
    results = {}
    for machine in ('clean', 'corrected'):
        line, tw, info = prepare_machine(clean=machine == 'clean')
        summary['machines'][machine] = info
        np.savez_compressed(OUTPUT_DIR / f'{machine}_optics.npz',
                            name=np.asarray(tw.name, dtype=str), s=tw.s, delta=tw.delta,
                            W_matrix_at_launch=tw.W_matrix[0], closed_orbit=info['closed_orbit'])
        model = RADIATION_MODEL
        line.configure_radiation(model=model)
        pilot_grid = dict(x_norm=np.linspace(.5, 5., PILOT_PARTICLES),
            y_norm=np.linspace(.5, 5., PILOT_PARTICLES), px_norm=np.zeros(PILOT_PARTICLES),
            py_norm=np.zeros(PILOT_PARTICLES), delta_offset=np.zeros(PILOT_PARTICLES))
        particles = make_particles(line, tw, pilot_grid)
        if model == 'quantum':
            particles._init_random_number_generator(
                seeds=np.arange(PILOT_PARTICLES, dtype=np.uint32)+RNG_SEED)
        t0 = perf_counter()
        line.track(particles, num_turns=PILOT_TURNS)
        seconds = perf_counter()-t0
        estimate = seconds*NUM_TURNS/PILOT_TURNS*count/PILOT_PARTICLES
        summary['pilots'].append(dict(machine=machine, model=model, seconds=seconds,
            particles=PILOT_PARTICLES, turns=PILOT_TURNS,
            estimated_scan_seconds_all_survive=estimate))
        print(f'{machine}/{model}: pilot {seconds:.1f} s; projected scans '
              f'{estimate/3600:.2f} h if all particles survive', flush=True)
        (OUTPUT_DIR/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        if not RUN_SCANS:
            del line, tw
            continue
        for scan, grid in (('da', da), ('ma', ma)):
            key = f'{machine}_{model}_{scan}'
            particles = make_particles(line, tw, grid)
            if model == 'quantum':
                particles._init_random_number_generator(
                    seeds=np.arange(len(grid['x_norm']), dtype=np.uint32)+RNG_SEED)
            initial = {f'initial_{k}': np.array(getattr(particles, k))
                       for k in ('x', 'px', 'y', 'py', 'zeta', 'delta')}
            print(f'Start {key}: {len(grid["x_norm"])} particles, {NUM_TURNS} turns', flush=True)
            t0 = perf_counter()
            for turn in range(0, NUM_TURNS, 50):
                line.track(particles, num_turns=min(50, NUM_TURNS-turn))
                print(f'{key}: {min(turn+50, NUM_TURNS)}/{NUM_TURNS} turns; '
                      f'{np.sum(particles.state > 0)} alive; {perf_counter()-t0:.1f} s', flush=True)
                if not np.any(particles.state > 0):
                    break
            data = dict(**grid, **initial, **ordered_result(particles, NUM_TURNS))
            results[key] = data
            np.savez_compressed(OUTPUT_DIR/f'{key}.npz', **data)
            summary['cases'][key] = dict(seconds=perf_counter()-t0,
                particles=len(grid['x_norm']), survived=int(data['survived'].sum()))
            (OUTPUT_DIR/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
            plot_results(results)
        del line, tw
    if CHECK_AMPLITUDE_DETUNING:
        plot_amplitude_detuning(summary['machines'])
    summary.update(status='complete' if RUN_SCANS else 'pilot_only',
                   elapsed_seconds=perf_counter()-started)
    (OUTPUT_DIR/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Saved', OUTPUT_DIR, flush=True)
    if SHOW_PLOTS:
        plt.show()


if __name__ == '__main__':
    main()
