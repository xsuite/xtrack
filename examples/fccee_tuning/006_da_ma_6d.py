"""Team-style DA and amplitude-versus-initial-delta survival maps at the septum.

6D tracking, active RF, independent mean-radiation tapering for clean/corrected
machines, followed by mean and quantum radiation tracking. No correction is run.
MA here is a survival map, not a local Touschek momentum acceptance or lifetime.
Run in IPython with %matplotlib followed by %run 006_da_ma_6d.py.
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
OUTPUT_DIR = HERE / 'analysis' / 'da_ma_6d'
LAUNCH_AT = 'septum'
NUM_TURNS = 300                    # Preliminary comparison; team benchmark: 2500.
TRACKING_THREADS = 8               # BLAS can stay single-threaded.
GEMITT_X = 0.71e-9                 # Geometric emittances [m].
GEMITT_Y = 1.4e-12
RADIATION_MODELS = ('mean', 'quantum')
DA_RADII = np.linspace(0., 20., 11)
DA_ANGLES = np.linspace(0., 2*np.pi, 16, endpoint=False)
MA_DELTAS = np.linspace(-0.03, 0.03, 13)
MA_AMPLITUDES = np.linspace(-20., 20., 11)
# Four phases for each amplitude sign = eight directions at nonzero |A|.
# The two transverse modes use the SAME phase; not a 4 x 4 phase product.
MA_PHASES = np.arange(4)*np.pi/4
RNG_SEED = 281047                  # One quantum realization per launch.
PILOT_TURNS = 64
PILOT_PARTICLES = 128
RUN_SCANS = True                  # False: preparation and runtime pilot only.
SHOW_PLOTS = True
CO_SEARCH_AT = 'ipd'
STEPS_R_MATRIX = dict(dx=1e-8, dpx=1e-9, dy=1e-8, dpy=1e-9,
                      dzeta=1e-7, ddelta=1e-8)


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
        values.update(circuits['tunes']['family_nominal_values'])
        line.vars.update(values)
    line.cycle(name_first_element=LAUNCH_AT, inplace=True)
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
    info = dict(tunes=[float(tw.qx), float(tw.qy), float(tw.qs)],
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
        fig, axes = plt.subplots(len(RADIATION_MODELS), 2, figsize=(12, 5*len(RADIATION_MODELS)),
                                 squeeze=False, layout='constrained',
                                 num=f'FCC-ee 6D {scan.upper()}', clear=True)
        for row, model in enumerate(RADIATION_MODELS):
            for col, machine in enumerate(('clean', 'corrected')):
                ax = axes[row, col]
                key = f'{machine}_{model}_{scan}'
                ax.set_title(f'{machine.capitalize()} — {model} radiation')
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
                           ylabel='Signed equal transverse amplitude A [σ]')
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


def main():
    started = perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    da, ma = launch_grids()
    count = len(da['x_norm']) + len(ma['x_norm'])
    summary = dict(input=str(INPUT), input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        launch_at=LAUNCH_AT, turns=NUM_TURNS, tracking_threads=TRACKING_THREADS,
        gemitt_x=GEMITT_X, gemitt_y=GEMITT_Y,
        radiation_models=list(RADIATION_MODELS), tapering='independent zero-mean-energy tapering',
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
        for model in RADIATION_MODELS:
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
    summary.update(status='complete' if RUN_SCANS else 'pilot_only',
                   elapsed_seconds=perf_counter()-started)
    (OUTPUT_DIR/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('Saved', OUTPUT_DIR, flush=True)
    if SHOW_PLOTS:
        plt.show()


if __name__ == '__main__':
    main()
