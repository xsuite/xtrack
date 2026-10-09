"""Two bunches racing in PIMM: C-12(6+) at 7 MeV/u and equal-rigidity protons.

Fixed magnetic rigidity, two h=1 RF systems, and no collective forces. Both
species see both cavities. Separate coasting BeamStatsMonitors select the two
particle-id ranges and reconstruct an ideal charge-current pickup in lab time.
Coasting here describes the monitor's full-period acquisition, not the beam:
the two beams are bunched.

Examples (paths are independent of the working directory)::

    python 004_pimm_carbon_protons.py
    python 004_pimm_carbon_protons.py --no-show --output-dir ./pickup_plots

The tracker is compiled from source because this example needs the coasting
monitor's new frame-clock time binning. No full kernel-cache rebuild is needed.
The PIMM test lattice is a demonstration model, not a CNAO/MedAustron model.
Their 7 MeV/u injection energy motivates the carbon energy used here; equal
rigidity requires protons at about 27.5 MeV, not their usual injection energy.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.constants import c as clight, elementary_charge

import xtrack as xt
import xtrack.synctime as st


CARBON_EKIN_PER_NUCLEON = 7e6  # eV/u; 84 MeV total kinetic energy per ion
FRAME_FRACTION = .45  # Window must move faster than the faster (proton) beam.
PROTON_ID_START = 1_000_000
COLORS = ('#007f86', '#d65b32')


def simulate(num_particles=4000, num_carbon_turns=64, num_slices=512):
    data = Path(__file__).resolve().parents[2] / 'test_data' / 'pimms'
    line = xt.load([data / 'PIMMS.seq', data / 'pimms_optics.str']).pimms
    line.set_particle_ref('Carbon-12', kinetic_energy0=12*CARBON_EKIN_PER_NUCLEON)
    line.configure_bend_model(core='bend-kick-bend', edge='full')
    carbon = line.particle_ref
    proton = xt.Particles('proton', p0c=carbon.p0c[0] / carbon.q0)
    circumference = line.get_length()
    beta_c = carbon.beta0[0]
    beta_p = proton.beta0[0]
    periods = circumference / (clight * np.array([beta_c, beta_p]))
    frequencies = 1 / periods
    mass_ratio = proton.mass0 / carbon.mass0
    charge_ratio = proton.q0 / carbon.q0
    chi_p = charge_ratio / mass_ratio

    # Find the two transverse closed orbits with RF off. Equal rigidity means
    # delta=chi-1 in Xsuite's shared reference, including the proton near +1.
    tw_c = line.twiss4d()
    tw_p = line.twiss4d(delta0=chi_p - 1,
                       mass_ratio=mass_ratio, charge_ratio=charge_ratio)
    rng = np.random.default_rng(20261009)
    sigma_t = .01 * periods[0]
    arrival_offsets = np.array([.08, .28]) * periods[0]
    voltages = (500., 300.)  # V, deliberately modest for this circulation demo
    beams = []
    for ii, (tw, species, chi, beta, voltage) in enumerate(zip(
            (tw_c, tw_p), (carbon, proton), (1., chi_p),
            (beta_c, beta_p), voltages)):
        # Approximate small-amplitude longitudinal matching to each own RF.
        # alpha refers to the common equal-rigidity orbit. With both RF systems
        # on this is only an initial match, not a stationary two-RF solution.
        eta = tw_c.momentum_compaction_factor - 1/species.gamma0[0]**2
        sigma_phase = 2*np.pi*frequencies[ii]*sigma_t
        sigma_dp = sigma_phase * np.sqrt(
            species.q0 * voltage
            / (2*np.pi*abs(eta)*beta**2*species.energy0[0]))
        dp = sigma_dp * rng.normal(size=num_particles)
        bunch = line.build_particles(
            particle_on_co=tw.particle_on_co, W_matrix=tw.W_matrix[0],
            x_norm=rng.normal(size=num_particles),
            px_norm=rng.normal(size=num_particles),
            y_norm=rng.normal(size=num_particles),
            py_norm=rng.normal(size=num_particles),
            nemitt_x=1e-7, nemitt_y=1e-7,
            delta=chi*(1 + dp) - 1,
            zeta=-beta_c*clight*(arrival_offsets[ii]
                                 + sigma_t*rng.normal(size=num_particles)),
            # Equal total bunch charge makes the two current traces comparable.
            weight=1e8 / species.q0 / num_particles)
        if ii == 1:
            bunch.particle_id += PROTON_ID_START
            bunch.parent_particle_id += PROTON_ID_START
        beams.append(bunch)
    particles = xt.Particles.merge(beams)

    line.discard_tracker()
    for ii, (name, position, beta, voltage) in enumerate(zip(
            ('rf_carbon', 'rf_proton'), (.001, .01), (beta_c, beta_p), voltages)):
        # Phase is referred to the same lab-time origin at both locations.
        phase = -2*np.pi*frequencies[ii]*(arrival_offsets[ii]
                                        + position/(beta*clight))
        line.insert(name, xt.Cavity(voltage=voltage, frequency=frequencies[ii],
                                    phase=phase, absolute_time=True), at=position)

    monitor_names = ('pickup_carbon', 'pickup_proton')
    monitors = []
    for name, first_id in zip(monitor_names, (0, PROTON_ID_START)):
        monitor = xt.BeamStatsMonitor(
            start_at_turn=0, stop_at_turn=num_carbon_turns + 1,
            coasting=True, num_slices=num_slices,
            particle_id_range=(first_id, first_id + num_particles),
            stats=['num_particles'])
        line.env.elements[name] = monitor
        monitors.append(monitor)
    line.insert(list(monitor_names), at=0)
    st.install_sync_time_at_collective_elements(
        line, frame_clock=True, frame_relative_length=FRAME_FRACTION,
        at_element_names=monitor_names)
    # Compile the updated monitor kernel without replacing the user's cache.
    with xt.settings.override(allow_kernel_compilation=True):
        line.build_tracker(use_prebuilt_kernels=False)

    num_frames = int(np.ceil((num_carbon_turns + 1) / FRAME_FRACTION)) + 1
    print(f'Ring length: {circumference:.2f} m')
    print(f'Carbon: 7 MeV/u, f_rev = {frequencies[0]/1e3:.3f} kHz, '
          f'T_rev = {periods[0]*1e6:.4f} us')
    print(f'Proton: {proton.kinetic_energy0[0]/1e6:.4f} MeV, '
          f'f_rev = {frequencies[1]/1e3:.3f} kHz, '
          f'T_rev = {periods[1]*1e6:.4f} us')
    print(f'f_proton / f_carbon = {frequencies[1]/frequencies[0]:.6f}')
    print(f'{num_slices} samples per carbon period '
          f'({periods[0]/num_slices*1e9:.2f} ns bins)')
    line.track(particles, num_turns=num_frames, with_progress=20)

    # Monitor rows are lab-time bins of one carbon reference period. They are
    # NOT SyncTime frames and NOT either species' individual turn counters.
    time = monitors[0].time_centers(line_length=circumference, beta0=beta_c)
    counts = np.stack([np.asarray(mon.num_particles) for mon in monitors])
    dt = periods[0] / num_slices
    currents = counts * np.array([carbon.q0, proton.q0])[:, None, None]
    currents *= elementary_charge / dt
    survivors = []
    for ii, first_id in enumerate((0, PROTON_ID_START)):
        mask = ((particles.particle_id >= first_id)
                & (particles.particle_id < first_id + num_particles))
        # Waiting SyncTime particles have reserved negative states; they live.
        alive = (particles.state > 0) | (particles.state < -st.COAST_STATE_RANGE_START)
        survivors.append(np.count_nonzero(mask & alive) / num_particles)
        turns = particles.at_turn[mask & alive]
        print(f'{("Carbon", "Proton")[ii]}: {survivors[-1]:.1%} surviving, '
              f'{turns.min()}--{turns.max()} completed revolutions')

    return dict(line=line, particles=particles, monitors=monitors,
                time_s=time, current_A=currents, counts=counts,
                periods_s=periods, frequencies_Hz=frequencies,
                proton_kinetic_energy_eV=proton.kinetic_energy0[0],
                arrival_offsets_s=arrival_offsets,
                num_carbon_turns=num_carbon_turns,
                survivors=np.array(survivors))


def plot_pickup(result, output_dir, show=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    time_us = result['time_s'].ravel() * 1e6
    currents = result['current_A'] * 1e3  # mA
    period_us = result['periods_s'][0] * 1e6
    n_turns = result['num_carbon_turns']
    labels = (r'$^{12}$C$^{6+}$', 'protons')
    freqs = result['frequencies_Hz'] / 1e3

    with plt.rc_context({'font.size': 8, 'axes.spines.top': False,
                         'axes.spines.right': False, 'figure.facecolor': 'white'}):
        fig = plt.figure(figsize=(9, 7), dpi=100, layout='constrained')
        grid = fig.add_gridspec(3, 2, height_ratios=(1, 1, 1.45))
        fig.suptitle('Two bunches, one magnetic rigidity', fontsize=15, weight='bold')
        first = fig.add_subplot(grid[0, :])
        last = fig.add_subplot(grid[1, :], sharey=first)
        for ax, start, title in (
                (first, 0., 'Initial passages'),
                (last, (n_turns - 8)*period_us, 'Later passages')):
            selection = (time_us >= start) & (time_us <= start + 8*period_us)
            for ii in range(2):
                ax.plot(time_us[selection], currents[ii].ravel()[selection],
                        color=COLORS[ii], lw=1.25,
                        label=f'{labels[ii]}  |  {freqs[ii]:.1f} kHz')
            ax.set(xlim=(start, start + 8*period_us),
                   xlabel='Laboratory time [us]', ylabel='Pickup current [mA]')
            ax.set_title(title, loc='left', weight='bold')
            ax.grid(alpha=.15)
        first.legend(loc='upper right', ncols=2, frameon=False)
        first.text(.005, .96,
                   f'Carbon: 7 MeV/u   |   Proton: '
                   f'{result["proton_kinetic_energy_eV"]/1e6:.2f} MeV\n'
                   f'PIMM: 75.24 m   |   f_p / f_C = {freqs[1]/freqs[0]:.3f}',
                   transform=first.transAxes, va='top', fontsize=7.5)
        first.set_ylim(0, 1.32*currents.max())

        vmax = currents.max()
        for ii in range(2):
            ax = fig.add_subplot(grid[2, ii])
            cmap = LinearSegmentedColormap.from_list('species', ['#ffffff', COLORS[ii]])
            picture = ax.imshow(currents[ii], origin='lower', aspect='auto',
                extent=(-.5*period_us, .5*period_us, -.5, n_turns + .5),
                cmap=cmap, vmin=0, vmax=vmax, interpolation='nearest')
            ax.set_title(f'{labels[ii]} — all recorded passages', loc='left', weight='bold')
            ax.set(xlabel='Time within a carbon period [us]',
                   ylabel='Carbon reference-period index')
            fig.colorbar(picture, ax=ax, label='Pickup current [mA]', fraction=.045)
        fig.savefig(output_dir / '004_pimm_carbon_protons.png', dpi=180)
        fig.savefig(output_dir / '004_pimm_carbon_protons.pdf')

    np.savez_compressed(output_dir / '004_pimm_carbon_protons.npz',
        time_s=result['time_s'], current_A=result['current_A'],
        counts=result['counts'], periods_s=result['periods_s'],
        frequencies_Hz=result['frequencies_Hz'],
        carbon_kinetic_energy_per_nucleon_eV=CARBON_EKIN_PER_NUCLEON,
        proton_kinetic_energy_eV=result['proton_kinetic_energy_eV'])
    print(f'Plots and pickup arrays saved in {output_dir.resolve()}')
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-particles', type=int, default=4000)
    parser.add_argument('--num-carbon-turns', type=int, default=64)
    parser.add_argument('--num-slices', type=int, default=512)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).with_suffix(''))
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.num_carbon_turns < 8 or args.num_particles < 1 or args.num_slices < 1:
        parser.error('Use at least 8 carbon periods, 1 particle and 1 slice')
    if args.no_show:
        import matplotlib
        matplotlib.use('Agg')
    result = simulate(args.num_particles, args.num_carbon_turns, args.num_slices)
    plot_pickup(result, args.output_dir, show=not args.no_show)
