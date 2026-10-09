"""Two bunches racing in PIMM: C-12(6+) at 7 MeV/u and equal-rigidity protons.

A common rigidity ramp, two chirped h=1 RF systems, and no collective forces. Both
species see both cavities. Separate coasting BeamStatsMonitors select the two
particle-id ranges and reconstruct an ideal charge-current pickup in lab time.
Coasting here describes the monitor's full-period acquisition, not the beam:
the two beams are bunched.

By default carbon accelerates from 7 to 8 MeV/u in about 422 us, followed by
105 us at the final energy. Protons follow the same magnetic rigidity. The
normalized magnet strengths stay fixed, and both RF frequencies and their
integrated phases follow the ramp. The RF voltages are 40 kV and 30 kV.
Pickup slices have fixed zeta width; their lab-time widths follow the ramp.
The initial rms bunch duration is 5% of the injection carbon period
(about 103 ns); the momentum spread uses the small-amplitude RF matching.

Examples (paths are independent of the working directory)::

    python 004a_pimm_carbon_protons.py
    python 004a_pimm_carbon_protons.py --output-dir ./pickup_data
    python 004a_pimm_carbon_protons.py --num-particles 128
    python 004b_plot_pimm_carbon_protons.py

Simulation uses six OpenMP threads by default and saves an NPZ file. Run the
separate plotting script to display or export figures. On macOS, source
compilation needs an OpenMP-enabled compiler (set CC/CXX in your environment).

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
from scipy.integrate import cumulative_trapezoid

import xobjects as xo
import xtrack as xt
import xtrack.synctime as st


CARBON_EKIN_PER_NUCLEON = 7e6  # eV/u; 84 MeV total kinetic energy per ion
FRAME_FRACTION = .45  # Window must move faster than the faster (proton) beam.
PROTON_ID_START = 1_000_000


def simulate(num_particles=4000, num_carbon_turns=256, num_slices=512,
             final_energy_per_nucleon=8e6, num_threads=6):
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

    # Keep the normalized magnetic strengths fixed: the physical dipole and
    # quadrupole fields then rise with the common reference rigidity.
    # Duration is measured in INJECTION carbon periods, not actual turns.
    duration = num_carbon_turns * periods[0]
    ramp_duration = .8 * duration
    times = np.linspace(0, duration + 8*periods[0], 20001)
    ramp_fraction = np.clip(times / ramp_duration, 0, 1)
    smooth = 10*ramp_fraction**3 - 15*ramp_fraction**4 + 6*ramp_fraction**5
    smooth_rate = 30*ramp_fraction**2*(1 - ramp_fraction)**2 / ramp_duration
    p_start = float(carbon.p0c[0])
    p_end = np.sqrt((carbon.mass0 + 12*final_energy_per_nucleon)**2
                    - carbon.mass0**2)
    momentum = p_start + (p_end - p_start)*smooth
    momentum_rate = (p_end - p_start)*smooth_rate
    voltage_acc = circumference/clight * momentum_rate/carbon.q0
    # Both beams require the same energy gain per charge per revolution.
    voltages = (40000., 30000.)
    if np.max(np.abs(voltage_acc)) >= min(voltages):
        raise ValueError('Ramp too fast for these RF voltages; increase num_carbon_turns')

    # Find the two transverse closed orbits with RF off. Equal rigidity means
    # delta=chi-1 in Xsuite's shared reference, including the proton near +1.
    tw_c = line.twiss4d()
    tw_p = line.twiss4d(delta0=chi_p - 1,
                       mass_ratio=mass_ratio, charge_ratio=charge_ratio)
    rng = np.random.default_rng(20261009)
    sigma_t = .05 * periods[0]
    # Delay both bunches so the longer Gaussian tails fit after the leading
    # edge of the first SyncTime window. Keep their relative timing unchanged.
    arrival_offsets = np.array([.58, .78]) * periods[0]
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
    rf_phases = []
    for ii, (name, position, beta, voltage) in enumerate(zip(
            ('rf_carbon', 'rf_proton'), (.001, .01), (beta_c, beta_p), voltages)):
        species = (carbon, proton)[ii]
        pc = momentum * species.q0/carbon.q0
        rf_frequency = clight/circumference * pc/np.sqrt(pc**2 + species.mass0**2)
        # Integrate frequency; frequency(t)*t is not the chirped RF phase.
        rf_phase = (2*np.pi*cumulative_trapezoid(rf_frequency, times, initial=0)
                    + np.arcsin(voltage_acc/voltage)
                    - 2*np.pi*frequencies[ii]*arrival_offsets[ii]
                    - 2*np.pi*position/circumference)
        rf_phases.append(rf_phase)
        line.insert(name, xt.Cavity(voltage=voltage, absolute_time=True), at=position)
        line.functions[f'frequency_{name}'] = xt.FunctionPieceWiseLinear(
            x=times, y=rf_frequency)
        line.functions[f'phase_{name}'] = xt.FunctionPieceWiseLinear(x=times, y=rf_phase)
        clock = line.ref['t_turn_s']
        line[name].frequency = line.functions[f'frequency_{name}'](clock)
        line[name].phase = (line.functions[f'phase_{name}'](clock)
                           - 2*np.pi*line.ref[name].frequency*clock)

    monitor_names = ('pickup_carbon', 'pickup_proton')
    # Fixed zeta slices, indexed by accumulated carbon reference turns.
    reference_turn_grid = cumulative_trapezoid(
        clight/circumference * momentum/np.sqrt(momentum**2 + carbon.mass0**2),
        times, initial=0)
    num_reference_turns = int(np.ceil(np.interp(duration, times, reference_turn_grid)))
    monitors = []
    for name, first_id in zip(monitor_names, (0, PROTON_ID_START)):
        monitor = xt.BeamStatsMonitor(
            start_at_turn=0, stop_at_turn=num_reference_turns + 1,
            coasting=True, num_slices=num_slices,
            coasting_reference_turn=0.,
            particle_id_range=(first_id, first_id + num_particles),
            stats=['num_particles'])
        line.env.elements[name] = monitor
        monitors.append(monitor)
    line.insert(list(monitor_names), at=0)
    # Install the EnergyProgram after lattice insertions, which create
    # intermediate lines sharing the environment's element dictionary.
    line.energy_program = xt.EnergyProgram(t_s=times, p0c=momentum)
    line.functions['reference_turn'] = xt.FunctionPieceWiseLinear(
        x=times, y=reference_turn_grid)
    for name in monitor_names:
        line[name].coasting_reference_turn = line.functions['reference_turn'](line.ref['t_turn_s'])
    st.install_sync_time_at_collective_elements(
        line, frame_clock=True, frame_relative_length=FRAME_FRACTION,
        at_element_names=monitor_names)
    line.enable_time_dependent_vars = True
    # Twiss and initial bunch generation above use the serial context. Move
    # both particles and the tracking line onto the OpenMP context for the run.
    context = xo.ContextCpu(omp_num_threads=num_threads)
    particles.move(_context=context)
    # Compile the updated monitor kernel without replacing the user's cache.
    with xt.settings.override(allow_kernel_compilation=True):
        line.build_tracker(_context=context, use_prebuilt_kernels=False)

    reference_turns = np.interp(duration + periods[0],
        line.energy_program.t_at_turn_interpolator.y,
        line.energy_program.t_at_turn_interpolator.x)
    num_frames = int(np.ceil(reference_turns / FRAME_FRACTION)) + 1
    print(f'Tracking on {context.omp_num_threads} OpenMP threads')
    print(f'Ring length: {circumference:.2f} m')
    print(f'Carbon: 7 MeV/u, f_rev = {frequencies[0]/1e3:.3f} kHz, '
          f'T_rev = {periods[0]*1e6:.4f} us')
    print(f'Proton: {proton.kinetic_energy0[0]/1e6:.4f} MeV, '
          f'f_rev = {frequencies[1]/1e3:.3f} kHz, '
          f'T_rev = {periods[1]*1e6:.4f} us')
    print(f'f_proton / f_carbon = {frequencies[1]/frequencies[0]:.6f}')
    print(f'{num_slices} samples per carbon period '
          f'({periods[0]/num_slices*1e9:.2f} ns bins)')
    print(f'Carbon ramp: 7 -> {final_energy_per_nucleon/1e6:g} MeV/u '
          f'in {ramp_duration*1e6:.1f} us, followed by a flat top')
    print(f'Peak accelerating voltage per charge: {voltage_acc.max():.1f} V')

    history = []
    def record_beams(line, p):
        snapshot = [float(p.t_frame)]
        alive = (p.state > 0) | (p.state < -st.COAST_STATE_RANGE_START)
        arrival = p.t_frame + (p.s - p.zeta)/(p.beta0*clight)
        target_pc = np.interp(arrival, times, momentum)
        rigidity_error = (1 + p.delta)/p.chi*p.p0c/target_pc - 1
        for first_id in (0, PROTON_ID_START):
            mask = (p.particle_id >= first_id) & (p.particle_id < first_id + num_particles)
            live = mask & alive
            # Include paused particles; each retains its physical energy.
            kinetic = (p.energy - p.mass0*p.mass_ratio)[live]
            snapshot.extend([np.count_nonzero(live)/num_particles,
                np.mean(kinetic), np.std(kinetic),
                np.max(np.abs(rigidity_error[live])) if np.any(live) else np.nan])
        history.append(snapshot)
        return 0

    line.track(particles, num_turns=num_frames, with_progress=20,
               log=xt.Log(beam_summary=record_beams))
    record_beams(line, particles)
    history = np.asarray(history)

    # Convert the fixed zeta grid to lab time with the reference-turn/time map.
    # Rows are NOT SyncTime frames or individual particles' turn counters.
    time = monitors[0].time_centers(line_length=circumference,
                                    energy_program=line.energy_program)
    counts = np.stack([np.asarray(mon.num_particles) for mon in monitors])
    turn_edges = (np.arange(counts[0].size + 1)/num_slices - .5)
    time_edges = line.energy_program.get_t_s_at_turn(turn_edges)
    time_edges[turn_edges < 0] = turn_edges[turn_edges < 0]*periods[0]
    dt = np.diff(time_edges).reshape(counts[0].shape)
    currents = counts * np.array([carbon.q0, proton.q0])[:, None, None]
    currents *= elementary_charge / dt
    survivors = []
    bucket_fractions = []
    for ii, first_id in enumerate((0, PROTON_ID_START)):
        mask = ((particles.particle_id >= first_id)
                & (particles.particle_id < first_id + num_particles))
        # Waiting SyncTime particles have reserved negative states; they live.
        alive = (particles.state > 0) | (particles.state < -st.COAST_STATE_RANGE_START)
        survivors.append(np.count_nonzero(mask & alive) / num_particles)
        # At the flat top, compare with the stationary bucket of the own RF.
        # Both cavities remain on: this is a capture diagnostic, not an exact
        # invariant of the driven two-RF system.
        arrival = particles.t_frame + (particles.s-particles.zeta)/(particles.beta0*clight)
        position = (.001, .01)[ii]
        phase = np.interp(arrival, times, rf_phases[ii]) - 2*np.pi*(particles.s-position)/circumference
        species = (carbon, proton)[ii]
        pc = p_end*species.q0/carbon.q0
        energy = np.sqrt(pc**2 + species.mass0**2)
        beta = pc/energy
        eta = tw_c.momentum_compaction_factor - (species.mass0/energy)**2
        bucket_dp = np.sqrt(2*species.q0*voltages[ii]/(np.pi*abs(eta)*beta**2*energy))
        dp = (1 + particles.delta)/particles.chi*particles.p0c/p_end - 1
        in_bucket = (dp/bucket_dp)**2 + np.sin(phase/2)**2 < 1
        bucket_fractions.append(np.count_nonzero(mask & alive & in_bucket)/num_particles)
        turns = particles.at_turn[mask & alive]
        print(f'{("Carbon", "Proton")[ii]}: {survivors[-1]:.1%} surviving, '
              f'{turns.min() if len(turns) else 0}--{turns.max() if len(turns) else 0} '
              f'completed revolutions; final mean kinetic energy '
              f'{history[-1, 2 + 4*ii]/1e6:.6f} MeV, '
              f'rms {history[-1, 3 + 4*ii]/1e3:.3f} keV; '
              f'max rigidity error {history[-1, 4 + 4*ii]:.3g}; '
              f'{bucket_fractions[-1]:.1%} inside own-RF flat-top bucket')

    return dict(line=line, particles=particles, monitors=monitors,
                num_particles_per_species=num_particles, num_threads=num_threads,
                time_s=time, current_A=currents, counts=counts,
                periods_s=periods, frequencies_Hz=frequencies,
                proton_kinetic_energy_eV=proton.kinetic_energy0[0],
                arrival_offsets_s=arrival_offsets,
                num_carbon_turns=num_reference_turns,
                acquisition_duration_s=duration, circumference_m=circumference,
                history=history, ramp_times_s=times, ramp_p0c=momentum,
                species_masses_eV=np.array([carbon.mass0, proton.mass0]),
                species_charge_ratios=np.array([1., charge_ratio]),
                final_energy_per_nucleon_eV=final_energy_per_nucleon,
                survivors=np.array(survivors), bucket_fractions=np.array(bucket_fractions))


def save_data(result, output_dir):
    """Save numerical results for the separate plotting script."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / '004_pimm_carbon_protons.npz',
        time_s=result['time_s'], current_A=result['current_A'],
        counts=result['counts'], periods_s=result['periods_s'],
        frequencies_Hz=result['frequencies_Hz'],
        carbon_kinetic_energy_per_nucleon_eV=CARBON_EKIN_PER_NUCLEON,
        proton_kinetic_energy_eV=result['proton_kinetic_energy_eV'],
        num_carbon_turns=result['num_carbon_turns'], history=result['history'],
        acquisition_duration_s=result['acquisition_duration_s'],
        circumference_m=result['circumference_m'],
        survivors=result['survivors'], bucket_fractions=result['bucket_fractions'],
        ramp_times_s=result['ramp_times_s'], ramp_p0c=result['ramp_p0c'],
        species_masses_eV=result['species_masses_eV'],
        species_charge_ratios=result['species_charge_ratios'],
        final_energy_per_nucleon_eV=result['final_energy_per_nucleon_eV'],
        num_particles_per_species=result['num_particles_per_species'],
        num_threads=result['num_threads'])
    print(f'Data saved in {output_dir.resolve() / "004_pimm_carbon_protons.npz"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-particles', type=int, default=4000)
    parser.add_argument('--num-carbon-turns', type=int, default=256,
                        help='Acquisition duration in injection carbon periods')
    parser.add_argument('--final-energy-mev-u', type=float, default=8.)
    parser.add_argument('--num-slices', type=int, default=512)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).with_name('004_pimm_carbon_protons'))
    parser.add_argument('--num-threads', type=int, default=6)
    args = parser.parse_args()
    if args.num_carbon_turns < 8 or args.num_particles < 1 or args.num_slices < 1:
        parser.error('Use at least 8 carbon periods, 1 particle and 1 slice')
    if args.num_threads < 1:
        parser.error('Use at least one CPU thread')
    result = simulate(args.num_particles, args.num_carbon_turns, args.num_slices,
                      args.final_energy_mev_u*1e6, args.num_threads)
    save_data(result, args.output_dir)
