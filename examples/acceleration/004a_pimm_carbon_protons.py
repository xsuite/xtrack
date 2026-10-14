"""Two bunches racing in PIMM: C-12(6+) at 7 MeV/u and equal-rigidity protons.

A common rigidity ramp, two chirped h=1 RF systems, and no collective forces. Both
species see both cavities. Separate coasting BeamStatsMonitors select the two
particle-id ranges and reconstruct an ideal charge-current pickup in lab time.
Coasting here describes the monitor's full-period acquisition, not the beam:
the two beams are bunched.

By default carbon accelerates from 7 to 9 MeV/u in about 527 us. The energy
slope turns on smoothly over the first 20% of the acquisition, then stays
constant. Protons follow the same magnetic rigidity. The
normalized magnet strengths stay fixed, and both RF frequencies and their
integrated phases follow the ramp. The RF voltages are 40 kV and 30 kV.
Pickup slices have fixed zeta width; their lab-time widths follow the ramp.
The same monitors record longitudinal moments for the energy-versus-time plot.
The initial rms bunch duration is 2% of the injection carbon period
(about 41 ns); the momentum spread uses the small-amplitude RF matching.
A horizontal +/-10 cm aperture at the entrance of qfb.3 (Dx about 8.34 m)
removes particles whose dispersive and betatron excursions exceed the limit.

Examples (paths are independent of the working directory)::

    python 004a_pimm_carbon_protons.py
    # In IPython (line, particles, monitors and arrays remain available):
    %run 004a_pimm_carbon_protons.py
    python 004b_plot_pimm_carbon_protons.py

Simulation uses six OpenMP threads by default and saves an NPZ file. Run the
separate plotting script to display or export figures. On macOS, source
compilation needs an OpenMP-enabled compiler (set CC/CXX in your environment).

The PIMM test lattice is a demonstration model, not a CNAO/MedAustron model.
Their 7 MeV/u injection energy motivates the carbon energy used here; equal
rigidity requires protons at about 27.5 MeV, not their usual injection energy.
"""

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


# Edit these settings before running the script (or use %run in IPython).
num_particles = 4000  # Per species
num_carbon_turns = 256  # Acquisition duration in injection carbon periods
num_slices = 512
final_energy_per_nucleon = 9e6  # eV/u at the end of the acquisition
turn_on_fraction = .2
num_threads = 6
output_dir = Path(__file__).with_name('004_pimm_carbon_protons')

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
turn_on_duration = turn_on_fraction * duration
times = np.linspace(0, duration + 8*periods[0], 20001)
# A raised-cosine energy slope joins zero smoothly to a constant slope.
# Its integral is t/2 - tau*sin(pi*t/tau)/(2*pi) during turn-on, then
# t - tau/2. Normalize to reach the requested energy at t=duration.
# Continue the linear ramp through the short tracking/acquisition margin.
turn_on = np.minimum(times/turn_on_duration, 1.)
ramp_time = times - .5*turn_on_duration*(turn_on + np.sin(np.pi*turn_on)/np.pi)
energy_slope = 12*(final_energy_per_nucleon - CARBON_EKIN_PER_NUCLEON) / (
    duration - .5*turn_on_duration)
kinetic_energy = 12*CARBON_EKIN_PER_NUCLEON + energy_slope*ramp_time
kinetic_energy_rate = energy_slope * .5*(1 - np.cos(np.pi*turn_on))
momentum = np.sqrt(kinetic_energy*(kinetic_energy + 2*carbon.mass0))
momentum_rate = (kinetic_energy + carbon.mass0)/momentum * kinetic_energy_rate
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
sigma_t = .02 * periods[0]
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
aperture_name = 'dispersive_aperture'
aperture_s = float(tw_c['s', 'qfb.3'])
aperture_dx = float(tw_c['dx', 'qfb.3'])
aperture_half_width = .10
line.insert(aperture_name,
            xt.LimitRect(min_x=-aperture_half_width, max_x=aperture_half_width),
            at=aperture_s)
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
        particle_id_range=(first_id, first_id + num_particles),
        stats=['num_particles', 'mean_pzeta', 'sigma_pzeta'])
    line.env.elements[name] = monitor
    monitors.append(monitor)
line.insert(list(monitor_names), at=0)
# Install the EnergyProgram after lattice insertions, which create
# intermediate lines sharing the environment's element dictionary.
line.energy_program = xt.EnergyProgram(t_s=times, p0c=momentum)
st.install_sync_time_at_collective_elements(
    line, frame_relative_length=FRAME_FRACTION,
    at_element_names=monitor_names)
line.enable_time_dependent_vars = True
# Twiss and initial bunch generation above use the serial context. Move
# both particles and the tracking line onto the OpenMP context for the run.
context = xo.ContextCpu(omp_num_threads=num_threads)
particles.move(_context=context)
line.build_tracker(_context=context)

reference_turns = line.energy_program.get_turn_at_t_s(duration + periods[0])
num_frames = int(np.ceil(reference_turns / FRAME_FRACTION)) + 1
print(f'Tracking on {context.omp_num_threads} OpenMP threads')
print(f'Ring length: {circumference:.2f} m')
print(f'Horizontal aperture: +/-{aperture_half_width*100:g} cm at '
      f's={aperture_s:.4f} m, Dx={aperture_dx:.3f} m '
      f'(about +/-{aperture_half_width/abs(aperture_dx):.2%} rigidity acceptance)')
print(f'Carbon: 7 MeV/u, f_rev = {frequencies[0]/1e3:.3f} kHz, '
      f'T_rev = {periods[0]*1e6:.4f} us')
print(f'Proton: {proton.kinetic_energy0[0]/1e6:.4f} MeV, '
      f'f_rev = {frequencies[1]/1e3:.3f} kHz, '
      f'T_rev = {periods[1]*1e6:.4f} us')
print(f'f_proton / f_carbon = {frequencies[1]/frequencies[0]:.6f}')
print(f'{num_slices} samples per carbon period '
      f'({periods[0]/num_slices*1e9:.2f} ns bins)')
print(f'Carbon ramp: 7 -> {final_energy_per_nucleon/1e6:g} MeV/u '
      f'in {duration*1e6:.1f} us; smooth turn-on over {turn_on_duration*1e6:.1f} us')
print(f'Peak accelerating voltage per charge: {voltage_acc.max():.1f} V')

line.track(particles, num_turns=num_frames, with_progress=20)

# Convert the fixed zeta grid to lab time with the reference-turn/time map.
# Rows are NOT SyncTime frames or individual particles' turn counters.
time = monitors[0].time_centers(line_length=circumference,
                                energy_program=line.energy_program)
counts = np.stack([np.asarray(mon.num_particles) for mon in monitors])
mean_pzeta = np.stack([np.asarray(mon.mean_pzeta) for mon in monitors])
sigma_pzeta = np.stack([np.asarray(mon.sigma_pzeta) for mon in monitors])
# This moment is always recorded by BeamStatsMonitor. Use the reference
# momentum seen by the particles, rather than evaluating the ramp at bin centers.
sum_p0c = np.stack([np.asarray(mon.data.sum_beta0_gamma0).reshape(time.shape)
                    for mon in monitors]) * carbon.mass0
reference_p0c = np.full_like(counts, np.nan)
np.divide(sum_p0c, counts, out=reference_p0c, where=counts > 0)
# Own-RF phase identifies successive bunch passages in the plotting script.
pickup_rf_phase = np.stack([np.interp(time, times, phase) + 2*np.pi*position/circumference
                           for phase, position in zip(rf_phases, (.001, .01))])
turn_edges = (np.arange(counts[0].size + 1)/num_slices - .5)
time_edges = line.energy_program.get_t_s_at_turn(turn_edges)
time_edges[turn_edges < 0] = turn_edges[turn_edges < 0]*periods[0]
dt = np.diff(time_edges).reshape(counts[0].shape)
currents = counts * np.array([carbon.q0, proton.q0])[:, None, None]
currents *= elementary_charge / dt
survivors = []
aperture_losses = []
aperture_index = line.element_names.index(aperture_name)
for ii, first_id in enumerate((0, PROTON_ID_START)):
    mask = ((particles.particle_id >= first_id)
            & (particles.particle_id < first_id + num_particles))
    # Waiting SyncTime particles have reserved negative states; they live.
    alive = (particles.state > 0) | (particles.state < -st.COAST_STATE_RANGE_START)
    survivors.append(np.count_nonzero(mask & alive) / num_particles)
    lost_here = (particles.state == 0) & (particles.at_element == aperture_index)
    aperture_losses.append(np.count_nonzero(mask & lost_here)/num_particles)
    turns = particles.at_turn[mask & alive]
    kinetic = (particles.energy - particles.mass0*particles.mass_ratio)[mask & alive]
    print(f'{("Carbon", "Proton")[ii]}: {survivors[-1]:.1%} surviving, '
          f'{turns.min() if len(turns) else 0}--{turns.max() if len(turns) else 0} '
          f'completed revolutions; final mean kinetic energy '
          f'{np.mean(kinetic)/1e6:.6f} MeV, '
          f'rms {np.std(kinetic)/1e3:.3f} keV; '
          f'{aperture_losses[-1]:.1%} lost at the dispersive aperture')

# Save numerical results for the separate plotting script.
output_dir.mkdir(parents=True, exist_ok=True)
np.savez_compressed(output_dir / '004_pimm_carbon_protons.npz',
    time_s=time, current_A=currents,
    counts=counts, periods_s=periods,
    frequencies_Hz=frequencies,
    carbon_kinetic_energy_per_nucleon_eV=CARBON_EKIN_PER_NUCLEON,
    proton_kinetic_energy_eV=proton.kinetic_energy0[0],
    num_carbon_turns=num_reference_turns,
    mean_pzeta=mean_pzeta, sigma_pzeta=sigma_pzeta,
    reference_p0c_eV=reference_p0c, pickup_rf_phase_rad=pickup_rf_phase,
    acquisition_duration_s=duration,
    circumference_m=circumference,
    survivors=np.array(survivors),
    ramp_times_s=times, ramp_p0c=momentum,
    species_masses_eV=np.array([carbon.mass0, proton.mass0]),
    species_charge_ratios=np.array([1., charge_ratio]),
    final_energy_per_nucleon_eV=final_energy_per_nucleon,
    aperture_s_m=aperture_s, aperture_dx_m=aperture_dx,
    aperture_half_width_m=aperture_half_width,
    aperture_losses=np.array(aperture_losses),
    num_particles_per_species=num_particles,
    num_threads=num_threads)
print(f'Data saved in {output_dir.resolve() / "004_pimm_carbon_protons.npz"}')
