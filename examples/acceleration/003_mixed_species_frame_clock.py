"""Fe/Bi acceleration with two RF systems and a common particle clock.

This is a longitudinal demonstration with a fixed path length, not a SIS18
lattice model. Both species see both cavities; there are no collective forces.
Use prebuilt kernels regenerated for this checkout's Particles layout, or run
with XSUITE_FORCE_KERNEL_COMPILATION=1 to compile kernels from source.

The reference momentum and cavity settings are updated once per time frame.
The RF phase is integrated analytically, then linearized within each frame.
For a real lattice, the normalized magnetic strengths should describe the
desired optics as its reference rigidity follows the common ramp.
"""

import numpy as np
from scipy.constants import c as clight

import xtrack as xt
import xtrack.synctime as st


def run(num_frames=900):
    circumference = 216.
    duration = .002
    p0c_start = 20e9
    p0c_rate = .02 * p0c_start / duration
    iron = xt.Particles('Iron-56', q0=25, p0c=p0c_start)
    # Xtrack's species database currently approximates the Bi-209 mass as 209u.
    bismuth = xt.Particles('Bismuth-209', q0=68)
    mass_ratio = bismuth.mass0 / iron.mass0
    charge_ratio = bismuth.q0 / iron.q0

    line = xt.Line(elements={
        'd0': xt.Drift(length=.25*circumference),
        'rf_fe': xt.Cavity(absolute_time=True),
        'd1': xt.Drift(length=.40*circumference),
        'rf_bi': xt.Cavity(absolute_time=True),
        'd2': xt.Drift(length=.35*circumference),
    })
    line.particle_ref = iron
    t = np.linspace(0, duration, 20001)
    p0c = p0c_start + p0c_rate * t
    line.energy_program = xt.EnergyProgram(t_s=t, p0c=p0c)

    # Energy gain per charge per revolution for the prescribed rigidity ramp.
    voltage_acc = circumference / clight * p0c_rate / iron.q0
    for name, species, q_ratio, s_fraction, voltage in (
            ('rf_fe', iron, 1., .25, 15000.),
            ('rf_bi', bismuth, charge_ratio, .65, 18000.)):
        harmonic = 2
        pc = q_ratio * p0c
        energy = np.sqrt(pc**2 + species.mass0**2)
        beta = pc / energy
        frequency = harmonic * beta * clight / circumference
        synchronous_phase = np.arcsin(voltage_acc / voltage)
        # Integral beta(t) dt = (E(t)-E(0)) / d(pc)/dt for a linear pc ramp.
        phase = (2*np.pi*harmonic*clight/circumference
                 * (energy - energy[0]) / (q_ratio*p0c_rate)
                 + synchronous_phase - 2*np.pi*harmonic*s_fraction)
        line.functions[f'f_{name}'] = xt.FunctionPieceWiseLinear(x=t, y=frequency)
        line.functions[f'phi_{name}'] = xt.FunctionPieceWiseLinear(x=t, y=phase)
        clock = line.ref['t_turn_s']
        line[name].frequency = line.functions[f'f_{name}'](clock)
        line[name].phase = (line.functions[f'phi_{name}'](clock)
                           - 2*np.pi*line.ref[name].frequency*clock)
        line[name].voltage = voltage

    st.install_sync_time_at_collective_elements(line)
    line.enable_time_dependent_vars = True
    line.build_tracker()
    # Shared Fe reference, species-dependent mass/charge ratios. Equal rigidity
    # corresponds to delta = chi - 1, not delta = 0 for both species.
    particles = xt.Particles(
        mass0=iron.mass0, q0=iron.q0, p0c=p0c_start,
        mass_ratio=[1., mass_ratio], charge_ratio=[1., charge_ratio],
        delta=[0., charge_ratio/mass_ratio - 1], zeta=0.)
    line.track(particles, num_turns=num_frames, log=xt.Log(
        time=lambda line, p: p.time_s,
        arrival_time=lambda line, p: (
            p.time_s + (p.s - p.zeta) / (p.beta0 * clight)),
        energy=lambda line, p: p.energy.copy(),
        particle_id=lambda line, p: p.particle_id.copy()))
    # Snapshots can contain temporarily waiting particles with negative states.
    for ii in np.argsort(particles.particle_id):
        name = ('Fe', 'Bi')[particles.particle_id[ii]]
        arrival_time = (particles.time_s
                        + (particles.s[ii] - particles.zeta[ii])
                        / (particles.beta0[ii] * clight))
        target_p0c = p0c_start + p0c_rate * arrival_time
        rigidity_error = ((1 + particles.delta[ii]) / particles.chi[ii]
                          * particles.p0c[ii] / target_p0c - 1)
        print(f'{name}: turns={particles.at_turn[ii]}, '
              f'rigidity error={rigidity_error:+.6e}')
    print(f'Common clock: {particles.time_s:.9e} s, '
          f'{num_frames} frames')
    return line, particles


if __name__ == '__main__':
    run()
