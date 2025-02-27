import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt

# Build a simple ring
env = xt.Environment()
pi = np.pi
lbend = 3
line = env.new_line(components=[
    # Three dipoles to make a closed orbit bump
    env.new('d0.1',  xt.Drift,  length=0.05),
    env.new('bumper_0',  xt.Bend, length=0.05, k0=0, h=0),
    env.new('d0.2',  xt.Drift, length=0.3),
    env.new('bumper_1',  xt.Bend, length=0.05, k0=0, h=0),
    env.new('d0.3',  xt.Drift, length=0.3),
    env.new('bumper_2',  xt.Bend, length=0.05, k0=0, h=0),
    env.new('d0.4',  xt.Drift, length=0.2),

    # Simple ring with two FODO cells
    env.new('mqf.1', xt.Quadrupole, length=0.3, k1=0.1),
    env.new('d1.1',  xt.Drift, length=1),
    env.new('mb1.1', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d2.1',  xt.Drift, length=1),
    env.new('mqd.1', xt.Quadrupole, length=0.3, k1=-0.7),
    env.new('d3.1',  xt.Drift, length=1),
    env.new('mb2.1', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d3.4',  xt.Drift, length=1),
    env.new('mqf.2', xt.Quadrupole, length=0.3, k1=0.1),
    env.new('d1.2',  xt.Drift, length=1),
    env.new('mb1.2', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    env.new('d2.2',  xt.Drift, length=1),
    env.new('mqd.2', xt.Quadrupole, length=0.3, k1=-0.7),
    env.new('d3.2',  xt.Drift, length=1),
    env.new('mb2.2', xt.Bend, length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
])

kin_energy_0 = 50e6 # 50 MeV
line.particle_ref = xt.Particles(energy0=kin_energy_0 + xt.PROTON_MASS_EV, # total energy
                                 mass0=xt.PROTON_MASS_EV)

# Twiss
tw = line.twiss(method='4d')

# Power the correctors to make a closed orbit bump
line.vars['bumper_strength'] = 0.
line.element_refs['bumper_0'].k0 = -line.vars['bumper_strength']
line.element_refs['bumper_1'].k0 = 2 * line.vars['bumper_strength']
line.element_refs['bumper_2'].k0 = -line.vars['bumper_strength']

# Drive the correctors with a sinusoidal function
T_sin = 100e-6
sin = line.functions.sin
line.vars['bumper_strength'] = (0.1 * sin(2 * np.pi / T_sin * line.vars['t_turn_s']))


# --- Probe behavior with twiss at different t_turn_s ---

t_test = np.linspace(0, 100e-6, 15)
tw_list = []
bumper_0_list = []
for tt in t_test:
    line.vars['t_turn_s'] = tt
    bumper_0_list.append(line.element_refs['bumper_0'].k0) # Inspect bumper
    tw_list.append(line.twiss(method='4d')) # Twiss

# Plot
plt.close('all')
plt.figure(1, figsize=(6.4*1.2, 4.8*0.85))
ax1 = plt.subplot(1, 2, 1, xlabel='Time [us]', ylabel='x at bump center [mm]')
ax2 = plt.subplot(1, 2, 2, xlabel='s [m]', ylabel='x [mm]')

colors = plt.cm.jet(np.linspace(0, 1, len(t_test)))

for ii, tt in enumerate(t_test):
    tw_tt = tw_list[ii]
    ax1.plot(tt * 1e6, tw_tt['x', 'bumper_1'] * 1e3, 'o', color=colors[ii])
    ax2.plot(tw_tt.s, tw_tt.x * 1e3, color=colors[ii], label=f'{tt * 1e6:.1f} us')

ax2.set_xlim(0, tw_tt['s', 'bumper_2'] + 1)
plt.subplots_adjust(left=.1, right=.97, top=.92, wspace=.27)
ax2.legend()


# --- Track particles with time-dependent bumpers ---

num_particles = 100
num_turns = 1000

# Install monitor in the middle of the bump
monitor = xt.ParticlesMonitor(num_particles=num_particles,
                              start_at_turn=0,
                              stop_at_turn =num_turns)
line.discard_tracker()
line.insert('monitor', monitor, at='bumper_1@start')
line.build_tracker()

# Generate particles
particles = line.build_particles(
                        x=np.random.uniform(-0.1e-3, 0.1e-3, num_particles),
                        px=np.random.uniform(-0.1e-3, 0.1e-3, num_particles))

# Enable time-dependent variables
line.enable_time_dependent_vars = True

# Track
line.track(particles, num_turns=num_turns, with_progress=True)

# Plot
plt.figure(2, figsize=(6.4*0.8, 4.8*0.8))
plt.plot(1e6 * monitor.at_turn.T * tw.T_rev0, 1e3 * monitor.x.T,
         lw=1, color='r', alpha=0.05)
plt.plot(1e6 * monitor.at_turn[0, :] * tw.T_rev0, 1e3 * monitor.x.mean(axis=0),
         lw=2, color='k')
plt.xlabel('Time [us]')
plt.ylabel('x [mm]')
plt.show()
