import numpy as np
import xtrack as xt
import xdeps as xd

# Define elements
pi = np.pi
lbend = 3
elements = {
    'mqf.1': xt.Quadrupole(length=0.3, k1=0.1),
    'd1.1':  xt.Drift(length=1),
    'mb1.1': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd2.1':  xt.Drift(length=1),

    'mqd.1': xt.Quadrupole(length=0.3, k1=-0.7),
    'd3.1':  xt.Drift(length=1),
    'mb2.1': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),

    # Three dipoles to make a closed orbit bump
    'd4.1':  xt.Drift(length=0.05),
    'bumper_0':  xt.Bend(length=0.05, k0=0, h=0),
    'd5.1':  xt.Drift(length=0.3),
    'bumper_1':  xt.Bend(length=0.05, k0=0, h=0),
    'd6.1':  xt.Drift(length=0.3),
    'bumper_2':  xt.Bend(length=0.05, k0=0, h=0),
    'd7.1':  xt.Drift(length=0.2),

    'mqf.2': xt.Quadrupole(length=0.3, k1=0.1),
    'd1.2':  xt.Drift(length=1),
    'mb1.2': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd2.2':  xt.Drift(length=1),

    'mqd.2': xt.Quadrupole(length=0.3, k1=-0.7),
    'd3.2':  xt.Drift(length=1),
    'mb2.2': xt.Bend(length=lbend, k0=pi / 2 / lbend, h=pi / 2 / lbend),
    'd4.2':  xt.Drift(length=1),
}

# Build the ring
line = xt.Line(elements=elements,
               element_names=['mqf.1', 'd1.1', 'mb1.1', 'd2.1', # defines the order
                              'mqd.1', 'd3.1', 'mb2.1', 'd4.1',
                               'bumper_0', 'd5.1',
                               'bumper_1', 'd6.1',
                               'bumper_2', 'd7.1',
                              'mqf.2', 'd1.2', 'mb1.2', 'd2.2',
                              'mqd.2', 'd3.2', 'mb2.2', 'd4.2'])
# Define reference particle
kin_energy_0 = 50e6 # 50 MeV
line.particle_ref = xt.Particles(energy0=kin_energy_0 + xt.PROTON_MASS_EV, # total energy
                                 mass0=xt.PROTON_MASS_EV)

# Power the correctors to make a closed orbit bump
line.vars['bumper_strength'] = 0.
line.element_refs['bumper_0'].k0 = -line.vars['bumper_strength']
line.element_refs['bumper_1'].k0 = 2 * line.vars['bumper_strength']
line.element_refs['bumper_2'].k0 = -line.vars['bumper_strength']

line.vars['bumper_strength'] = 0.1

tw = line.twiss(method='4d')

T_sin = 100e-6
sin = line.functions.sin
line.vars['bumper_strength'] = (0.1 * sin(2 * np.pi / T_sin * line.vars['t_turn_s']))

# probe behavior with twiss
t_test = np.linspace(0, 100e-6, 20)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1, figsize=(6.4*1.5, 4.8))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
for ii, tt, in enumerate(t_test):
    line.vars['t_turn_s'] = tt
    tw_tt = line.twiss(method='4d')
    ax1.plot(tt * 1e6, tw_tt['x', 'bumper_1'] * 1e3, 'o',
             color=plt.cm.jet(ii / len(t_test)))
    ax2.plot(tw_tt.s, tw_tt.x * 1e3,
             color=plt.cm.jet(ii / len(t_test)), lw=2)

ax2.set_xlim(tw_tt['s', 'bumper_0'] - 1, tw_tt['s', 'bumper_2'] + 1)

for cc in ['bumper_0', 'bumper_1', 'bumper_2']:
    l_corr = line[cc].length
    ax2.axvspan(tw_tt['s', cc], (tw_tt['s', cc] + l_corr),
                color='k', alpha=0.2)
# --- Track particles with time-dependent bumpers ---
num_particles = 100
num_turns = 1000

# Install monitor in the middle of the bump
monitor = xt.ParticlesMonitor(num_particles=num_particles,
                              start_at_turn=0,
                              stop_at_turn =num_turns)
line.discard_tracker()
line.insert_element('monitor', monitor, index='bumper_1')
line.build_tracker()

# Generate particles
particles = line.build_particles(
                        x=np.random.uniform(-0.1e-3, 0.1e-3, num_particles),
                        px=np.random.uniform(-0.1e-3, 0.1e-3, num_particles))

# Enable time-dependent variables
line.enable_time_dependent_vars = True

# Track
line.track(particles, num_turns=num_turns, with_progress=True)

plt.figure(2)
plt.plot(1e6 * monitor.at_turn.T * tw.T_rev0, 1e3 * monitor.x.T,
         lw=1, color='r', alpha=0.05)
plt.plot(1e6 * monitor.at_turn[0, :] * tw.T_rev0, 1e3 * monitor.x.mean(axis=0),
         lw=2, color='k')
plt.xlabel('Time [us]')
plt.ylabel('x [mm]')

plt.show()


