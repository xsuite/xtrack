import numpy as np
import xtrack as xt

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

    # 'd4.1':  xt.Drift(length=1),
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

line.build_tracker()

line.vars['bumper_strength'] = 0.
line.element_refs['bumper_0'].k0 = -line.vars['bumper_strength']
line.element_refs['bumper_1'].k0 = 2 * line.vars['bumper_strength']
line.element_refs['bumper_2'].k0 = -line.vars['bumper_strength']

line.vars['bumper_strength'] = 0.1

tw = line.twiss(method='4d')



# Control the bumpers with a sinusoidal function of time
T_sin = 100e-6
sin = line.functions.sin
line.vars['bumper_strength'] = 0.1 * sin(2 * np.pi / T_sin * line.vars['t_turn_s'])

num_particles = 100
particles = line.build_particles(
                        x=np.random.uniform(-0.1e-3, 0.1e-3, num_particles),
                        px=np.random.uniform(-0.1e-3, 0.1e-3, num_particles))

num_turns = 1000

monitor = xt.ParticlesMonitor(num_particles=num_particles,
                              start_at_turn=0,
                              stop_at_turn =num_turns)
line.discard_tracker()
line.insert_element('monitor', monitor, index='bumper_1')
line.build_tracker()

line.enable_time_dependent_vars = True
line.track(particles, num_turns=num_turns, with_progress=True)

import matplotlib.pyplot as plt
plt.close('all')
plt.plot(monitor.x.T, lw=1, color='k', alpha=0.005)
plt.plot(monitor.x.mean(axis=0), lw=2, color='k')

plt.show()


