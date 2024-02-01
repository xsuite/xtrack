import numpy as np

import xobjects as xo
import xtrack as xt

## Generate a simple line
line = xt.Line(
    elements=[xt.Drift(length=2.),
              xt.Multipole(knl=[0, 0.5], ksl=[0,0]),
              xt.Drift(length=1.),
              xt.Multipole(knl=[0, -0.5], ksl=[0,0])],
    element_names=['drift_0', 'quad_0', 'drift_1', 'quad_1'])

## Attach a reference particle to the line (optional)
## (defines the reference mass, charge and energy)
line.particle_ref = xt.Particles(p0c=6500e9, #eV
                                 q0=1, mass0=xt.PROTON_MASS_EV)

## Choose a context
context = xo.ContextCpu()         # For CPU
# context = xo.ContextCupy()      # For CUDA GPUs
# context = xo.ContextPyopencl()  # For OpenCL GPUs

## Transfer lattice on context and compile tracking code
line.build_tracker(_context=context)

## Build particle object on context
n_part = 20
particles = line.build_particles(
                        x=np.random.uniform(-1e-3, 1e-3, n_part),
                        px=np.random.uniform(-1e-5, 1e-5, n_part),
                        y=np.random.uniform(-2e-3, 2e-3, n_part),
                        py=np.random.uniform(-3e-5, 3e-5, n_part),
                        zeta=np.random.uniform(-1e-2, 1e-2, n_part),
                        delta=np.random.uniform(-1e-4, 1e-4, n_part))
# Reference mass, charge, energy are taken from the reference particle.
# Particles are allocated on the context chosen for the line.

## Track (no saving of turn-by-turn data)
n_turns = 100
line.track(particles, num_turns=n_turns)

particles.state # > 0 for particles still alive
particles.at_turn # turn number (for lost particles, it is the turn of loss)
particles.x # x position after tracking
particles.px # x momentum after tracking
# etc...

## Track (saving turn-by-turn data)
particles = line.build_particles( # fresh particles
                        x=np.random.uniform(-1e-3, 1e-3, n_part),
                        px=np.random.uniform(-1e-5, 1e-5, n_part),
                        y=np.random.uniform(-2e-3, 2e-3, n_part),
                        py=np.random.uniform(-3e-5, 3e-5, n_part),
                        zeta=np.random.uniform(-1e-2, 1e-2, n_part),
                        delta=np.random.uniform(-1e-4, 1e-4, n_part))
n_turns = 100
line.track(particles, num_turns=n_turns,
              turn_by_turn_monitor=True)

## Turn-by-turn data is available at:
line.record_last_track.x
line.record_last_track.px
# etc...