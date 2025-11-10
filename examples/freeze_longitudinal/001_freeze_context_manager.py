import json

import xtrack as xt

# import a line and add reference particle
# import a line and add reference particle
line = xt.load(
    '../../test_data/hllhc15_noerrors_nobb/line_w_knobs_and_particle.json')
line.set_particle_ref('proton', p0c=7e12)

# Perform a set of operations with frozen longitudinal coordinates
with xt.freeze_longitudinal(line):
    # Track some particles with frozen longitudinal coordinates
    particles = line.build_particles(delta=1e-3, x=[-1e-3, 0, 1e-3])
    line.track(particles, num_turns=10)
    print(particles.delta) # gives [0.001 0.001 0.001], same as initial value

    # Twiss with frozen longitudinal coordinates (needs to be 4d)
    twiss = line.twiss(method='4d')
    print(twiss.slip_factor) # gives 0 (no longitudinal motion)

# 6d tracking is automatically restored when the with block is exited

# Track some particles with unfrozen longitudinal coordinates
particles = line.build_particles(delta=1e-3, x=[-1e-3, 0, 1e-3])
line.track(particles, num_turns=10)
print(particles.delta) # gives [0.00099218, ...], different from initial value

# Twiss with unfrozen longitudinal coordinates (can be 6d)
twiss = line.twiss(method='6d')
print(twiss.slip_factor) # gives 0.00032151, from longitudinal motion