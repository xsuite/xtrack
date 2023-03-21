import json

import xtrack as xt
import xpart as xp

fname_line = '../../test_data/lhc_no_bb/line_and_particle.json'

# import a line and add reference particle
with open(fname_line) as fid:
    line_dict = json.load(fid)
line = xt.Line.from_dict(line_dict['line'])
line.particle_ref = xp.Particles.from_dict(line_dict['particle'])

# Build the tracker
line.build_tracker()

# Track some particles with frozen longitudinal coordinates
particles = line.build_particles(delta=1e-3, x=[-1e-3, 0, 1e-3])
line.track(particles, num_turns=10, freeze_longitudinal=True)
print(particles.delta) # gives [0.001 0.001 0.001], same as initial value

# Twiss with frozen longitudinal coordinates (needs to be 4d)
twiss = line.twiss(method='4d', freeze_longitudinal=True)
print(twiss.slip_factor) # gives 0 (no longitudinal motion)

# Track some particles with unfrozen longitudinal coordinates
particles = line.build_particles(delta=1e-3, x=[-1e-3, 0, 1e-3])
line.track(particles, num_turns=10)
print(particles.delta) # gives [0.00099218, ...], different from initial value

# Twiss with unfrozen longitudinal coordinates (can be 6d)
twiss = line.twiss(method='6d')
print(twiss.slip_factor) # gives 0.00032151, from longitudinal motion
