import xtrack as xt
import numpy as np

line = xt.load('../../test_data/sps_w_spacecharge/line_no_spacecharge_and_particle.json')
tw = line.twiss4d()

n_particles = 1_000
n_turns = 100

p = line.build_particles(x=np.linspace(-1e-5, 1e-5, n_particles),
                     delta=np.linspace(-1e-5, 1e-5, n_particles))

line.track(p, num_turns=n_turns, with_progress=10, time=True)
print(f'Average time: {line.time_last_track/n_particles/n_turns*1e6:.3f} us/particle/turn')

line_opt = line.copy()
line_opt.build_tracker()
line_opt.optimize_for_tracking()
line.build_tracker()
p = line_opt.build_particles(x=np.linspace(-1e-5, 1e-5, n_particles),
                     delta=np.linspace(-1e-5, 1e-5, n_particles))

line_opt.track(p, num_turns=n_turns, with_progress=10, time=True)
print(f'Average time: {line_opt.time_last_track/n_particles/n_turns*1e6:.3f} us/particle/turn')
