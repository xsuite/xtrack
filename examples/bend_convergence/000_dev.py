import xtrack as xt
from time import perf_counter

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

p0 = line.build_particles(x=1e-5)

num_turns = 1000

p = p0.copy()
t0 = perf_counter()
line.track(p, num_turns=num_turns)
t_track_default = perf_counter() - t0

print(f'Track time with default integrator: {t_track_default/num_turns:.3e} s per turn')

line.configure_bend_model(core='rot-kick-rot-low-order')
p = p0.copy()
t0 = perf_counter()
line.track(p, num_turns=num_turns)
t_track_rot_kick_rot = perf_counter() - t0
print(f'Track time with rot-kick-rot-low-order: {t_track_rot_kick_rot/num_turns:.3e} s per turn')

line.configure_bend_model(core='bend-kick-bend')
p = p0.copy()
t0 = perf_counter()
line.track(p, num_turns=num_turns)
t_track_bend_kick_bend = perf_counter() - t0
print(f'Track time with bend-kick-bend: {t_track_bend_kick_bend/num_turns:.3e} s per turn')

line.configure_bend_model(core='rot-kick-rot')
p = p0.copy()
t0 = perf_counter()
line.track(p, num_turns=num_turns)
t_track_rot_kick_rot = perf_counter() - t0
print(f'Track time with rot-kick-rot: {t_track_rot_kick_rot/num_turns:.3e} s per turn')
