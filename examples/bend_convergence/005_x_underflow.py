import xtrack as xt
import numpy as np

length = 14
angle = 2 * np.pi / 1000
k2l = 0


env = xt.Environment()
line = env.new_line(components=[
    env.new('b', 'Bend', length=length, angle=angle,
            model='bend-kick-bend',
            integrator='yoshida4', # is actually yoshida6
            num_multipole_kicks=1)
])
line.set_particle_ref('positron', energy0=1e9)

x_test_list = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-16]

p0 = line.particle_ref.copy()

for x_test in x_test_list:
    p = p0.copy()
    p.x = x_test
    line.track(p)
    print(f"x_test={x_test:.1e} -> dx = {p.x[0] - x_test:.10e}")