import xtrack as xt
import xobjects as xo
import numpy as np

env = xt.Environment()
env.particle_ref = xt.Particles(
    mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=2.7e9
)

n_part = 20

# TODO: Insert the proper element here.
test_element = xt.Sietse(Bs=0.5, length=1)

particles = xt.Particles(
    x=np.linspace(-1e-3, 1e-3, n_part),
    px=np.linspace(-1e-3, 1e-3, n_part),
    y=np.linspace(-1e-3, 1e-3, n_part),
    py=np.linspace(-1e-3, 1e-3, n_part),
    zeta=np.zeros(n_part),
    delta=np.zeros(n_part),
)

initial_x = particles.x.copy()
initial_y = particles.y.copy()
initial_px = particles.px.copy()
initial_py = particles.py.copy()
initial_zeta = particles.zeta.copy()
initial_delta = particles.delta.copy()

print("Initial State:")
print(f"initial_x = {initial_x}")
print(f"initial_y = {initial_y}")
print(f"initial_px = {initial_px}")
print(f"initial_py = {initial_py}")
print(f"initial_zeta = {initial_zeta}")
print(f"initial_delta = {initial_delta}")

test_element.track(particles)