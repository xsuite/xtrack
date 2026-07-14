import numpy as np

import xtrack as xt


rng = np.random.default_rng(12345)
num_particles = 2000

particles = xt.Particles(
    p0c=7e12,
    x=1.0e-3 * rng.normal(size=num_particles),
    px=2.0e-6 * rng.normal(size=num_particles),
    y=0.8e-3 * rng.normal(size=num_particles),
    py=1.5e-6 * rng.normal(size=num_particles),
    zeta=0.05 * rng.normal(size=num_particles),
    delta=1.0e-4 * rng.normal(size=num_particles),
    weight=np.ones(num_particles),
)

monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1,
    zeta_range=(-0.3, 0.3),
    num_slices=1,
    stats=[
        "num_particles",
        "gemitt_x_projected",
        "nemitt_x_projected",
        "gemitt_y_projected",
        "nemitt_y_projected",
    ],
)

monitor.track(particles)

print("recorded statistics:", monitor.stats)
print("num_particles:", monitor.num_particles[0, 0, 0])
print("gemitt_x_projected:", monitor.gemitt_x_projected[0, 0, 0])
print("nemitt_x_projected:", monitor.nemitt_x_projected[0, 0, 0])
print("gemitt_y_projected:", monitor.get("gemitt_y_projected")[0, 0, 0])
