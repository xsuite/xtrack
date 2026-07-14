import numpy as np
import matplotlib.pyplot as plt

import xtrack as xt


rng = np.random.default_rng(12345)
num_particles = 5000

particles = xt.Particles(
    p0c=7e12,
    x=1.0e-3 * rng.normal(size=num_particles),
    px=np.zeros(num_particles),
    y=np.zeros(num_particles),
    py=np.zeros(num_particles),
    zeta=rng.uniform(-0.3, 0.3, num_particles),
    delta=np.zeros(num_particles),
    weight=np.ones(num_particles),
)

monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=20,
    zeta_range=(-0.3, 0.3),
    num_slices=32,
    stats=["num_particles", "mean_x", "sigma_x"],
)

for _ in range(20):
    monitor.track(particles)

    # Create a coherent motion that depends on turn and longitudinal position.
    particles.x += 0.1e-3 * np.sin(2 * np.pi * particles.zeta / 0.6)
    particles.at_turn += 1

fig, ax = plt.subplots()
pcm = ax.pcolormesh(
    monitor.zeta_centers[0, :],
    monitor.turns,
    monitor.mean_x[:, 0, :],
    shading="auto",
)
fig.colorbar(pcm, ax=ax, label="mean_x [m]")
ax.set_xlabel("zeta [m]")
ax.set_ylabel("turn")
ax.set_title("Slice-by-slice horizontal centroid")
plt.show()
