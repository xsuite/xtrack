import numpy as np

import xtrack as xt


particles = xt.Particles(
    p0c=7e12,
    x=[1.0e-3, 2.0e-3, 4.0e-3, 5.0e-3],
    px=[1.0e-6, 2.0e-6, 4.0e-6, 5.0e-6],
    y=[0.0, 0.0, 0.0, 0.0],
    py=[0.0, 0.0, 0.0, 0.0],
    zeta=[-0.12, -0.04, 0.04, 0.12],
    delta=[0.0, 0.0, 0.0, 0.0],
    weight=[2.0, 1.0, 1.0, 3.0],
)

monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=5,
    every_n_turns=2,
    zeta_range=(-0.15, 0.15),
    num_slices=3,
    stats=["num_particles", "mean_x", "sigma_x", "cov_x_px"],
)

for _ in range(5):
    monitor.track(particles)
    particles.at_turn += 1

# The first axis of each recorded array corresponds to monitor.turns.
assert np.all(monitor.turns == [0, 2, 4])
assert monitor.mean_x.shape == (3, 1, 3)

# num_particles is the sum of particle weights in each bin.
print("recorded statistics:", monitor.stats)
print("recorded turns:", monitor.turns)
print("zeta centers:", monitor.zeta_centers[0, :])
print("num_particles at first recorded turn:", monitor.num_particles[0, 0, :])

# A recorded statistic can be accessed as an attribute or with monitor.get().
i_record = np.where(monitor.turns == 2)[0][0]
print("mean_x at turn 2:", monitor.mean_x[i_record, 0, :])
print("mean_x at turn 2:", monitor.get("mean_x")[i_record, 0, :])
