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

# No zeta_range or num_slices are provided: this records whole-beam statistics.
monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=5,
    every_n_turns=2,
    stats=["num_particles", "mean_x", "sigma_x", "cov_x_px"],
)

for _ in range(5):
    monitor.track(particles)
    particles.x += 0.1e-3
    particles.at_turn += 1

# Inspect what has been recorded.
monitor.stats  # is ('num_particles', 'mean_x', 'sigma_x', 'cov_x_px')
monitor.available_levels  # is ('beam',)
monitor.default_level  # is 'beam'
monitor.turns  # is array([0, 2, 4])

# The shape is (logged turns,).
monitor.mean_x.shape  # is (3,)

# num_particles is the sum of particle weights in the whole beam.
monitor.num_particles  # is array([7.0, 7.0, 7.0])

# A recorded statistic can be accessed as an attribute or with monitor.get().
monitor.mean_x[monitor.record_index(2)]
monitor.get("mean_x", turn=2)  # same as above

# Covariances and sigmas follow the same indexing convention.
monitor.get("sigma_x", turn=2)
monitor.get("cov_x_px", turn=2)
