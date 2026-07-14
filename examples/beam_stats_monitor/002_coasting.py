import xtrack as xt


circumference = 100.0

particles = xt.Particles(
    p0c=7e12,
    x=[1.0e-3, 2.0e-3, 4.0e-3, 8.0e-3],
    px=[0.0, 0.0, 0.0, 0.0],
    y=[0.0, 0.0, 0.0, 0.0],
    py=[0.0, 0.0, 0.0, 0.0],
    zeta=[-35.0, -10.0, 10.0, 35.0],
    delta=[0.0, 0.0, 0.0, 0.0],
    weight=[1.0, 2.0, 3.0, 4.0],
)

monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1,
    coasting=True,
    zeta_range=(-circumference / 2, circumference / 2),
    num_slices=4,
    stats=["num_particles", "mean_x"],
)

monitor.track(particles)

# In coasting mode the artificial bunch axis is hidden by default.
assert monitor.num_particles.shape == (1, 4)
assert monitor.get("num_particles", turn=0).shape == (4,)

print("zeta centers:", monitor.zeta_centers)
print("public shape:", monitor.num_particles.shape)
print("num_particles:", monitor.num_particles[0, :])
print("mean_x:", monitor.mean_x[0, :])
