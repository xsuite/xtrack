import xtrack as xt


# Build a minimal FODO cell with the Environment interface.
env = xt.Environment()
env.new("qf", xt.Quadrupole, length=0.2, k1=0.2)
env.new("qd", xt.Quadrupole, length=0.2, k1=-0.2)
env.new("d1", xt.Drift, length=1.0)
env.new("d2", xt.Drift, length=1.0)

line = env.new_line(name="fodo", components=["qf", "d1", "qd", "d2"])
line.set_particle_ref("proton", p0c=7e12)

monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=5,
    every_n_turns=2,
    zeta_range=(-0.15, 0.15),
    num_slices=3,
    stats=["num_particles", "mean_x", "sigma_x", "cov_x_px"],
)

# Insert the monitor immediately before the defocusing quadrupole.
line.insert("beam_stats_at_qd", monitor, at="qd@start")

particles = xt.Particles(
    p0c=7e12,
    x=[1.0e-3, 2.0e-3, 4.0e-3, 5.0e-3],
    px=[1.0e-6, 2.0e-6, 4.0e-6, 5.0e-6],
    zeta=[-0.12, -0.04, 0.04, 0.12],
    weight=[2.0, 1.0, 1.0, 3.0],
)

line.track(particles, num_turns=5)

# The monitor is an element of the line and can be retrieved by name.
line["beam_stats_at_qd"] is monitor  # is True

# Inspect what has been recorded.
monitor.stats  # is ('num_particles', 'mean_x', 'sigma_x', 'cov_x_px')
monitor.turns  # is array([0, 2, 4])
monitor.selected_slots  # is array([0])
monitor.zeta_centers  # is array([[-0.1, 0.0, 0.1]])

# The shape is (logged turns, selected bunch slots, slices).
monitor.mean_x.shape  # is (3, 1, 3)

# num_particles is the sum of particle weights in each slice.
monitor.num_particles[0, 0, :]  # is array([2.0, 2.0, 3.0])

# A recorded statistic can be accessed as an attribute or with monitor.get().
monitor.mean_x[monitor.record_index(2), monitor.slot_index(0), :]
monitor.get("mean_x", turn=2, slot=0)  # same as above

# Covariances and sigmas follow the same indexing convention.
monitor.get("sigma_x", turn=2, slot=0)
monitor.get("cov_x_px", turn=2, slot=0, slice_index=1)
