import numpy as np

import xtrack as xt


bunch_spacing_zeta = 25.0

particles = xt.Particles(
    p0c=7e12,
    x=[1.0e-3, 2.0e-3, 10.0e-3, 14.0e-3, 20.0e-3, 22.0e-3],
    px=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    y=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    py=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    zeta=[
        -0.05, 0.05,
        -2 * bunch_spacing_zeta - 0.05,
        -2 * bunch_spacing_zeta + 0.05,
        -5 * bunch_spacing_zeta - 0.05,
        -5 * bunch_spacing_zeta + 0.05,
    ],
    delta=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    weight=[1.0, 1.0, 1.0, 3.0, 2.0, 2.0],
)

monitor = xt.BeamStatsMonitor(
    start_at_turn=0,
    stop_at_turn=1,
    zeta_range=(-0.2, 0.2),
    num_slices=1,
    filled_slots=[0, 2, 5],
    selected_slots=[2, 5],
    bunch_spacing_zeta=bunch_spacing_zeta,
    stats=["num_particles", "mean_x"],
)

monitor.track(particles)

assert np.all(monitor.selected_slots == [2, 5])
assert monitor.mean_x.shape == (1, 2, 1)

# The selected-slot axis follows selected_slots: slot 2 first, slot 5 second.
print("selected slots:", monitor.selected_slots)
print("zeta centers per selected slot:")
print(monitor.zeta_centers)

for i_slot, slot in enumerate(monitor.selected_slots):
    print(
        "slot", slot,
        "num_particles", monitor.num_particles[0, i_slot, 0],
        "mean_x", monitor.mean_x[0, i_slot, 0],
    )
