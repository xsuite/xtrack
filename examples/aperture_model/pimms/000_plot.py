import xtrack as xt
import matplotlib.pyplot as plt
import numpy as np

from pimms import aperture, ring

ring_table = ring.get_table()
bounds_table = aperture.get_bounds_table()
s_edges = np.sort(np.concatenate([bounds_table.s_start, bounds_table.s_end]))
eps = 1e-4
s_positions = np.sort(np.concatenate([s_edges - eps, s_edges + eps]))
s_positions = np.clip(s_positions, 0.0, ring_table.s[-1])
s_positions = np.unique(s_positions)

aperture.plot_extents(s_positions=s_positions, sigmas=1)
plt.show()

sv = ring.survey()
sv.plot()
plt.show()
