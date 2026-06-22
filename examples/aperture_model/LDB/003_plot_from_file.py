import matplotlib.pyplot as plt
import numpy as np
import xtrack as xt
import xobjects as xo
from xtrack.aperture import Aperture

# Run 002_build_model.py first to generate the aperture model in JSON format.

PIPE_OVERLAP_TOL = 1e-3

context = xo.ContextCpu(omp_num_threads='auto')

# Load and straighten the LHC
lattice = xt.load('lhc.json')
lattice.vars.load('opt_6000.madx')

b1 = lattice.b1
b2 = lattice.b2

# Check that they can twiss
b1.twiss4d()
b2.twiss4d()

sv_b1 = b1.survey()
sv_b2 = b2.survey(theta0=np.pi)

# Build the curved model
aperture = Aperture.from_json('b1_aperture.json', line=b1, s_tol=PIPE_OVERLAP_TOL, context=context)
b_tab = aperture.get_bounds_table()
s_positions_at_lattice_changes = np.array(sorted(set(b_tab.s_start) | set(b_tab.s_end) | set(sv_b1.s)))
s_positions = np.array(sorted(set(s_positions_at_lattice_changes - PIPE_OVERLAP_TOL) | set(s_positions_at_lattice_changes + PIPE_OVERLAP_TOL)))
s_positions %= b1.get_length()
s_positions = sorted(s_positions)
aperture.plot_extents(s_positions)
plt.show()

plt.plot(sv_b1.Z, sv_b1.X, c='red', label='B1')
plt.plot(sv_b2.Z, sv_b2.X, c='blue', label='B2')
ax = plt.gca()
aperture.plot_floor_projection(ax=ax)
ax.set_aspect('auto')
plt.title('LHC LS3 Aperture and Survey Floor Plot (Pipe Model for B1)')
plt.xlabel('$Z$ [m]')
plt.ylabel('$X$ [m]')
plt.show()
