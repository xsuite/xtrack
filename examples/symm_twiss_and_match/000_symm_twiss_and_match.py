import xtrack as xt
import xobjects as xo
import numpy as np

# Build line with half a cell
half_cell = xt.Line(
    elements={
        'start_cell': xt.Marker(),
        'drift0': xt.Drift(length=1.),
        'qf1': xt.Quadrupole(k1=0.027/2, length=1.),
        'drift1_1': xt.Drift(length=1),
        'bend1': xt.Bend(k0=3e-4, h=3e-4, length=45.),
        'drift1_2': xt.Drift(length=1.),
        'qd1': xt.Quadrupole(k1=-0.0271/2, length=1.),
        'drift2': xt.Drift(length=1),
        'mid_cell': xt.Marker(),
    }
)
half_cell.particle_ref = xt.Particles(p0c=2e9)

# Add observation points every 1 m (to see betas inside bends)
half_cell.discard_tracker()
s_cut = np.arange(0, half_cell.get_length(), 1.)
half_cell.cut_at_s(s_cut)

# Attach knobs to quadrupoles
half_cell.vars['kqf'] = 0.027/2
half_cell.vars['kqd'] = -0.0271/2
half_cell.element_refs['qf1'].k1 = half_cell.vars['kqf']
half_cell.element_refs['qd1'].k1 = half_cell.vars['kqd']

# Match with periodic symmetric boundary
opt_halfcell = half_cell.match(
    method='4d',
    start='start_cell', end='mid_cell',
    init='periodic_symmetric',
    targets=xt.TargetSet(mux=0.2501/2, muy=0.2502/2, at='mid_cell'),
    vary=xt.VaryList(['kqf', 'kqd'], step=1e-5),
)

# Twiss with periodic symmetric boundary
tw_half_cell = half_cell.twiss4d(init='periodic_symmetric')

# Plot
import matplotlib.pyplot as plt
plt.close('all')
tw_half_cell.plot()
plt.show()