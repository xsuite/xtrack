import xtrack as xt
import xobjects as xo

xo.context_default.kernels.clear()

bend_only_e1 = xt.Bend(
    length=0, k0=0.1,
    edge_entry_angle=0.05,
    edge_exit_active=False)

edge_e1 = xt.DipoleEdge(
    k=0.1,
    model='linear', side='entry',
    e1=0.05)

edge_e1.model = 'full'
bend_only_e1.edge_entry_model = 'full'

p1 = xt.Particles(p0c=10e9, x=0.1, px=0.2, y=0.3, py=0.4, delta=0.03)
p2 = p1.copy()

bend_only_e1.track(p1)
edge_e1.track(p2)