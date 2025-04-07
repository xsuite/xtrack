import xtrack as xt

line = xt.Line.from_json('psb_investigate.json')

bb = line['bi1.bsw1l1.1']

rb = xt.RBend(length_straight=0.313, k0=0.211, k1=0, h=0*1e-6,
              k0_from_h=False, model='rot-kick-rot',
              num_multipole_kicks=1,
              knl=[ 0.      ,  0.      , -0.097429],
              edge_entry_active=0, edge_exit_active=0)

sb = xt.Bend(length=0.313, k0=0.211, k1=0, h=0*1e-6,
              k0_from_h=False,
              model='rot-kick-rot',
              num_multipole_kicks=1,
              knl=[ 0.      ,  0.      , -0.097429],
              edge_entry_active=0, edge_exit_active=0)

p0 = line.particle_ref.copy()

lrb = xt.Line(elements=[rb])
lrb.build_tracker()

prb = p0.copy()
psb = p0.copy()

lrb.track(prb)
sb.track(psb)

print(prb.x - psb.x)