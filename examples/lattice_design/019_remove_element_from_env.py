import xtrack as xt

env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')

nn = 'mcdxf.3l1/lhcb1'

env.ref[nn].knl[4].xdeps.info()


env.ref['kcdx3.l1'].xdeps.info()


env.elements.remove(nn)