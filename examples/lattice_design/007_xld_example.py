import xtrack as xt

env = xt.Environment()
eset = env.set
builder = env.new_builder
new = env.new
new_line = env.new_line


eset('l.mq', 0.5)
eset('kqf', 0.027)
eset('kqd', -0.0271)
eset('l.mb', 10)
eset('l.ms', 0.3)
eset('k2sf', 0.001)
eset('k2sd', -0.001)
eset('angle.mb', '2 * np.pi / n_bends')
eset('k0.mb', 'angle.mb / l.mb')
eset('k0l.corrector', 0)
eset('k1sl.corrector', 0)
eset('l.halfcell', 38)

new('mb', 'Bend', length='l.mb', k0='k0.mb', h='k0.mb')
new('mq', 'Quadrupole', length='l.mq')
new('ms', 'Sextupole', length='l.ms')
new('corrector', xt.Multipole, knl=[0], ksl=[0])

new('mq.f', 'mq', k1='kqf')
new('mq.d', 'mq', k1='kqd')

halfcell = builder()
# End of the half cell (will be mid of the cell)
halfcell.new('mid', xt.Marker, at='l.halfcell'),

# Bends
halfcell.new('mb.2', 'mb', at='l.halfcell / 2'),
halfcell.new('mb.1', 'mb', at='-l.mb - 1', from_='mb.2'),
halfcell.new('mb.3', 'mb', at='l.mb + 1', from_='mb.2'),

# Quads
halfcell.place('mq.d', at = '0.5 + l.mq / 2'),
halfcell.place('mq.f', at = 'l.halfcell - l.mq / 2 - 0.5'),

# Sextupoles
halfcell.new('ms.d', 'ms', k2='k2sf', at=1.2, from_='mq.d'),
halfcell.new('ms.f', 'ms', k2='k2sd', at=-1.2, from_='mq.f'),

# Dipole correctors
halfcell.new('corrector.v', 'corrector', at=0.75, from_='mq.d'),
halfcell.new('corrector.h', 'corrector', at=-0.75, from_='mq.f')
