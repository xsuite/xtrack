import xtrack as xt

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)

class MyKicker:
    def __init__(self, kickx=0, kicky=0):
        self.kickx = kickx
        self.kicky = kicky

    def track(self, p):
        p.px += self.kickx
        p.py += self.kicky

kick1 = MyKicker(kickx=1e-4)
kick2 = MyKicker(kickx=-2e-4)
kick3 = MyKicker(kickx=1e-4)

line = env.new_line(components=[
    env.place('k1', kick1, at=0),
    env.place('k2', kick2, at=2),
    env.place('k3', kick3, at=4),
    ])

tw = line.twiss(betx=1, bety=1, include_collective=True)


class MyQuad:
    def __init__(self, k1l=0):
        self.k1l = k1l

    def track(self, p):
        p.px += -self.k1l * p.x
        p.py += self.k1l * p.y

kq = 0.1
qf = MyQuad(k1l=0.1)
qd = MyQuad(k1l=-0.1)

lfodo = env.new_line(components=[
    env.place('qf', qf, at=0),
    env.place('qd', qd, at=10),
    env.new('end', 'Marker', at=20),
])

twfodo = lfodo.twiss4d(include_collective=True)