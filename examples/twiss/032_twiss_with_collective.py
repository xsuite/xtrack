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

line.twiss_default['include_collective'] = True

tw = line.twiss(betx=1, bety=1)

