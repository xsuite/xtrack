import xtrack as xt
import xobjects as xo

env = xt.Environment()
env.particle_ref = xt.Particles(p0c=7e12)

class MyKicker:
    def __init__(self, kickx=0, kicky=0):
        self.kickx = kickx
        self.kicky = kicky

    def track(self, p):
        p.px += self.kickx
        p.py += self.kicky

kick1 = MyKicker(kickx=1e-3)
kick2 = MyKicker(kickx=-2e-3)
kick3 = MyKicker(kickx=1e-3)

line = env.new_line(components=[
    env.place('k1', kick1, at=0),
    env.place('k2', kick2, at=2),
    env.place('k3', kick3, at=4),
    ])

tw = line.twiss(betx=1, bety=1, include_collective=True)
xo.assert_allclose(tw['x', 'k2'], 2e-3, atol=1e-8, rtol=0)

tw_nocoll = line.twiss(betx=1, bety=1)
xo.assert_allclose(tw_nocoll['x', 'k2'], 0, atol=1e-8, rtol=0)


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

lfodo_ref = env.new_line(components=[
    env.new('mqf', 'Multipole', knl=[0, kq], at=0),
    env.new('mqd', 'Multipole', knl=[0, -kq], at=10),
    env.new('mend', 'Marker', at=20),
])

twfodo_ref = lfodo_ref.twiss4d(include_collective=True)

xo.assert_allclose(twfodo.qx, twfodo_ref.qx, atol=1e-8, rtol=0)
xo.assert_allclose(twfodo.qy, twfodo_ref.qy, atol=1e-8, rtol=0)
xo.assert_allclose(twfodo.dqx, twfodo_ref.dqx, atol=1e-8, rtol=0)
xo.assert_allclose(twfodo.dqy, twfodo_ref.dqy, atol=1e-8, rtol=0)

twfodo_open = lfodo.twiss4d(include_collective=True, betx=1, bety=1,
                            compute_chromatic_properties=True)
twfodo_open_ref = lfodo_ref.twiss4d(include_collective=True, betx=1, bety=1,
                                    compute_chromatic_properties=True)

xo.assert_allclose(twfodo_open.betx, twfodo_open_ref.betx, atol=0, rtol=1e-8)
xo.assert_allclose(twfodo_open.bety, twfodo_open_ref.bety, atol=0, rtol=1e-8)
xo.assert_allclose(twfodo_open.ax_chrom, twfodo_open_ref.ax_chrom, atol=0, rtol=1e-8)
xo.assert_allclose(twfodo_open.ay_chrom, twfodo_open_ref.ay_chrom, atol=0, rtol=1e-8)