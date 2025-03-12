import xtrack as xt
import xobjects as xo
from xtrack.beam_elements.magnets import Magnet

env = xt.Environment()
env.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, energy0=100e9)

line = env.new_line(components=[
    env.new('mb', 'Bend', angle=0.01, length=10., k0_from_h=True),
])

line.slice_thick_elements(
    slicing_strategies=[
        xt.Strategy(slicing=xt.Teapot(5))
    ]
)

line.build_tracker()
line.configure_radiation(model='mean')

p_ref = env.particle_ref.copy()
line.track(p_ref)

mm = Magnet(angle=0.01, length=10., k0_from_h=True)
mm.radiation_flag = 1 # 1: mean
p_test = p_ref.copy()

mm.track(p_test)

xo.assert_allclose(p_ref.delta, p_test.delta, atol=0, rtol=1e-6)