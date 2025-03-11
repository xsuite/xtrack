import xtrack as xt
import xobjects as xo

from xtrack.beam_elements.magnets import Magnet

m = Magnet(length=1.0, k0=0.0, k1=0.0, h=0.0)

p0 = xt.Particles(kinetic_energy0=50e6,
                  x=1e-3, y=2e-3, zeta=1e-2, px=10e-3, py=20e-3, delta=1e-2)

m.compile_kernels()
