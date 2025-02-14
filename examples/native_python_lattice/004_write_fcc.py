import xtrack as xt
from xtrack.mad_parser.loader import MadxLoader
from xtrack.mad_parser.env_writer import EnvWriterProxy

env_proxy = EnvWriterProxy()
loader = MadxLoader(env=env_proxy)
loader.load_file("fccee_z.seq")
env = loader.env.env
line = env.lines['ring_full']
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=50_000e9)
tw = line.twiss4d()

print(f'qx = {tw.qx}, qy = {tw.qy}')

env_proxy.to_file('fcc_out.py')
