import xtrack as xt
from xtrack.mad_parser.loader import MadxLoader
from xtrack.mad_parser.env_writer import EnvWriterProxy

env_proxy = EnvWriterProxy()
loader = MadxLoader(reverse_lines=['lhcb2'], env=env_proxy)
loader.load_file("lhc.seq")
# loader.load_file("optics.madx")
env = loader.env.env
line = env.lines['lhcb1']
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)
tw = line.twiss4d()

print(f'qx = {tw.qx}, qy = {tw.qy}')

env_proxy.to_file('lhc_out.py')
