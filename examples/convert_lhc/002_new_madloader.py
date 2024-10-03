import xtrack as xt
from xtrack.mad_parser.loader import MadxLoader
from cpymad.madx import Madx

particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7000e9)

mad = Madx(stdout=False)
mad.call('lhc.seq')
mad.call('optics.madx')
mad.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
mad.use('lhcb1')
lhcb1_ref = xt.Line.from_madx_sequence(mad.sequence.lhcb1, deferred_expressions=True)
lhcb1_ref.particle_ref = particle_ref
lhcb1_ref.twiss()

loader = MadxLoader()
loader.load_file("lhc.seq")
loader.load_file("optics.madx")
lhcb1 = loader.env.lines['lhcb1']
lhcb1.particle_ref = particle_ref
lhcb1.twiss()
