from cpymad.madx import Madx
import xtrack as xt

env = xt.load(['../../test_data/lhc_2024/lhc.seq',
               '../../test_data/lhc_2024/injection_optics.madx'],
               reverse_lines=['lhcb2'])

env.lhcb1.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=450e9)
env.lhcb2.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=450e9)

env.lhcb2.twiss_default['reverse'] = True

tt1 = env.lhcb1.get_table(attr=True)
tt2 = env.lhcb2.get_table(attr=True)

tw1 = env.lhcb1.twiss4d()
tw2 = env.lhcb2.twiss4d()
