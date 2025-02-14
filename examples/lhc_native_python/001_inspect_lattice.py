import xtrack as xt

env = xt.Environment()
env.call('lhc_out.py')

env.vars.load_madx('../../test_data/lhc_2024/injection_optics.madx')

env.lhcb1.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7e12)