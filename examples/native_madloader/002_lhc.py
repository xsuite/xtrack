import xtrack as xt

env = xt.load_madx_lattice('../../test_data/lhc_2024/lhc.seq')
env.vars.load_madx('../../test_data/lhc_2024/injection_optics.madx')

line = env.lhcb1
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=7e12)
