import xtrack as xt

env = xt.load_madx_lattice('../../test_data/sps_thick/sps.seq')
env.vars.load_madx('../../test_data/sps_thick/lhc_q20.str')

# RF set tp stay in the linear region
env['actcse.31632'].voltage = 2500e6
env['actcse.31632'].frequency = 3e6
env['actcse.31632'].lag = 180.

line = env.sps
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, energy0=26e9)

tw4d = line.twiss4d()
tw6d = line.twiss()

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True)