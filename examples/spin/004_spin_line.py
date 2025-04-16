import xtrack as xt

env = xt.load_madx_lattice('../../test_data/sps_thick/sps.seq')
env.vars.load_madx('../../test_data/sps_thick/lhc_q20.str')

line = env.sps
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=1, p0c=20e9,
                                 anomalous_magnetic_moment=0.00115965218128)
p = line.build_particles(x=[0,0,0],
                         spin_x=[1,0,0],
                         spin_y=[0,1,0],
                         spin_z=[0,0,1])

line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False # To enable spin tracking

line.track(p)
