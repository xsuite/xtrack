import xtrack as xt

env = xt.load_madx_lattice('../../test_data/pimms/PIMMS.seq')

line = env.pimms
line.particle_ref = xt.Particles(kinetic_energy0=100e6)

line['kqfa']=   0.30247
line['kqfb']=  0.523281
line['kqd']= -0.518932

line.twiss4d().plot()