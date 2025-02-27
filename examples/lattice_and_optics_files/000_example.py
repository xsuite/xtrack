import xtrack as xt

env = xt.Environment()

env.call('pimm_lattice.py')
env.call('pimm_optics.py')

env['pimm'].particle_ref = xt.Particles(q0=1, mass0=xt.PROTON_MASS_EV,
                                        kinetic_energy0=200e6) # eV

env['pimm'].twiss4d().plot()