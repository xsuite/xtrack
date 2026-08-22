import xtrack as xt

env = xt.Environment()

env['p0c_optics_gev'] = 1 # in GeV

env.new_particle('particle/b1', mass0=xt.PROTON_MASS_EV,
                 q0=1, p0c=['p0c_optics_gev * 1e9'])

# Element controlled in field
env['spectrometer_b_tesla'] = 3.0
env['l.spectrometer'] = 2.0

env.new('spectrometer.b1', 'Bend', angle=0, length='l.spectrometer',
    k0=env.ref['spectrometer_b_tesla'] / env.ref['particle/b1'].rigidity0[0])

# Make a line
line = env.new_line(components=['spectrometer.b1'])
line.particle_ref='particle/b1'

env['spectrometer.b1'].k0 # is 0.899377374

# Load an optics file setting the energy
env.vars.load(format='madx', string='''
    !------ Set the energy -------
    p0c_optics_gev = 7000.;
''')

env['spectrometer.b1'].k0 # is 0.000128482482

line.particle_ref.p0c = 450e9

env['spectrometer.b1'].k0 # is 0.0019986163866


