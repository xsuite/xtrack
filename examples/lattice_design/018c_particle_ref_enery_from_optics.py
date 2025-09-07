import xtrack as xt

env = xt.Environment()

env['nrg'] = 7000. # in GeV

env.new_particle('particle/b1', mass0=xt.PROTON_MASS_EV,
                 q0=1, p0c='nrg * 1e9')

# Line with a spectrometer controlled in field
env['spectrometer_b_tesla'] = 3.0
env['l.spectrometer'] = 2.0

env.new_line('b1', components=[
    env.new('spectrometer.b1', 'Bend', h=0, length='l.spectrometer',
            angle='l.spectrometer * spectrometer_b_tesla * 0.299792458 / particle/b1.p0c'),