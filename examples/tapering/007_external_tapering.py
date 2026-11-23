from cpymad.madx import Madx
import xtrack as xt

# SPS optics and RF configuration from radiation/009b.
mad = Madx()
mad.call('../../test_data/sps_thick/sps.seq')
mad.input('beam, particle=electron, pc=45;')
mad.call('../../test_data/sps_thick/lhc_q20.str')
mad.use(sequence='sps')
mad.sequence.sps.elements['actcse.31632'].volt = 250 * 10 / 6
mad.sequence.sps.elements['actcse.31632'].freq = 10
mad.sequence.sps.elements['actcse.31632'].lag = 0.5
line = xt.Line.from_madx_sequence(mad.sequence.sps, allow_thick=True,
                                  deferred_expressions=True)
line.particle_ref = xt.Particles(mass0=xt.ELECTRON_MASS_EV, q0=-1, p0c=45e9)
line.cycle('bpv.11706', inplace=True)

# Add the six distributed cavities
locations = ['bpv.11706', 'bpv.21508', 'bpv.31508',
             'bpv.41508', 'bpv.51508', 'bpv.61508']
line.insert([
    line.env.new(f'cav{i}', 'actcse.31632', length=0, at=location)
    for i, location in enumerate(locations, start=1)])
line.build_tracker()
line.configure_radiation(model='mean')

# Taper all magnets using the local momentum.
magnets = [(i, line.element_dict[name]) for i, name in enumerate(line.element_names)
           if hasattr(line.element_dict[name], 'delta_taper')]

for iteration in range(10):
    line.compensate_radiation_energy_loss(
        taper_magnets=False, delta0='zero_mean', verbose=False)
    tw = line.twiss5d(only_orbit=True)
    if iteration == 0:
        tw_before = tw
    delta_taper = 0.5 * (tw.delta[:-1] + tw.delta[1:])
    error = max(abs(el.delta_taper - delta_taper[i]) for i, el in magnets)
    print(f'{iteration=}: {error=:.3g}')
    if error < 1e-9:
        break
    for i, el in magnets:
        el.delta_taper = delta_taper[i]
else:
    raise RuntimeError('Tapering did not converge')

tw = line.twiss5d()
tw['x_before'] = tw_before.x
tw.plot()
tw.plot('x_before x')
tw.plot('delta')
