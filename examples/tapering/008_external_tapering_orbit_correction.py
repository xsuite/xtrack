import numpy as np
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

# Add the six distributed cavities, keeping the original cavity as in 009b.
locations = ['bpv.11706', 'bpv.21508', 'bpv.31508',
             'bpv.41508', 'bpv.51508', 'bpv.61508']
line.insert([
    line.env.new(f'cav{i}', 'actcse.31632', length=0, at=location)
    for i, location in enumerate(locations, start=1)])
line.build_tracker()
line.configure_radiation(model='mean')

# Keep the magnets thick; taper quadrupoles and sextupoles, leaving bends untapered.
tt = line.get_table()
tt_magnets = tt.rows.match(element_type='Quadrupole|Sextupole')
magnets = [(i, line.element_dict[tt.name[i]])
           for i in tt.rows.indices[tt_magnets.name]]
# Use knl[0] at the quadrupoles to correct the orbit from the untapered bends.
quads = tt.rows.match(element_type='Quadrupole').name

line.compensate_radiation_energy_loss(taper_magnets=False, verbose=False)
tw0 = line.twiss5d()
tw_before = tw0
correction = line.correct_trajectory(
    method='5d', twiss_table=tw_before, run=False,
    monitor_names_x=quads, corrector_names_x=quads, rcond=1e-6)

for iteration in range(20):
    line.compensate_radiation_energy_loss(
        taper_magnets=False, delta0='zero_mean', verbose=False)
    correction.correct(planes='x', n_iter=1, verbose=False)
    tw = line.twiss5d(only_orbit=True)
    delta_taper = 0.5 * (tw.delta[:-1] + tw.delta[1:])
    taper_error = max(abs(el.delta_taper - delta_taper[i]) for i, el in magnets)
    orbit_error = np.max(np.abs(tw.rows[quads].x))
    delta_mean = np.trapezoid(tw.delta, tw.s) / tw.s[-1]
    print(f'{iteration=}: {taper_error=:.3g}, {orbit_error=:.3g}, {delta_mean=:.3g}')
    if taper_error < 1e-9 and orbit_error < 1e-8 and abs(delta_mean) < 1e-9:
        break
    for i, el in magnets:
        el.delta_taper = delta_taper[i]
else:
    raise RuntimeError('Tapering and orbit correction did not converge')

# Correct the transverse orbit; the energy sawtooth remains.
import matplotlib.pyplot as plt
plt.close('all')
tw = line.twiss5d()
tw['x_before'] = tw_before.x
tw.plot()
tw.plot('x_before x')
tw.plot('delta')
