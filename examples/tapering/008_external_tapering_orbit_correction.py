from pathlib import Path
import numpy as np
import xtrack as xt

line = xt.load(Path(__file__).parent / '../../test_data/clic_dr/line_for_taper.json')
line.particle_ref.p0c = 4e9
line.build_tracker()
line.configure_radiation(model='mean')

# Taper the quadrupoles and sextupoles, leaving dipoles and their edges fixed.
magnets = [(i, line.element_dict[name]) for i, name in enumerate(line.element_names)
           if isinstance(line.element_dict[name], xt.Multipole)
           and line.element_dict[name].order >= 1
           and line.element_dict[name].hxl == 0]
# Use knl[0] and measure the orbit at one slice of each four-slice quadrupole.
quads = [line.element_names[i] for i, el in magnets if el.knl[1] != 0][::4]

line.compensate_radiation_energy_loss(taper_magnets=False, verbose=False)
tw_before = line.twiss5d(chrom=False)
correction = line.correct_trajectory(
    method='5d', twiss_table=tw_before, run=False,
    monitor_names_x=quads, corrector_names_x=quads, rcond=1e-6)

for iteration in range(10):
    line.compensate_radiation_energy_loss(
        taper_magnets=False, delta0='zero_mean', verbose=False)
    correction.correct(planes='x', n_iter=1, verbose=False)
    tw = line.twiss5d(only_orbit=True)
    delta_taper = 0.5 * (tw.delta[:-1] + tw.delta[1:])
    taper_error = max(abs(el.delta_taper - delta_taper[i]) for i, el in magnets)
    orbit_error = np.max(np.abs(tw['x', quads]))
    delta_mean = np.trapezoid(tw.delta, tw.s) / tw.s[-1]
    print(f'{iteration=}: {taper_error=:.3g}, {orbit_error=:.3g}, {delta_mean=:.3g}')
    if taper_error < 1e-9 and orbit_error < 1e-8 and abs(delta_mean) < 1e-9:
        break
    for i, el in magnets:
        el.delta_taper = delta_taper[i]
else:
    raise RuntimeError('Tapering and orbit correction did not converge')

# Correct the transverse orbit; the energy sawtooth remains.
tw = line.twiss5d()
tw['x_before'] = tw_before.x
tw.plot()
tw.plot('x_before x')
tw.plot('delta')
