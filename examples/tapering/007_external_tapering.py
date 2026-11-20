from pathlib import Path
import xtrack as xt

line = xt.load(Path(__file__).parent / '../../test_data/clic_dr/line_for_taper.json')
line.particle_ref.p0c = 4e9
line.build_tracker()
line.configure_radiation(model='mean')
magnets = [(i, line.element_dict[name]) for i, name in enumerate(line.element_names)
           if hasattr(line.element_dict[name], 'delta_taper')]

for iteration in range(10):
    line.compensate_radiation_energy_loss(
        taper_magnets=False, delta0='zero_mean', verbose=False)
    tw = line.twiss5d(chrom=False)
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
tw.plot()
tw.plot('delta')
