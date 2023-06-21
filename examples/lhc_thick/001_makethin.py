import xtrack as xt

from xtrack.slicing import Teapot, Strategy

line = xt.Line.from_json('lhc_thick_with_knobs.json')
line.build_tracker()

line_thick = line.copy()
line_thick.build_tracker()

slicing_strategies = [
    Strategy(slicing=Teapot(1)),  # Default catch-all as in MAD-X
    Strategy(slicing=Teapot(4), element_type=xt.Bend),
    Strategy(slicing=Teapot(20), element_type=xt.Quadrupole),
    Strategy(slicing=Teapot(2), name=r'^mb\..*'),
    Strategy(slicing=Teapot(5), name=r'^mq\..*'),
    Strategy(slicing=Teapot(3), name=r'^mqt.*'),
    Strategy(slicing=Teapot(60), name=r'^mqx.*'),
]

line.discard_tracker()
line.slice_thick_elements(slicing_strategies=slicing_strategies)
line.build_tracker()

tw = line.twiss()
tw_thick = line_thick.twiss()

beta_beat_x_at_ips = [tw['betx', f'ip{nn}'] / tw_thick['betx', f'ip{nn}'] - 1
                        for nn in range(1, 9)]
beta_beat_y_at_ips = [tw['bety', f'ip{nn}'] / tw_thick['bety', f'ip{nn}'] - 1
                        for nn in range(1, 9)]

assert np.allclose(beta_beat_x_at_ips, 0, atol=3e-3)
assert np.allclose(beta_beat_y_at_ips, 0, atol=3e-3)