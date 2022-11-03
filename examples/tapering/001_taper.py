import json
import xtrack as xt

with open('line_no_radiation.json', 'r') as f:
    line = xt.Line.from_dict(json.load(f))

tracker = xt.Tracker(line = line)
tw_no_rad = tracker.twiss(method='4d')

p_test = tw_no_rad.particle_on_co.copy()
tracker.configure_radiation(mode='mean')

tracker.track(p_test, turn_by_turn_monitor='ONE_TURN_EBE')
mon = tracker.record_last_track

cavities = [el for el in line.elements if isinstance(el, xt.Cavity)]
n_cavities = len(cavities)

tracker_taper = xt.Tracker(line = line, extra_headers=["#define XTRACK_MULTIPOLE_TAPER"])

import matplotlib.pyplot as plt

for _ in range(5):
    p_test = tw_no_rad.particle_on_co.copy()
    tracker_taper.configure_radiation(mode='mean')
    tracker_taper.track(p_test, turn_by_turn_monitor='ONE_TURN_EBE')
    mon = tracker_taper.record_last_track

    eloss = (mon.ptau[0, -1] - mon.ptau[0, 0]) * p_test.p0c[0]

    for cc in cavities:
        cc.lag = -90.0
        cc.frequency = 0.0000001
        cc.voltage += eloss/n_cavities

    plt.plot(mon.s.T, mon.ptau.T)

line_df = line.to_pandas()
multipoles = line_df[line_df['element_type'] == 'Multipole']
i_multipoles = multipoles.index.values

delta_taper = ((mon.delta[0,:][i_multipoles+1] + mon.delta[0,:][i_multipoles]) / 2)
for nn, dd in zip(multipoles['name'].values, delta_taper):
    line[nn].knl *= (1 + dd)
    line[nn].ksl *= (1 + dd)

tw_taper = tracker.twiss(method='4d', matrix_stability_tol=0.5)

tracker_twiss = xt.Tracker(line = line, extra_headers=["#define XSUITE_SYNRAD_TWISS_MODE"])