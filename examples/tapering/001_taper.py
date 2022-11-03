import json
import xtrack as xt

with open('line_no_radiation.json', 'r') as f:
    line = xt.Line.from_dict(json.load(f))

line_df = line.to_pandas()
multipoles = line_df[line_df['element_type'] == 'Multipole']
cavities = line_df[line_df['element_type'] == 'Cavity'].copy()

# save voltages
cavities['voltage'] = [cc.voltage for cc in cavities.element.values]
cavities['frequency'] = [cc.frequency for cc in cavities.element.values]

# set voltages to zero
for cc in cavities.element.values:
    cc.voltage = 0

tracker = xt.Tracker(line = line)
tw_no_rad = tracker.twiss(method='4d')

p_test = tw_no_rad.particle_on_co.copy()
tracker.configure_radiation(mode='mean')

tracker.track(p_test, turn_by_turn_monitor='ONE_TURN_EBE')
mon = tracker.record_last_track

n_cavities = len(cavities)

tracker_taper = xt.Tracker(line = line, extra_headers=["#define XTRACK_MULTIPOLE_TAPER"])

import matplotlib.pyplot as plt

while True:
    p_test = tw_no_rad.particle_on_co.copy()
    tracker_taper.configure_radiation(mode='mean')
    tracker_taper.track(p_test, turn_by_turn_monitor='ONE_TURN_EBE')
    mon = tracker_taper.record_last_track

    eloss = -(mon.ptau[0, -1] - mon.ptau[0, 0]) * p_test.p0c[0]
    print(f"Energy loss: {eloss:.3f} eV")

    if eloss < p_test.energy0[0]*1e-6:
        break

    for cc in cavities.element:
        cc.lag = 90.0
        cc.frequency = 0.0000001
        cc.voltage += eloss/n_cavities

    plt.plot(mon.s.T, mon.ptau.T)

delta_taper = ((mon.delta[0,:][i_multipoles+1] + mon.delta[0,:][i_multipoles]) / 2)
for nn, dd in zip(multipoles['name'].values, delta_taper):
    line[nn].knl *= (1 + dd)
    line[nn].ksl *= (1 + dd)

tw_taper = tracker.twiss(method='4d', matrix_stability_tol=0.5)

for icav in cavities.index:
    inst_phase = 2*np.pi*freq*zeta

tracker_twiss = xt.Tracker(line = line, extra_headers=["#define XSUITE_SYNRAD_TWISS_MODE"])