import json

# TODO:
# - Put lag on the stable slope
# - Assert that calculated enekick is smaller than voltage at each cavity
# - Check 6d kick
# - Assert no collective
# - Assert no ions

# Some considerations:
# - One preserves angles not normalized momenta (in this sense the right beta to be
# preserved is in the plane x-x' and not x-px)
# - I observe the right tune on v on the real tracker and not with the one with the
# eneloss of the closed orbit.
# On the real tracker I see that the beta beating is introduced by the cavities
# not by the multipoles.

import numpy as np
from scipy.constants import c as clight
import xtrack as xt

with open('line_no_radiation.json', 'r') as f:
    line = xt.Line.from_dict(json.load(f))

line_df = line.to_pandas()
multipoles = line_df[line_df['element_type'] == 'Multipole']
cavities = line_df[line_df['element_type'] == 'Cavity'].copy()

# save voltages
cavities['voltage'] = [cc.voltage for cc in cavities.element.values]
cavities['frequency'] = [cc.frequency for cc in cavities.element.values]
cavities['eneloss_partitioning'] = cavities['voltage'] / cavities['voltage'].sum()


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
plt.close('all')

rtot_eneloss = 1e-10

# Put all cavities on crest and at zero frequency
for cc in cavities.element.values:
    cc.lag = 90
    cc.frequency = 0

while True:
    p_test = tw_no_rad.particle_on_co.copy()
    tracker_taper.configure_radiation(mode='mean')
    tracker_taper.track(p_test, turn_by_turn_monitor='ONE_TURN_EBE')
    mon = tracker_taper.record_last_track

    eloss = -(mon.ptau[0, -1] - mon.ptau[0, 0]) * p_test.p0c[0]
    print(f"Energy loss: {eloss:.3f} eV")

    if eloss < p_test.energy0[0]*rtot_eneloss:
        break

    for ii in cavities.index:
        cc = cavities.loc[ii, 'element']
        eneloss_partitioning = cavities.loc[ii, 'eneloss_partitioning']
        cc.voltage += eloss * eneloss_partitioning

    plt.plot(mon.s.T, mon.ptau.T)

i_multipoles = multipoles.index.values
delta_taper = ((mon.delta[0,:][i_multipoles+1] + mon.delta[0,:][i_multipoles]) / 2)
#delta_taper = mon.delta[0,:][i_multipoles]
for nn, dd in zip(multipoles['name'].values, delta_taper):
    line[nn].knl *= (1 + dd)
    line[nn].ksl *= (1 + dd)

# i_multipoles = multipoles.index.values
# delta_taper = ((mon.delta[0,:][i_multipoles+1] + mon.delta[0,:][i_multipoles]) / 2)
# for nn, dd in zip(multipoles['name'].values, delta_taper):
#     #line[nn].knl *= (1 + dd)
#     #line[nn].ksl *= (1 + dd)
#     line[nn].new_p0c = p_test.p0c[0] * (1 + dd)

# delta_taper_cavities = ((mon.delta[0,:][cavities.index.values+1] + mon.delta[0,:][cavities.index.values]) / 2)
# for nn, dd in zip(cavities['name'].values, delta_taper_cavities):
#     line[nn].new_p0c = p_test.p0c[0] * (1 + dd)

beta0 = p_test.beta0[0]
v_ratio = []
for icav in cavities.index:
    v_ratio.append(cavities.loc[icav, 'element'].voltage / cavities.loc[icav, 'voltage'])
    inst_phase = np.arcsin(cavities.loc[icav, 'element'].voltage / cavities.loc[icav, 'voltage'])
    freq = cavities.loc[icav, 'frequency']

    zeta = mon.zeta[0, icav]
    lag = 360.*(inst_phase / (2*np.pi) - freq*zeta/beta0/clight)
    lag = 180. - lag # we are above transition

    cavities.loc[icav, 'element'].lag = lag
    cavities.loc[icav, 'element'].frequency = freq
    cavities.loc[icav, 'element'].voltage = cavities.loc[icav, 'voltage']


tw_damp = tracker.twiss(method='6d', matrix_stability_tol=0.5)
tracker_twiss = xt.Tracker(line = line, extra_headers=["#define XSUITE_SYNRAD_TWISS_MODE"])
tw_nodamp = tracker_twiss.twiss(method='6d')

tracker_twiss_cav = xt.Tracker(line = line, extra_headers=["#define XTRACK_CAVITY_TWISS_MODE"])
tw_nodamp = tracker_twiss_cav.twiss(method='6d', matrix_stability_tol=0.5)

print(f'{tw_no_rad.qx=}\n{tw_damp.qx=}\n{tw_nodamp.qx=}')
print(f'{tw_no_rad.qy=}\n{tw_damp.qy=}\n{tw_nodamp.qy=}')

plt.figure(2)

plt.subplot(2,1,1)
plt.plot(tw_no_rad.s, tw_damp.betx/tw_no_rad.betx - 1, 'b')
plt.plot(tw_no_rad.s, tw_nodamp.betx/tw_no_rad.betx - 1, 'r')

plt.subplot(2,1,2)
plt.plot(tw_no_rad.s, tw_damp.bety/tw_no_rad.bety - 1, 'b')
plt.plot(tw_no_rad.s, tw_nodamp.bety/tw_no_rad.bety - 1, 'r')

plt.figure(10)
plt.subplot(2,1,1)
plt.plot(tw_no_rad.s, tw_no_rad.x, 'k')
plt.plot(tw_no_rad.s, tw_damp.x, 'b')
plt.plot(tw_no_rad.s, tw_nodamp.x, 'r')

plt.subplot(2,1,2)
plt.plot(tw_no_rad.s, tw_no_rad.y, 'k')
plt.plot(tw_no_rad.s, tw_damp.y, 'b')
plt.plot(tw_no_rad.s, tw_nodamp.y, 'r')


plt.show()