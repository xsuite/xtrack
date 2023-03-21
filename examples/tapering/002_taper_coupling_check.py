import json
import numpy as np
import xtrack as xt


case_name = 'fcc-ee'
filename = 'line_no_radiation.json'

with open(filename, 'r') as f:
    line = xt.Line.from_dict(json.load(f))

#line['qc1l1.1..1'].knl[0] += 1e-5
#line['qc1l1.1..1'].ksl[0] += 1e-6/5
line['qc1l1.1..1'].ksl[1] = 1e-4

line.build_tracker()

# Initial twiss (no radiation)
line.configure_radiation(model=None)
tw_no_rad = line.twiss(method='4d', freeze_longitudinal=True)

# Enable radiation
line.configure_radiation(model='mean')
# - Set cavity lags to compensate energy loss
# - Taper magnet strengths
line.compensate_radiation_energy_loss(record_iterations=True)

import matplotlib.pyplot as plt
plt.close('all')

tw = line.twiss(eneloss_and_damping=True)

p0corr = 1 + line.delta_taper

plt.figure(1, figsize=(6.4*1.3, 4.8))

betx_beat = tw.betx*p0corr/tw_no_rad.betx-1
bety_beat = tw.bety*p0corr/tw_no_rad.bety-1
max_betx_beat = np.max(np.abs(betx_beat))
max_bety_beat = np.max(np.abs(bety_beat))
spx = plt.subplot(2,1,1)
sp0 = spx
plt.title(f'error on Qx: {abs(tw.qx - tw_no_rad.qx):.2e}     '
            r'$(\Delta \beta_x / \beta_x)_{max}$ = '
            f'{max_betx_beat:.2e}')
plt.plot(tw.s, betx_beat)
plt.ylabel(r'$\Delta \beta_x / \beta_x$')
plt.ylim(np.max([0.01, 1.1 * max_betx_beat])*np.array([-1, 1]))
plt.xlim([0, tw.s[-1]])

plt.subplot(2,1,2, sharex=sp0)
plt.title(f'error on Qy: {abs(tw.qy - tw_no_rad.qy):.2e}     '
            r'$(\Delta \beta_y / \beta_y)_{max}$ = '
            f'{max_bety_beat:.2e}')
plt.plot(tw.s, bety_beat)
plt.ylabel(r'$\Delta \beta_y / \beta_y$')
plt.ylim(np.max([0.01, 1.1 * max_bety_beat])*np.array([-1, 1]))
plt.xlabel('s [m]')

plt.subplots_adjust(hspace=0.35, top=.85)


plt.figure(11, figsize=(6.4*1.3, 4.8))

spx = plt.subplot(2,1,1, sharex=sp0)
plt.plot(tw.s, tw.betx, label='tapered')
plt.plot(tw.s, tw_no_rad.betx, '--', label='no radiation')
plt.ylabel(r'$\beta_{x}}$ [m]')
plt.xlim([0, tw.s[-1]])
plt.xlabel('s [m]')
plt.legend(loc='upper right')

plt.subplot(2,1,2, sharex=spx)
plt.plot(tw.s, tw.bety)
plt.plot(tw.s, tw_no_rad.bety, '--')
plt.ylabel(r'$\beta_{y}$ [m]')
plt.xlabel('s [m]')

plt.subplots_adjust(hspace=0.35, top=.85)


plt.figure(2, figsize=(6.4*1.3, 4.8))

bety1_beat = tw.bety1*p0corr/tw_no_rad.bety1-1
betx2_beat = tw.betx2*p0corr/tw_no_rad.betx2-1
max_bety1_beat = np.max(np.abs(bety1_beat))
max_betx2_beat = np.max(np.abs(betx2_beat))
spx = plt.subplot(2,1,1, sharex=sp0)
plt.title(f'error on Qx: {abs(tw.qx - tw_no_rad.qx):.2e}     '
            r'$(\Delta \beta_{y1} / \beta_{y1})_{max}$ = '
            f'{max_bety1_beat:.2e}')
plt.plot(tw.s, bety1_beat)
plt.ylabel(r'$\Delta \beta_{y1} / \beta_{y1}$')
plt.ylim(np.max([0.01, 1.1 * max_bety1_beat])*np.array([-1, 1]))
plt.xlim([0, tw.s[-1]])

plt.subplot(2,1,2, sharex=spx)
plt.title(f'error on Qy: {abs(tw.qy - tw_no_rad.qy):.2e}     '
            r'$(\Delta \beta_{x2} / \beta_{x2})_{max}$ = '
            f'{max_betx2_beat:.2e}')
plt.plot(tw.s, betx2_beat)
plt.ylabel(r'$\Delta \beta_{x2} / \beta_{x2}$')
plt.ylim(np.max([0.01, 1.1 * max_betx2_beat])*np.array([-1, 1]))
plt.xlabel('s [m]')

plt.subplots_adjust(hspace=0.35, top=.85)



plt.figure(22, figsize=(6.4*1.3, 4.8))

spx = plt.subplot(2,1,1, sharex=sp0)
plt.plot(tw.s, tw.bety1, label='tapered')
plt.plot(tw.s, tw_no_rad.bety1, '--', label='no radiation')
plt.ylabel(r'$\beta_{y1}}$ [m]')
plt.xlim([0, tw.s[-1]])
plt.xlabel('s [m]')
plt.legend(loc='upper right')

plt.subplot(2,1,2, sharex=spx)
plt.plot(tw.s, tw.betx2)
plt.plot(tw.s, tw_no_rad.betx2, '--')
plt.ylabel(r'$\beta_{x2}$ [m]')
plt.xlabel('s [m]')

plt.subplots_adjust(hspace=0.35, top=.85)


plt.figure(33, figsize=(6.4*1.3, 4.8))

spx = plt.subplot(2,1,1, sharex=sp0)
plt.plot(tw.s, tw.x, label='tapered')
plt.plot(tw.s, tw_no_rad.x, '--', label='no radiation')
plt.ylabel(r'x [m]')
plt.xlim([0, tw.s[-1]])
plt.xlabel('s [m]')
plt.legend(loc='upper right')

plt.subplot(2,1,2, sharex=spx)
plt.plot(tw.s, tw.y)
plt.plot(tw.s, tw_no_rad.y, '--')
plt.ylabel(r'y [m]')
plt.xlabel('s [m]')

plt.subplots_adjust(hspace=0.35, top=.85)




plt.figure(100)
for i_iter, mon in enumerate(line._tapering_iterations):
    plt.plot(mon.s.T, mon.delta.T,
                label=f'iter {i_iter} - Ene. loss: {-mon.delta[-1, -1]*100:.2f} %')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.ylim(top=0.01)
    plt.xlim(left=0, right=mon.s[-1, -1])
    plt.xlabel('s [m]')
    plt.ylabel(r'$\delta$')

plt.show()
