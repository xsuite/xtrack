import xtrack as xt

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider['lhcb1'].twiss_default['method'] = '4d'
collider['lhcb2'].twiss_default['method'] = '4d'

line = collider.lhcb1

tw = line.twiss()

init = xt.TwissInit(betx=0.15, bety=0.15,
                           ax_chrom=42.7928, bx_chrom=-18.4181,
                           ay_chrom=-18.0191, by_chrom= 11.54862)
tw_open = line.twiss(start='ip5', end='ip7', init=init,
                        compute_chromatic_properties=True)

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1, figsize=(6.4, 4.8*1.5))
ax1 = plt.subplot(4, 1, 1)
plt.plot(tw.s, tw.ax_chrom, label='closed')
plt.plot(tw_open.s, tw_open.ax_chrom, label='open')
plt.ylabel(r'$A_x$')
plt.legend()

plt.subplot(4, 1, 2, sharex=ax1)
plt.plot(tw.s, tw.bx_chrom, label='closed')
plt.plot(tw_open.s, tw_open.bx_chrom, label='open')
plt.ylabel(r'$B_x$')

plt.subplot(4, 1, 3, sharex=ax1)
plt.plot(tw.s, tw.ay_chrom, label='closed')
plt.plot(tw_open.s, tw_open.ay_chrom, label='open')
plt.ylabel(r'$A_y$')

plt.subplot(4, 1, 4, sharex=ax1)
plt.plot(tw.s, tw.by_chrom, label='closed')
plt.plot(tw_open.s, tw_open.by_chrom, label='open')
plt.ylabel(r'$B_y$')

plt.xlabel('s [m]')

plt.figure(2, figsize=(6.4, 4.8))
plt.subplot(2, 1, 1, sharex=ax1)
plt.plot(tw.s, tw.wx_chrom, label='closed')
plt.plot(tw_open.s, tw_open.wx_chrom, label='open')
plt.ylabel(r'$W_x$')

plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(tw.s, tw.wy_chrom, label='closed')
plt.plot(tw_open.s, tw_open.wy_chrom, label='open')
plt.ylabel(r'$W_y$')

plt.legend


plt.show()
