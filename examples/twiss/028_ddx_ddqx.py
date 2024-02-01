import numpy as np
from cpymad.madx import Madx
import xtrack as xt

mad = Madx()
mad.call('../../test_data/hllhc15_thick/lhc.seq')
mad.call('../../test_data/hllhc15_thick/hllhc_sequence.madx')
mad.call('../../test_data/hllhc15_thick/opt_round_150_1500.madx')
mad.input('beam, sequence=lhcb1, particle=proton, pc=7000;')
mad.use(sequence='lhcb1')
mad.globals['on_x5'] = 300
tw_chr = mad.twiss(chrom=True)

twm = xt.Table(tw_chr)

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1, deferred_expressions=True)
line.config.XTRACK_USE_EXACT_DRIFTS = True
line.configure_bend_model(core='expanded', edge='linear')

# tt = line.get_table()
# tt_sol = tt.rows[tt.element_type == 'Solenoid']
# for nn in tt_sol.name:
#     line.element_dict[nn] = xt.Drift(length=line[nn].length)

line.particle_ref = xt.Particles(mass0=mad.sequence.lhcb1.beam.mass*1e9,
                                    q0=mad.sequence.lhcb1.beam.charge,
                                    gamma0=mad.sequence.lhcb1.beam.gamma)

tw = line.twiss(method='4d')
nlchr = line.get_non_linear_chromaticity(delta0_range=(-1e-4, 1e-4),
                                        num_delta=21, fit_order=1, method='4d')
tw_fw = line.twiss(start='ip4', end='ip6', init_at='ip4',
              x=tw['x', 'ip4'], px=tw['px', 'ip4'],
              y=tw['y', 'ip4'], py=tw['py', 'ip4'],
              betx=tw['betx', 'ip4'], bety=tw['bety', 'ip4'],
              alfx=tw['alfx', 'ip4'], alfy=tw['alfy', 'ip4'],
              dx=tw['dx', 'ip4'], dpx=tw['dpx', 'ip4'],
              dy=tw['dy', 'ip4'], dpy=tw['dpy', 'ip4'],
              ddx=tw['ddx', 'ip4'], ddy=tw['ddy', 'ip4'],
              ddpx=tw['ddpx', 'ip4'], ddpy=tw['ddpy', 'ip4'],
              compute_chromatic_properties=True)

tw_bw = line.twiss(start='ip4', end='ip6', init_at='ip6',
              x=tw['x', 'ip6'], px=tw['px', 'ip6'],
              y=tw['y', 'ip6'], py=tw['py', 'ip6'],
              betx=tw['betx', 'ip6'], bety=tw['bety', 'ip6'],
              alfx=tw['alfx', 'ip6'], alfy=tw['alfy', 'ip6'],
              dx=tw['dx', 'ip6'], dpx=tw['dpx', 'ip6'],
              dy=tw['dy', 'ip6'], dpy=tw['dpy', 'ip6'],
              ddx=tw['ddx', 'ip6'], ddy=tw['ddy', 'ip6'],
              ddpx=tw['ddpx', 'ip6'], ddpy=tw['ddpy', 'ip6'],
              compute_chromatic_properties=True)


tw_fw_mad = line.twiss(start='ip4', end='ip6', init_at='ip4',
              x=twm['x', 'ip4:1'], px=twm['px', 'ip4:1'],
              y=twm['y', 'ip4:1'], py=twm['py', 'ip4:1'],
              betx=twm['betx', 'ip4:1'], bety=twm['bety', 'ip4:1'],
              alfx=twm['alfx', 'ip4:1'], alfy=twm['alfy', 'ip4:1'],
              dx=twm['dx', 'ip4:1'], dpx=twm['dpx', 'ip4:1'],
              dy=twm['dy', 'ip4:1'], dpy=twm['dpy', 'ip4:1'],
              ddx=twm['ddx', 'ip4:1']*2, ddy=twm['ddy', 'ip4:1']*2,
              ddpx=twm['ddpx', 'ip4:1']*2, ddpy=twm['ddpy', 'ip4:1']*2,
              compute_chromatic_properties=True)

ddy_mad = np.interp(tw_fw.s, twm.s, twm.ddy)
ddx_mad = np.interp(tw_fw.s, twm.s, twm.ddx)

nlchr = line.get_non_linear_chromaticity(delta0_range=(-1e-3, 1e-3),
                                        num_delta=21, fit_order=1, method='4d')

tw_mad = []
for dd in nlchr.delta0:
    mad.input(f'twiss, deltap={dd};')
    tw_mad.append(mad.table.twiss.dframe())

x_mad = np.array([tw.x[0] for tw in tw_mad])
y_mad = np.array([tw.y[0] for tw in tw_mad])
x_xs = np.array([tw.x[0] for tw in nlchr.twiss])
y_xs = np.array([tw.y[0] for tw in nlchr.twiss])

import matplotlib.pyplot as plt
plt.figure(1)
ax1 = plt.subplot(4, 1, 1)
plt.plot(tw.s, tw.ddx)
plt.plot(tw_chr.s, 2 * tw_chr.ddx) # in MAD-X ddx = 0.5 d2x/ddelta2
plt.plot(tw_fw.s, tw_fw.ddx, '.')
plt.plot(tw_bw.s, tw_bw.ddx, '-')
plt.ylabel('ddx')
plt.subplot(4, 1, 2, sharex=ax1)
plt.plot(tw.s, tw.ddy)
plt.plot(tw_chr.s, tw_chr.ddy * 2)
plt.plot(tw_fw.s, tw_fw.ddy, '.')
plt.plot(tw_bw.s, tw_bw.ddy, '-')
plt.ylabel('ddy')
plt.subplot(4, 1, 3, sharex=ax1)
plt.plot(tw.s, tw.ddpx)
plt.plot(tw_chr.s, tw_chr.ddpx * 2)
plt.plot(tw_fw.s, tw_fw.ddpx, '.')
plt.plot(tw_bw.s, tw_bw.ddpx, '-')
plt.ylabel('ddpx')
plt.subplot(4, 1, 4, sharex=ax1)
plt.plot(tw.s, tw.ddpy)
plt.plot(tw_chr.s, tw_chr.ddpy * 2)
plt.plot(tw_fw.s, tw_fw.ddpy, '.')
plt.plot(tw_bw.s, tw_bw.ddpy, '-')
plt.ylabel('ddpy')

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(nlchr.delta0*1e3, x_xs, label='xsuite')
plt.plot(nlchr.delta0*1e3, x_mad, label='madx')
plt.ylabel('x')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(nlchr.delta0*1e3, y_xs, label='xsuite')
plt.plot(nlchr.delta0*1e3, y_mad, label='madx')
plt.ylabel('y')
plt.legend()
plt.xlabel(r'$\delta$ [1e-3]')

plt.show()