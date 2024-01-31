from cpymad.madx import Madx
import xtrack as xt

mad = Madx()
mad.call('../../test_data/hllhc15_thick/lhc.seq')
mad.call('../../test_data/hllhc15_thick/hllhc_sequence.madx')
mad.call('../../test_data/hllhc15_thick/opt_round_150_1500.madx')
mad.input('beam, sequence=lhcb1, particle=proton, pc=7000;')
mad.use(sequence='lhcb1')
mad.globals['on_x5'] = 100
tw_chr = mad.twiss(chrom=True)

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1, deferred_expressions=True)

line.particle_ref = xt.Particles(mass0=mad.sequence.lhcb1.beam.mass*1e9,
                                    q0=mad.sequence.lhcb1.beam.charge,
                                    gamma0=mad.sequence.lhcb1.beam.gamma)

tw = line.twiss(method='4d')
nlchr = line.get_non_linear_chromaticity(delta0_range=(-1e-4, 1e-4),
                                        num_delta=2, fit_order=1, method='4d')

import matplotlib.pyplot as plt
plt.figure(1)
ax1 = plt.subplot(4, 1, 1)
plt.plot(tw.s, tw.ddx / 2)
plt.plot(tw_chr.s, tw_chr.ddx)
plt.ylabel('ddx')
plt.subplot(4, 1, 2, sharex=ax1)
plt.plot(tw.s, tw.ddy / 2)
plt.plot(tw_chr.s, tw_chr.ddy)
plt.ylabel('ddy')
plt.subplot(4, 1, 3, sharex=ax1)
plt.plot(tw.s, tw.ddpx / 2)
plt.plot(tw_chr.s, tw_chr.ddpx)
plt.ylabel('ddpx')
plt.subplot(4, 1, 4, sharex=ax1)
plt.plot(tw.s, tw.ddpy / 2)
plt.plot(tw_chr.s, tw_chr.ddpy)
plt.ylabel('ddpy')
plt.show()