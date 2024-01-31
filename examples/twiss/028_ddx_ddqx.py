from cpymad.madx import Madx
import xtrack as xt

mad = Madx()
mad.call('../../test_data/hllhc15_thick/lhc.seq')
mad.call('../../test_data/hllhc15_thick/hllhc_sequence.madx')
mad.call('../../test_data/hllhc15_thick/opt_round_150_1500.madx')
mad.input('beam, sequence=lhcb1, particle=proton, pc=7000;')
mad.use(sequence='lhcb1')
tw_chr = mad.twiss(chrom=True)

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1, deferred_expressions=True)

line.particle_ref = xt.Particles(mass0=mad.sequence.lhcb1.beam.mass*1e9,
                                    q0=mad.sequence.lhcb1.beam.charge,
                                    gamma0=mad.sequence.lhcb1.beam.gamma)

