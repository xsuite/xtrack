import sys
import pathlib
import numpy as np

import xtrack as xt
import xpart as xp
import xobjects as xo

from cpymad.madx import Madx

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

def test_twiss():

    path = test_data_folder.joinpath('hllhc14_input_mad/')

    mad = Madx(command_log="mad_final.log")
    mad.call(str(path.joinpath("final_seq.madx")))
    mad.use(sequence="lhcb1")
    mad.twiss()
    mad.readtable(file=str(path.joinpath("final_errors.tfs")),
                  table="errtab")
    mad.seterr(table="errtab")
    mad.set(format=".15g")
    twmad = mad.twiss(rmatrix=True, chrom=True)

    line = xt.Line.from_madx_sequence(
            mad.sequence['lhcb1'], apply_madx_errors=True)
    line.elements[10].iscollective = True # we make it artificially collective to test this option
    part_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                            gamma0=mad.sequence.lhcb1.beam.gamma)

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        tracker = xt.Tracker(_context=context, line=line)
        assert tracker.iscollective

        twxt = tracker.twiss(particle_ref=part_ref)
        assert np.isclose(np.modf(mad.table.summ.q1)[0], twxt['qx'], rtol=1e-4, atol=0)
        assert np.isclose(np.modf(mad.table.summ.q2)[0], twxt['qy'], rtol=1e-4, atol=0)
        assert np.isclose(mad.table.summ.dq1, twxt['dqx'], atol=0.1, rtol=0)
        assert np.isclose(mad.table.summ.dq2, twxt['dqy'], atol=0.1, rtol=0)
        assert np.isclose(mad.table.summ.alfa[0],
            twxt['momentum_compaction_factor'],
            atol=2e-10, rtol=0)

        for name in ['mb.b19r5.b1', 'mb.b19r1.b1',
                    'ip1', 'ip2', 'ip5', 'ip8',
                    'mbxf.4l1', 'mbxf.4l5']:

            imad = list(twmad['name']).index(name+':1')
            ixt = list(twxt['name']).index(name)

            assert np.isclose(twxt['betx'][ixt], twmad['betx'][imad],
                            atol=0, rtol=3e-4)
            assert np.isclose(twxt['bety'][ixt], twmad['bety'][imad],
                            atol=0, rtol=3e-4)
            assert np.isclose(twxt['alfx'][ixt], twmad['alfx'][imad],
                              atol=1e-1, rtol=0)
            assert np.isclose(twxt['alfy'][ixt], twmad['alfy'][imad],
                              atol=1e-1, rtol=0)
            assert np.isclose(twxt['dx'][ixt], twmad['dx'][imad],
                              atol=1e-2, rtol=0)
            assert np.isclose(twxt['dy'][ixt], twmad['dy'][imad],
                              atol=1e-2, rtol=0)
            assert np.isclose(twxt['dpx'][ixt], twmad['dpx'][imad],
                              atol=3e-4, rtol=0)
            assert np.isclose(twxt['dpy'][ixt], twmad['dpy'][imad],
                              atol=3e-4, rtol=0)

            assert np.isclose(twxt['s'][ixt], twmad['s'][imad], atol=5e-6, rtol=0)
            assert np.isclose(twxt['x'][ixt], twmad['x'][imad], atol=5e-6, rtol=0)
            assert np.isclose(twxt['y'][ixt], twmad['y'][imad], atol=5e-6, rtol=0)
            assert np.isclose(twxt['px'][ixt], twmad['px'][imad], atol=1e-7, rtol=0)
            assert np.isclose(twxt['py'][ixt], twmad['py'][imad], atol=1e-7, rtol=0)

