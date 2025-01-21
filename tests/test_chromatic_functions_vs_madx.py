import pathlib

from cpymad.madx import Madx

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

import numpy as np

@for_all_test_contexts
def test_chromatic_functions_vs_madx(test_context):

    collider = xt.Environment.from_json(test_data_folder /
                                'hllhc15_thick/hllhc15_collider_thick.json')
    collider['lhcb1'].twiss_default['method'] = '4d'
    collider['lhcb2'].twiss_default['method'] = '4d'


    # Slice to have the exit markers
    collider.lhcb1.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(1, mode='thick'))])
    collider.lhcb2.slice_thick_elements(
        slicing_strategies=[xt.Strategy(xt.Uniform(1, mode='thick'))])

    collider.build_trackers(_context=test_context)

    mad = Madx(stdout=False)
    mad.input(f"""
    call,file="{str(test_data_folder)}/hllhc15_thick/lhc.seq";
    call,file="{str(test_data_folder)}/hllhc15_thick/hllhc_sequence.madx";
    beam, sequence=lhcb1, particle=proton, pc=7000;
    beam, sequence=lhcb2, particle=proton, pc=7000, bv=-1;
    call,file="{str(test_data_folder)}/hllhc15_thick/opt_round_150_1500.madx";
    """)

    for line_name in ['lhcb1', 'lhcb2']:

        if line_name == 'lhcb1':
            line = collider.lhcb1
            mad.use(sequence="lhcb1")
        elif line_name == 'lhcb2':
            line = collider.lhcb2
            mad.use(sequence="lhcb2")
            line.twiss_default['reverse'] = True

        tw = line.twiss()
        twmad = mad.twiss(chrom=True)

        tw_test = tw.rows['.*_exit']

        wx_ref = np.interp(tw_test['s'], twmad['s'], twmad['wx'])
        wy_ref = np.interp(tw_test['s'], twmad['s'], twmad['wy'])
        phix_ref = np.interp(tw_test['s'], twmad['s'], twmad['phix'])
        phiy_ref = np.interp(tw_test['s'], twmad['s'], twmad['phiy'])

        zx_ref = wx_ref * np.exp(1j * 2 * np.pi * phix_ref)
        zy_ref = wy_ref * np.exp(1j * 2 * np.pi * phiy_ref)
        ax_ref = np.imag(zx_ref)
        ay_ref = np.imag(zy_ref)
        bx_ref = np.real(zx_ref)
        by_ref = np.real(zy_ref)

        xo.assert_allclose(tw_test.wx_chrom, wx_ref, rtol=0, atol=2e-3 * np.max(wx_ref))
        xo.assert_allclose(tw_test.wy_chrom, wy_ref, rtol=0, atol=2e-3 * np.max(wy_ref))
        xo.assert_allclose(tw_test.ax_chrom, ax_ref, rtol=0, atol=2e-3 * np.max(ax_ref))
        xo.assert_allclose(tw_test.ay_chrom, ay_ref, rtol=0, atol=2e-3 * np.max(ay_ref))
        xo.assert_allclose(tw_test.bx_chrom, bx_ref, rtol=0, atol=2e-3 * np.max(bx_ref))
        xo.assert_allclose(tw_test.by_chrom, by_ref, rtol=0, atol=2e-3 * np.max(by_ref))

        # Open twiss
        init = tw.get_twiss_init('ip3')
        tw_open = line.twiss(start='ip3', end='ip6', init=init,
                            compute_chromatic_properties=True)

        tw_ref_open = tw.rows['ip3':'ip6']
        xo.assert_allclose(tw_open.wx_chrom[:-1], tw_ref_open.wx_chrom,
                        rtol=0, atol=2e-3 * np.max(tw_ref_open.wx_chrom))
        xo.assert_allclose(tw_open.wy_chrom[:-1], tw_ref_open.wy_chrom,
                        rtol=0, atol=2e-3 * np.max(tw_ref_open.wy_chrom))
        xo.assert_allclose(tw_open.ax_chrom[:-1], tw_ref_open.ax_chrom,
                            rtol=0, atol=2e-3 * np.max(tw_ref_open.ax_chrom))
        xo.assert_allclose(tw_open.ay_chrom[:-1], tw_ref_open.ay_chrom,
                                rtol=0, atol=2e-3 * np.max(tw_ref_open.ay_chrom))
        xo.assert_allclose(tw_open.bx_chrom[:-1], tw_ref_open.bx_chrom,
                                rtol=0, atol=2e-3 * np.max(tw_ref_open.bx_chrom))