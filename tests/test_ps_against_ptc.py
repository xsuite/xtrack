import pathlib

import numpy as np
from cpymad.madx import Madx

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

@for_all_test_contexts
def test_ps_against_ptc(test_context):

    # Verify correct result with Yoshida integration in CombinedFunctionMagnet

    mad = Madx(stdout=False)

    mad.call(str(test_data_folder / 'ps_sftpro/ps.seq'))
    mad.call(str(test_data_folder / 'ps_sftpro/ps_hs_sftpro.str'))
    mad.input('beam, particle=proton, pc = 14.0; BRHO = BEAM->PC * 3.3356;')
    mad.use('ps')
    seq = mad.sequence.ps

    line = xt.Line.from_madx_sequence(seq, deferred_expressions=True)
    line.particle_ref = xt.Particles(gamma0=seq.beam.gamma,
                                    mass0=seq.beam.mass * 1e9,
                                    q0=seq.beam.charge)
    line.build_tracker(_context=test_context)

    assert isinstance(line['pr.bhr00.f'], xt.Bend)
    assert line['pr.bhr00.f'].model == 'adaptive'
    assert line['pr.bhr00.f'].num_multipole_kicks == 0

    tw = line.twiss(method='4d')

    delta_chrom = 1e-4
    mad.input(f'''
    ptc_create_universe;
    ptc_create_layout, time=false, model=1, exact=true, method=6, nst=10;
        select, flag=ptc_twiss, clear;
        select, flag=ptc_twiss, column=name,keyword,s,l,x,px,y,py,beta11,beta22,disp1,k1l;
        ptc_twiss, closed_orbit, icase=56, no=2, deltap=0, table=ptc_twiss,
                summary_table=ptc_twiss_summary, slice_magnets=true;
        ptc_twiss, closed_orbit, icase=56, no=2, deltap={delta_chrom:e}, table=ptc_twiss_pdp,
                summary_table=ptc_twiss_summary_pdp, slice_magnets=true;
        ptc_twiss, closed_orbit, icase=56, no=2, deltap={-delta_chrom:e}, table=ptc_twiss_mdp,
                summary_table=ptc_twiss_summary_mdp, slice_magnets=true;
    ptc_end;
    ''')

    qx_ptc = mad.table.ptc_twiss.mu1[-1]
    qy_ptc = mad.table.ptc_twiss.mu2[-1]
    dq1_ptc = (mad.table.ptc_twiss_pdp.mu1[-1] - mad.table.ptc_twiss_mdp.mu1[-1]) / (2 * delta_chrom)
    dq2_ptc = (mad.table.ptc_twiss_pdp.mu2[-1] - mad.table.ptc_twiss_mdp.mu2[-1]) / (2 * delta_chrom)

    ddq1_ptc = (mad.table.ptc_twiss_pdp.mu1[-1] + mad.table.ptc_twiss_mdp.mu1[-1]
                - 2 * mad.table.ptc_twiss.mu1[-1]) / delta_chrom**2
    ddq2_ptc = (mad.table.ptc_twiss_pdp.mu2[-1] + mad.table.ptc_twiss_mdp.mu2[-1]
                - 2 * mad.table.ptc_twiss.mu2[-1]) / delta_chrom**2

    tptc = xt.Table(mad.table.ptc_twiss)
    tptc_p = xt.Table(mad.table.ptc_twiss_pdp)
    tptc_m = xt.Table(mad.table.ptc_twiss_mdp)

    fp = 1 + delta_chrom
    fm = 1 - delta_chrom

    # The MAD-X PTC interface rescales the beta functions with (1 + deltap)
    # see: https://github.com/MethodicalAcceleratorDesign/MAD-X/blob/eb495b4f926db53f3cd05133638860f910f42fe2/src/madx_ptc_twiss.f90#L1982
    # We need to undo that
    beta11_p = tptc_p.beta11 / fp
    beta11_m = tptc_m.beta11 / fm
    beta22_p = tptc_p.beta22 / fp
    beta22_m = tptc_m.beta22 / fm
    alfa11_p = tptc_p.alfa11
    alfa11_m = tptc_m.alfa11
    alfa22_p = tptc_p.alfa22
    alfa22_m = tptc_m.alfa22

    dx_ptc = (tptc_p.x - tptc_m.x) / (2 * delta_chrom)
    dy_ptc = (tptc_p.y - tptc_m.y) / (2 * delta_chrom)

    betx = 0.5 * (beta11_p + beta11_m)
    bety = 0.5 * (beta22_p + beta22_m)
    alfx = 0.5 * (alfa11_p + alfa11_m)
    alfy = 0.5 * (alfa22_p + alfa22_m)
    d_betx = (beta11_p - beta11_m) / (2 * delta_chrom)
    d_bety = (beta22_p - beta22_m) / (2 * delta_chrom)
    d_alfx = (alfa11_p - alfa11_m) / (2 * delta_chrom)
    d_alfy = (alfa22_p - alfa22_m) / (2 * delta_chrom)

    bx_ptc = d_betx / betx
    by_ptc = d_bety / bety
    ax_ptc = d_alfx - d_betx * alfx / betx
    ay_ptc = d_alfy - d_bety * alfy / bety
    wx_ptc = np.sqrt(ax_ptc**2 + bx_ptc**2)
    wy_ptc = np.sqrt(ay_ptc**2 + by_ptc**2)

    xo.assert_allclose(tw.qx, qx_ptc, atol=1e-4, rtol=0)
    xo.assert_allclose(tw.qy, qy_ptc, atol=1e-4, rtol=0)

    xo.assert_allclose(tw.dqx, dq1_ptc, atol=1e-2, rtol=0)
    xo.assert_allclose(tw.dqy, dq2_ptc, atol=1e-2, rtol=0)

    xo.assert_allclose(tw.dqx, dq1_ptc, atol=1e-2, rtol=0)
    xo.assert_allclose(tw.dqy, dq2_ptc, atol=1e-2, rtol=0)

    nlchr = line.get_non_linear_chromaticity()
    xo.assert_allclose(nlchr['ddqx'], ddq1_ptc, atol=0, rtol=5e-3)
    xo.assert_allclose(nlchr['ddqy'], ddq2_ptc, atol=0, rtol=5e-3)

    xo.assert_allclose(nlchr['dqx'], dq1_ptc, atol=0, rtol=5e-3)
    xo.assert_allclose(nlchr['dqy'], dq2_ptc, atol=0, rtol=5e-3)

    xo.assert_allclose(nlchr.dnqx[:3], [tw.qx, nlchr.dqx, nlchr.ddqx], atol=0, rtol=1e-6)
    xo.assert_allclose(nlchr.dnqy[:3], [tw.qy, nlchr.dqy, nlchr.ddqy], atol=0, rtol=1e-6)

    markers_ptc = tptc.rows[tptc.keyword == 'marker']
    markers_common_ptc = [nn for nn in markers_ptc.name if nn.split(':')[0] in tw.name]
    markers_common_xs = [nn.split(':')[0] for nn in markers_common_ptc]
    mask_ptc = tptc.rows.mask[markers_common_ptc]
    mask_xs = tw.rows.mask[markers_common_xs]

    xo.assert_allclose(tw.ax_chrom[mask_xs], ax_ptc[mask_ptc], atol=1e-2, rtol=0)
    xo.assert_allclose(tw.bx_chrom[mask_xs], bx_ptc[mask_ptc], atol=1e-2, rtol=0)
    xo.assert_allclose(tw.ay_chrom[mask_xs], ay_ptc[mask_ptc], atol=1e-2, rtol=0)
    xo.assert_allclose(tw.by_chrom[mask_xs], by_ptc[mask_ptc], atol=1e-2, rtol=0)
    xo.assert_allclose(tw.wx_chrom[mask_xs], wx_ptc[mask_ptc], atol=1e-2, rtol=0)
    xo.assert_allclose(tw.wy_chrom[mask_xs], wy_ptc[mask_ptc], atol=1e-2, rtol=0)

    xo.assert_allclose(tw.betx[mask_xs], betx[mask_ptc], rtol=1e-4, atol=0)
    xo.assert_allclose(tw.bety[mask_xs], bety[mask_ptc], rtol=1e-4, atol=0)
    xo.assert_allclose(tw.dx[mask_xs], dx_ptc[mask_ptc], rtol=1e-4, atol=1e-6)
    xo.assert_allclose(tw.dy[mask_xs], dy_ptc[mask_ptc], rtol=1e-4, atol=1e-6)
