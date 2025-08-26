import pathlib
from cpymad.madx import Madx
import numpy as np
import xtrack as xt
import xobjects as xo

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()

def test_h6_sps_beamline():

    '''
    This checks the parsing of a sequence defined in madx with refer=exit
    and the TwissTable.get_R_matrix_table() method
    '''

    lattice_path = test_data_folder / 'h6_experimental_line/h6-fm.str'
    sequence_path = test_data_folder / 'h6_experimental_line/h6fm04.seq'

    mad = Madx()
    mad.call(str(lattice_path))
    mad.call(str(sequence_path))

    mad.input('''
    beam, particle=proton, sequence=H6, PC=120.0,
        ex = 2e-07,
        ey = 5e-08;

    use, sequence=H6;
    twiss, chrom=true, rmatrix=true, betx=10, alfx=0, bety=10, alfy=0;
    ''')
    tw_mad = xt.Table(mad.table.twiss)

    env = xt.load(sequence_path)
    env.vars.load(lattice_path)
    line = env['h6']
    line.particle_ref = xt.Particles(p0c=120e9, mass0=xt.PROTON_MASS_EV)
    tt = line.get_table(attr=True)
    line.configure_bend_model(edge='full')

    tw = line.twiss(betx=10, bety=10)

    check_at = [
    't4..centre',
    'vxsv.x0410104',
    'begin.vac',
    'xwca.x0410404',
    'xsci.x0410475',
    'xemc.x0410476',
    'xdwc.x0410488',
    'h6a',
    'h6b',
    'h6c']

    tw_check = tw.rows[check_at]
    tw_mad_check = tw_mad.rows[[nn+':1' for nn in check_at]]

    xo.assert_allclose(tw_check.betx, tw_mad_check.betx, rtol=2e-5, atol=0)
    xo.assert_allclose(tw_check.bety, tw_mad_check.bety, rtol=2e-5, atol=0)
    xo.assert_allclose(tw_check.dx, tw_mad_check.dx, rtol=2e-5, atol=1e-4)
    xo.assert_allclose(tw_check.dy, tw_mad_check.dy, rtol=2e-5, atol=1e-4)

    trm = tw.get_R_matrix_table()
    trm_check = trm.rows[check_at]

    for ii in range(6):
        for jj in range(6):
            rterm_mad = tw_mad_check[f're{ii+1}{jj+1}']
            rterm_xs = trm_check[f'r{ii+1}{jj+1}']
            atol=1e-4*np.max(np.abs(rterm_mad))
            if atol<1e-14:
                atol=1e-14
            xo.assert_allclose(rterm_xs, rterm_mad, rtol=2e-5, atol=atol)