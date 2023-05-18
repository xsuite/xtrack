# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

import xtrack as xt
import xpart as xp

collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_collider/collider_00_from_mad.json')
collider.build_trackers()

collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

collider.vars['kqs.a23b1'] = 1e-4
collider.lhcb1['mq.10l3.b1..2'].knl[0] = 2e-6
collider.lhcb1['mq.10l3.b1..2'].ksl[0] = -1.5e-6

collider.vars['kqs.a23b2'] = -1e-4
collider.lhcb2['mq.10l3.b2..2'].knl[0] = 3e-6
collider.lhcb2['mq.10l3.b2..2'].ksl[0] = -1.3e-6

line = collider.lhcb2
line_name = 'lhcb2'

# line = collider.lhcb1
# line_name = 'lhcb1'

atols = dict(
    alfx=1e-8, alfy=1e-8,
    dzeta=2e-7, dx=1e-4, dy=1e-4, dpx=1e-5, dpy=1e-5,
    nuzeta=1e-5
)

rtols = dict(
    alfx=5e-9, alfy=5e-8,
    betx=5e-9, bety=5e-9, betx1=5e-9, bety2=5e-9, betx2=1e-8, bety1=1e-8,
    gamx=5e-9, gamy=5e-9,
)

atol_default = 1e-11
rtol_default = 1e-9

for line_name, line in zip(['lhcb1', 'lhcb2'], [collider.lhcb1, collider.lhcb2]):

    tw = line.twiss(r_sigma=0.01)

    tw_init_ip5 = tw.get_twiss_init('ip5')
    tw_init_ip6 = tw.get_twiss_init('ip6')

    tw_forward = line.twiss(ele_start='ip5', ele_stop='ip6',
                            twiss_init=tw_init_ip5)

    tw_backward = line.twiss(ele_start='ip5', ele_stop='ip6',
                            twiss_init=tw_init_ip6)

    assert tw_init_ip5.reference_frame == (
        {'lhcb1': 'proper', 'lhcb2': 'reverse'}[line_name])
    assert tw_init_ip5.element_name == 'ip5'

    tw_part = tw.rows['ip5':'ip6']
    assert tw_part.name[0] == 'ip5'
    assert tw_part.name[-1] == 'ip6'

    for check, tw_test in zip(('fw', 'bw'), [tw_forward, tw_backward]):

        print(f'Checking {line_name} {check}')

        assert tw_test.name[-1] == '_end_point'

        tw_test = tw_test.rows[:-1]
        assert np.all(tw_test.name == tw_part.name)

        for kk in tw_test._data.keys():
            if kk in ['name', 'W_matrix', 'particle_on_co', 'values_at', 'method',
                    'radiation_method', 'reference_frame', 'orientation']:
                continue # tested separately
            atol = atols.get(kk, atol_default)
            rtol = rtols.get(kk, rtol_default)
            assert np.allclose(
                tw_test._data[kk], tw_part._data[kk], rtol=rtol, atol=atol)

        assert tw_test.values_at == tw_part.values_at == 'entry'
        assert tw_test.method == tw_part.method == '4d'
        assert tw_test.radiation_method == tw_part.radiation_method == 'full'
        assert tw_test.reference_frame == tw_part.reference_frame == (
            {'lhcb1': 'proper', 'lhcb2': 'reverse'}[line_name])

        W_matrix_part = tw_part.W_matrix
        W_matrix_test = tw_test.W_matrix

        for ss in range(W_matrix_part.shape[0]):
            this_part = W_matrix_part[ss, :, :]
            this_test = W_matrix_test[ss, :, :]

            for ii in range(this_part.shape[1]):
                assert np.isclose((np.linalg.norm(this_part[ii, :] - this_test[ii, :])
                                /np.linalg.norm(this_part[ii, :])), 0, atol=2e-4)
