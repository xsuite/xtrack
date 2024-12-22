# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xtrack as xt
import numpy as np

collider = xt.Environment.from_json(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')
collider.build_trackers()

collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

line = collider.lhcb2 # <- use lhcb2 to test the reverse option

tw8_closed = line.twiss(start='ip8')
tw8_open = line.twiss(start='ip8', betx=1.5, bety=1.5)

tw_part = line.twiss(start='ip8', end='ip2', zero_at='ip1', init='full_periodic')

# Test get strengths (to be moved to another script)
line.get_strengths().rows['mbw\..*l3.b2'].cols['k0l angle_rad']
# is:
# Table: 6 rows, 3 cols
# name                            k0l    angle_rad
# mbw.f6l3.b2            -0.000188729 -0.000188729
# mbw.e6l3.b2            -0.000188729 -0.000188729
# mbw.d6l3.b2            -0.000188729 -0.000188729
# mbw.c6l3.b2             0.000188729  0.000188729
# mbw.b6l3.b2             0.000188729  0.000188729
# mbw.a6l3.b2             0.000188729  0.000188729

line.get_strengths(reverse=False).rows['mbw\..*l3.b2'].cols['k0l angle_rad']
# is:
# Table: 6 rows, 3 cols
# name                            k0l    angle_rad
# mbw.a6l3.b2            -0.000188729 -0.000188729
# mbw.b6l3.b2            -0.000188729 -0.000188729
# mbw.c6l3.b2            -0.000188729 -0.000188729
# mbw.d6l3.b2             0.000188729  0.000188729
# mbw.e6l3.b2             0.000188729  0.000188729
# mbw.f6l3.b2             0.000188729  0.000188729

import xobjects as xo
str_table_rev = line.get_strengths() # Takes reverse from twiss_default
xo.assert_allclose(line['mbw.a6l3.b2'].k0,
        -str_table_rev['k0l', 'mbw.a6l3.b2'] / str_table_rev['length', 'mbw.a6l3.b2'],
        rtol=0, atol=1e-14)
xo.assert_allclose(line['mbw.a6l3.b2'].h,
        -str_table_rev['angle_rad', 'mbw.a6l3.b2'] / str_table_rev['length', 'mbw.a6l3.b2'],
        rtol=0, atol=1e-14)

str_table = line.get_strengths(reverse=False) # Takes reverse from twiss_default
xo.assert_allclose(line['mbw.a6l3.b2'].k0,
        str_table['k0l', 'mbw.a6l3.b2'] / str_table['length', 'mbw.a6l3.b2'],
        rtol=0, atol=1e-14)
xo.assert_allclose(line['mbw.a6l3.b2'].h,
        str_table['angle_rad', 'mbw.a6l3.b2'] / str_table['length', 'mbw.a6l3.b2'],
        rtol=0, atol=1e-14)

tw = line.twiss()

for tw8 in [tw8_closed, tw8_open]:
    assert tw8.name[-1] == '_end_point'
    assert np.all(tw8.rows['ip.?'].name
        == np.array(['ip8', 'ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7']))

    for nn in ['s', 'mux', 'muy']:
        assert np.all(np.diff(tw8.rows['ip.?'][nn]) > 0)
        assert tw8[nn][0] == 0
        xo.assert_allclose(tw8[nn][-1], tw[nn][-1], rtol=1e-12, atol=5e-7)

    xo.assert_allclose(
            tw8['betx', ['ip8', 'ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7']],
            tw[ 'betx', ['ip8', 'ip1', 'ip2', 'ip3', 'ip4', 'ip5', 'ip6', 'ip7']],
            rtol=1e-5, atol=0)

tw_part1 = line.twiss(start='ip8', end='ip2', zero_at='ip1', init='full_periodic')

assert tw_part1.name[0] == 'ip8'
assert tw_part1.name[-2] == 'ip2'
assert tw_part1.name[-1] == '_end_point'

for kk in ['s', 'mux', 'muy']:
    tw_part1[kk, 'ip1'] == 0.
    assert np.all(np.diff(tw_part1[kk]) >= 0)
    xo.assert_allclose(
        tw_part1[kk, 'ip8'], -(tw[kk, '_end_point'] - tw[kk, 'ip8']),
        rtol=1e-12, atol=5e-7)
    xo.assert_allclose(
        tw_part1[kk, 'ip2'], tw[kk, 'ip2'] - tw[kk, 0],
        rtol=1e-12, atol=5e-7)

tw_part2 = line.twiss(start='ip8', end='ip2', init='full_periodic')

assert tw_part2.name[0] == 'ip8'
assert tw_part2.name[-2] == 'ip2'
assert tw_part2.name[-1] == '_end_point'

for kk in ['s', 'mux', 'muy']:
    tw_part2[kk, 'ip8'] == 0.
    assert np.all(np.diff(tw_part2[kk]) >= 0)
    xo.assert_allclose(
        tw_part2[kk, 'ip2'],
        tw[kk, 'ip2'] - tw[kk, 'ip1'] +(tw[kk, '_end_point'] - tw[kk, 'ip8']),
        rtol=1e-12, atol=5e-7)
