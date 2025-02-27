
from cpymad.madx import Madx
import xtrack as xt
import xobjects as xo

assert_allclose = xo.assert_allclose

# def assert_allclose(a, b, **kwargs):
#     print(a, b)

test_data_folder = '../../test_data'

test_data_folder_str = str(test_data_folder)

mad1=Madx(stdout=False)
mad1.call(test_data_folder_str + '/hllhc15_thick/lhc.seq')
mad1.call(test_data_folder_str + '/hllhc15_thick/hllhc_sequence.madx')
mad1.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
mad1.use('lhcb1')
mad1.input('beam, sequence=lhcb2, particle=proton, energy=7000, bv=-1;')
mad1.use('lhcb2')
mad1.call(test_data_folder_str + '/hllhc15_thick/opt_round_150_1500.madx')

mad1.globals['on_x1'] = 100 # Check kicker expressions
mad1.globals['on_sep2'] = 2 # Check kicker expressions
mad1.globals['on_x5'] = 123 # Check kicker expressions

mad1.globals['kqs.a23b1'] = 1e-4 # Check skew quad expressions
mad1.globals['kqs.a12b2'] = 1e-4 # Check skew quad expressions

mad1.globals['ksf.b1'] = 1e-3  # Check sext expressions
mad1.globals['ksf.b2'] = 1e-3  # Check sext expressions

mad1.globals['kss.a45b1'] = 1e-4 # Check skew sext expressions
mad1.globals['kss.a45b2'] = 1e-4 # Check skew sext expressions

mad1.globals['kof.a34b1'] = 3 # Check oct expressions
mad1.globals['kof.a34b2'] = 3 # Check oct expressions

mad1.globals['kcosx3.l2'] = 5 # Check skew oct expressions

mad1.globals['kcdx3.r1'] = 1e-4 # Check thin decapole expressions

mad1.globals['kcdsx3.r1'] = 1e-4 # Check thin skew decapole expressions

mad1.globals['kctx3.l1'] = 1e-5 # Check thin dodecapole expressions

mad1.globals['kctsx3.r1'] = 1e-5 # Check thin skew dodecapole expressions

mad1.input('twiss, sequence=lhcb1, table=twisslhcb1;')
mad1.input('twiss, sequence=lhcb2, table=twisslhcb2;')

twm1 = xt.Table(mad1.table.twisslhcb1)
twm2 = xt.Table(mad1.table.twisslhcb2)

collider = xt.Environment.from_madx(madx=mad1)
tw = collider.twiss(strengths=True, method='4d')

# Normal strengths
assert_allclose(twm1['k0l', 'mb.a20r8.b1:1'], tw.lhcb1['k0l', 'mb.a20r8.b1'], rtol=0, atol=1e-14)
assert_allclose(twm2['k0l', 'mb.a20r8.b2:1'], tw.lhcb2['k0l', 'mb.a20r8.b2'], rtol=0, atol=1e-14)

assert_allclose(twm1['k1l', 'mq.22l3.b1:1'], tw.lhcb1['k1l', 'mq.22l3.b1'], rtol=0, atol=1e-14)
assert_allclose(twm2['k1l', 'mq.22l3.b2:1'], tw.lhcb2['k1l', 'mq.22l3.b2'], rtol=0, atol=1e-14)

assert_allclose(twm1['k2l', 'ms.16l1.b1:1'], tw.lhcb1['k2l', 'ms.16l1.b1'], rtol=0, atol=1e-14)
assert_allclose(twm2['k2l', 'ms.16l1.b2:1'], tw.lhcb2['k2l', 'ms.16l1.b2'], rtol=0, atol=1e-14)

assert_allclose(twm1['k3l', 'mo.25l4.b1:1'], tw.lhcb1['k3l', 'mo.25l4.b1'], rtol=0, atol=1e-14)
assert_allclose(twm2['k3l', 'mo.24l4.b2:1'], tw.lhcb2['k3l', 'mo.24l4.b2'], rtol=0, atol=1e-14)

assert_allclose(twm1['k4l', 'mcdxf.3r1:1'], tw.lhcb1['k4l', 'mcdxf.3r1'], rtol=0, atol=1e-14)
assert_allclose(twm2['k4l', 'mcdxf.3r1:1'], tw.lhcb2['k4l', 'mcdxf.3r1'], rtol=0, atol=1e-14)

assert_allclose(twm1['k5l', 'mctxf.3l1:1'], tw.lhcb1['k5l', 'mctxf.3l1'], rtol=0, atol=1e-14)
assert_allclose(twm2['k5l', 'mctxf.3l1:1'], tw.lhcb2['k5l', 'mctxf.3l1'], rtol=0, atol=1e-14)

# Skew strengths
assert_allclose(twm1['k1sl', 'mqs.27l3.b1:1'], tw.lhcb1['k1sl', 'mqs.27l3.b1'], rtol=0, atol=1e-14)
assert_allclose(twm2['k1sl', 'mqs.23l2.b2:1'], tw.lhcb2['k1sl', 'mqs.23l2.b2'], rtol=0, atol=1e-14)

assert_allclose(twm1['k2sl', 'mss.28l5.b1:1'], tw.lhcb1['k2sl', 'mss.28l5.b1'], rtol=0, atol=1e-14)
assert_allclose(twm2['k2sl', 'mss.33l5.b2:1'], tw.lhcb2['k2sl', 'mss.33l5.b2'], rtol=0, atol=1e-14)

assert_allclose(twm1['k3sl', 'mcosx.3l2:1'], tw.lhcb1['k3sl', 'mcosx.3l2'], rtol=0, atol=1e-14)
assert_allclose(twm2['k3sl', 'mcosx.3l2:1'], tw.lhcb2['k3sl', 'mcosx.3l2'], rtol=0, atol=1e-14)

assert_allclose(twm1['k4sl', 'mcdsxf.3r1:1'], tw.lhcb1['k4sl', 'mcdsxf.3r1'], rtol=0, atol=1e-14)
assert_allclose(twm2['k4sl', 'mcdsxf.3r1:1'], tw.lhcb2['k4sl', 'mcdsxf.3r1'], rtol=0, atol=1e-14)

assert_allclose(twm1['k5sl', 'mctsxf.3r1:1'], tw.lhcb1['k5sl', 'mctsxf.3r1'], rtol=0, atol=1e-14)
assert_allclose(twm2['k5sl', 'mctsxf.3r1:1'], tw.lhcb2['k5sl', 'mctsxf.3r1'], rtol=0, atol=1e-14)