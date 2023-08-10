import numpy as np

import xtrack as xt
import xdeps as xd

import xtrack._temp.lhc_match as lm

collider = xt.Multiline.from_json('collider_01_changed_ip15_phase.json')
collider.build_trackers()

lm.gen_madx_optics_file_auto(collider, 'opt_round_150_1500_xs.madx')


# Try loading optics file in MAD-X
from cpymad.madx import Madx

mad=Madx()
mad.call('../../test_data/hllhc15_thick/lhc.seq')
mad.call('../../test_data/hllhc15_thick/hllhc_sequence.madx')
mad.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
mad.use('lhcb1')
mad.input('beam, sequence=lhcb2, particle=proton, energy=7000, bv=-1;')
mad.use('lhcb2')
mad.call("opt_round_150_1500_xs.madx")

mad.input('twiss, sequence=lhcb1, table=twb1')
mad.input('twiss, sequence=lhcb2, table=twb2')
twmad_b1 = xd.Table(mad.table.twb1)
twmad_b2 = xd.Table(mad.table.twb2)

assert np.isclose(twmad_b1['betx', 'ip1:1'], 0.15, rtol=1e-8, atol=0)
assert np.isclose(twmad_b1['bety', 'ip1:1'], 0.15, rtol=1e-8, atol=0)
assert np.isclose(twmad_b2['betx', 'ip1:1'], 0.15, rtol=1e-8, atol=0)
assert np.isclose(twmad_b2['bety', 'ip1:1'], 0.15, rtol=1e-8, atol=0)

import xdeps as xd

twmad_b1.rows['ip.*'].cols['betx bety x y px py'].show()
twmad_b2.rows['ip.*'].cols['betx bety x y px py'].show()

# Test orbit knobs
mad.globals.on_x8 = 100
mad.globals.on_x2 = 110

mad.input('twiss, sequence=lhcb1, table=twb1')
mad.input('twiss, sequence=lhcb2, table=twb2')
twmad_b1 = xd.Table(mad.table.twb1)
twmad_b2 = xd.Table(mad.table.twb2)

assert np.isclose(twmad_b1['px', 'ip8:1'], 100e-6, rtol=0, atol=1e-10)
assert np.isclose(twmad_b2['px', 'ip8:1'], -100e-6, rtol=0, atol=1e-10)
assert np.isclose(twmad_b1['py', 'ip2:1'], 110e-6, rtol=0, atol=1e-10)
assert np.isclose(twmad_b2['py', 'ip2:1'], -110e-6, rtol=0, atol=1e-10)

# Match tunes and chromaticity in the Xsuite model
opt = collider.match(
    solve=False,
    vary=[
        xt.VaryList(['kqtf.b1', 'kqtd.b1', 'ksf.b1', 'ksd.b1'], step=1e-7),
        xt.VaryList(['kqtf.b2', 'kqtd.b2', 'ksf.b2', 'ksd.b2'], step=1e-7)],
    targets = [
        xt.TargetSet(line='lhcb1', qx=62.315, qy=60.325, tol=1e-10),
        xt.TargetSet(line='lhcb1', dqx=10.0, dqy=12.0, tol=1e-5),
        xt.TargetSet(line='lhcb2', qx=62.316, qy=60.324, tol=1e-10),
        xt.TargetSet(line='lhcb2', dqx=9.0, dqy=11.0, tol=1e-5)])
opt.solve()

# Transfer knobs to madx model and check matched values

for kk, vv in opt.get_knob_values().items():
    mad.globals[kk] = vv

mad.input('twiss, sequence=lhcb1, table=twb1')
mad.input('twiss, sequence=lhcb2, table=twb2')

assert np.isclose(mad.table.twb1.summary.q1, 62.315, rtol=0, atol=1e-6)
assert np.isclose(mad.table.twb1.summary.q2, 60.325, rtol=0, atol=1e-6)
assert np.isclose(mad.table.twb2.summary.q1, 62.316, rtol=0, atol=1e-6)
assert np.isclose(mad.table.twb2.summary.q2, 60.324, rtol=0, atol=1e-6)
assert np.isclose(mad.table.twb1.summary.dq1, 10.0, rtol=0, atol=0.3)
assert np.isclose(mad.table.twb1.summary.dq2, 12.0, rtol=0, atol=0.3)
assert np.isclose(mad.table.twb2.summary.dq1, 9.0, rtol=0, atol=0.3)
assert np.isclose(mad.table.twb2.summary.dq2, 11.0, rtol=0, atol=0.3)

