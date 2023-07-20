import xtrack as xt
import numpy as np

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()

# Check varval behaviour
collider.vars['on_x1'] = 40
assert collider.varval['on_x1'] == 40
assert collider.lhcb1.vars['on_x1']._value == 40
assert collider.lhcb2.vars['on_x1']._value == 40
assert collider.lhcb1.varval['on_x1'] == 40
assert collider.lhcb2.varval['on_x1'] == 40

collider.varval['on_x1'] = 50
assert collider.vars['on_x1']._value == 50
assert collider.lhcb1.vars['on_x1']._value == 50
assert collider.lhcb2.vars['on_x1']._value == 50
assert collider.lhcb1.varval['on_x1'] == 50
assert collider.lhcb2.varval['on_x1'] == 50

collider.lhcb1.varval['on_x1'] = 60
assert collider.vars['on_x1']._value == 60
assert collider.lhcb1.vars['on_x1']._value == 60
assert collider.lhcb2.vars['on_x1']._value == 60
assert collider.lhcb1.varval['on_x1'] == 60
assert collider.lhcb2.varval['on_x1'] == 60

collider.lhcb2.varval['on_x1'] = 70
assert collider.vars['on_x1']._value == 70
assert collider.lhcb1.vars['on_x1']._value == 70
assert collider.lhcb2.vars['on_x1']._value == 70
assert collider.lhcb1.varval['on_x1'] == 70
assert collider.lhcb2.varval['on_x1'] == 70

collider.vars['on_disp'] = 0 # more precise angle
assert np.isclose(collider.twiss().lhcb1['px', 'ip1'], 70e-6, atol=1e-8, rtol=0)
assert np.isclose(collider.twiss().lhcb2['px', 'ip1'], -70e-6, atol=1e-8, rtol=0)

collider.vars.load_madx_optics_file(
    '../../test_data/hllhc15_thick/opt_round_300_1500.madx')

collider._xdeps_manager.verify()

tw = collider.twiss()
assert np.isclose(tw.lhcb1.qx, 62.31000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1.qy, 60.32000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2.qx, 62.31000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2.qy, 60.32000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1['betx', 'ip1'], 0.30, atol=0, rtol=1e-4)
assert np.isclose(tw.lhcb1['bety', 'ip1'], 0.30, atol=0, rtol=1e-4)
assert np.isclose(tw.lhcb2['betx', 'ip1'], 0.30, atol=0, rtol=1e-4)
assert np.isclose(tw.lhcb2['bety', 'ip1'], 0.30, atol=0, rtol=1e-4)

# Check a knob
collider.vars['on_x1'] = 30
collider.vars['on_disp'] = 0
tw = collider.twiss()
assert np.isclose(tw.lhcb1['px', 'ip1'], 30e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip1'], -30e-6, atol=1e-9, rtol=0)

collider.vars.load_madx_optics_file(
    '../../test_data/hllhc15_thick/opt_round_150_1500.madx')

tw = collider.twiss()
assert np.isclose(tw.lhcb1.qx, 62.31000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1.qy, 60.32000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2.qx, 62.31000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2.qy, 60.32000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1['betx', 'ip1'], 0.15, atol=0, rtol=1e-6)
assert np.isclose(tw.lhcb1['bety', 'ip1'], 0.15, atol=0, rtol=1e-6)
assert np.isclose(tw.lhcb2['betx', 'ip1'], 0.15, atol=0, rtol=1e-6)
assert np.isclose(tw.lhcb2['bety', 'ip1'], 0.15, atol=0, rtol=1e-6)

# Check a knob
collider.vars['on_x1'] = 10
collider.vars['on_disp'] = 0
tw = collider.twiss()
assert np.isclose(tw.lhcb1['px', 'ip1'], 10e-6, atol=1e-9, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip1'], -10e-6, atol=1e-9, rtol=0)
