import xtrack as xt
import numpy as np

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()

collider.vars.load_madx_optics_file(
    '../../../hllhc15/ramp/opt_endoframp_500_1500.madx')

tw = collider.twiss()
assert np.isclose(tw.lhcb1.qx, 62.31000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1.qy, 60.32000751, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2.qx, 62.31000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2.qy, 60.32000751, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1['betx', 'ip1'], 0.50, atol=0, rtol=1e-4)
assert np.isclose(tw.lhcb1['bety', 'ip1'], 0.50, atol=0, rtol=1e-4)
assert np.isclose(tw.lhcb2['betx', 'ip1'], 0.50, atol=0, rtol=1e-4)
assert np.isclose(tw.lhcb2['bety', 'ip1'], 0.50, atol=0, rtol=1e-4)

# Check a knob
collider.vars['on_x1'] = 30
collider.vars['on_disp'] = 0
tw = collider.twiss()
assert np.isclose(tw.lhcb1['px', 'ip1'], 30e-6, atol=5e-10, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip1'], -30e-6, atol=5e-10, rtol=0)

collider.vars.load_madx_optics_file(
    '../../../hllhc15/round/opt_round_300_1500.madx')

tw = collider.twiss()
assert np.isclose(tw.lhcb1.qx, 62.31000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1.qy, 60.32000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2.qx, 62.31000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb2.qy, 60.32000000, atol=1e-6, rtol=0)
assert np.isclose(tw.lhcb1['betx', 'ip1'], 0.30, atol=0, rtol=1e-6)
assert np.isclose(tw.lhcb1['bety', 'ip1'], 0.30, atol=0, rtol=1e-6)
assert np.isclose(tw.lhcb2['betx', 'ip1'], 0.30, atol=0, rtol=1e-6)
assert np.isclose(tw.lhcb2['bety', 'ip1'], 0.30, atol=0, rtol=1e-6)

# Check a knob
collider.vars['on_x1'] = 10
collider.vars['on_disp'] = 0
tw = collider.twiss()
assert np.isclose(tw.lhcb1['px', 'ip1'], 10e-6, atol=1e-10, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip1'], -10e-6, atol=1e-10, rtol=0)
