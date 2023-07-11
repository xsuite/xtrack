import xtrack as xt
import numpy as np

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()

collider.vars.load_madx_optics_file(
    '../../../hllhc15/ramp/opt_endoframp_500_1500.madx')

collider.vars['on_x1'] = 30
tw = collider.twiss()
assert np.isclose(tw.lhcb1['px', 'ip1'], 30e-6, atol=1e-8, rtol=0)
assert np.isclose(tw.lhcb2['px', 'ip1'], -30e-6, atol=1e-8, rtol=0)