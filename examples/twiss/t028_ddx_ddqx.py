import numpy as np
import xtrack as xt


line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.vars['on_x5'] = 300

tw = line.twiss(method='4d')
nlchr = line.get_non_linear_chromaticity(delta0_range=(-1e-4, 1e-4),
                                        num_delta=21, fit_order=1, method='4d')
tw_fw = line.twiss(start='ip4', end='ip6', init_at='ip4',
              x=tw['x', 'ip4'], px=tw['px', 'ip4'],
              y=tw['y', 'ip4'], py=tw['py', 'ip4'],
              betx=tw['betx', 'ip4'], bety=tw['bety', 'ip4'],
              alfx=tw['alfx', 'ip4'], alfy=tw['alfy', 'ip4'],
              dx=tw['dx', 'ip4'], dpx=tw['dpx', 'ip4'],
              dy=tw['dy', 'ip4'], dpy=tw['dpy', 'ip4'],
              ddx=tw['ddx', 'ip4'], ddy=tw['ddy', 'ip4'],
              ddpx=tw['ddpx', 'ip4'], ddpy=tw['ddpy', 'ip4'],
              compute_chromatic_properties=True)

tw_bw = line.twiss(start='ip4', end='ip6', init_at='ip6',
              x=tw['x', 'ip6'], px=tw['px', 'ip6'],
              y=tw['y', 'ip6'], py=tw['py', 'ip6'],
              betx=tw['betx', 'ip6'], bety=tw['bety', 'ip6'],
              alfx=tw['alfx', 'ip6'], alfy=tw['alfy', 'ip6'],
              dx=tw['dx', 'ip6'], dpx=tw['dpx', 'ip6'],
              dy=tw['dy', 'ip6'], dpy=tw['dpy', 'ip6'],
              ddx=tw['ddx', 'ip6'], ddy=tw['ddy', 'ip6'],
              ddpx=tw['ddpx', 'ip6'], ddpy=tw['ddpy', 'ip6'],
              compute_chromatic_properties=True)



nlchr = line.get_non_linear_chromaticity(delta0_range=(-1e-4, 1e-4),
                                        num_delta=21, fit_order=2, method='4d')

location = 'ip3'

x_xs = np.array([tt['x', location] for tt in nlchr.twiss])
px_xs = np.array([tt['px', location] for tt in nlchr.twiss])
y_xs = np.array([tt['y', location] for tt in nlchr.twiss])
py_xs = np.array([tt['py', location] for tt in nlchr.twiss])
qx_xs = np.array([tt['qx'] for tt in nlchr.twiss])
qy_xs = np.array([tt['qy'] for tt in nlchr.twiss])
delta = np.array([tt['delta', location] for tt in nlchr.twiss])

pxs_x = np.polyfit(delta, x_xs, 3)
pxs_px = np.polyfit(delta, px_xs, 3)
pxs_y = np.polyfit(delta, y_xs, 3)
pxs_py = np.polyfit(delta, py_xs, 3)
pxs_qx = np.polyfit(delta, qx_xs, 3)
pxs_qy = np.polyfit(delta, qy_xs, 3)

assert np.allclose(delta, nlchr.delta0, atol=1e-6, rtol=0)
assert np.allclose(tw['dx', location], pxs_x[-2], atol=0, rtol=1e-4)
assert np.allclose(tw['dpx', location], pxs_px[-2], atol=0, rtol=1e-4)
assert np.allclose(tw['dy', location], pxs_y[-2], atol=0, rtol=1e-4)
assert np.allclose(tw['dpy', location], pxs_py[-2], atol=0, rtol=1e-4)
assert np.allclose(tw['ddx', location], 2*pxs_x[-3], atol=0, rtol=1e-4)
assert np.allclose(tw['ddpx', location], 2*pxs_px[-3], atol=0, rtol=1e-4)
assert np.allclose(tw['ddy', location], 2*pxs_y[-3], atol=0, rtol=1e-4)
assert np.allclose(tw['ddpy', location], 2*pxs_py[-3], atol=0, rtol=1e-4)
assert np.isclose(tw['dqx'], pxs_qx[-2], atol=0, rtol=1e-3)
assert np.isclose(tw['ddqx'], pxs_qx[-3]*2, atol=0, rtol=1e-4)
assert np.isclose(tw['dqy'], pxs_qy[-2], atol=0, rtol=1e-3)
assert np.isclose(tw['ddqy'], pxs_qy[-3]*2, atol=0, rtol=1e-4)

assert np.isclose(nlchr['dqx'], pxs_qx[-2], atol=0, rtol=2e-3)
assert np.isclose(nlchr['dqy'], pxs_qy[-2], atol=0, rtol=2e-3)
assert np.isclose(nlchr['ddqx'], pxs_qx[-3]*2, atol=0, rtol=1e-4)
assert np.isclose(nlchr['ddqy'], pxs_qy[-3]*2, atol=0, rtol=1e-4)

tw_part = tw.rows['ip4':'ip6']
assert np.allclose(tw_part['ddx'], tw_fw.rows[:-1]['ddx'], atol=1e-2, rtol=0)
assert np.allclose(tw_part['ddy'], tw_fw.rows[:-1]['ddy'], atol=1e-2, rtol=0)
assert np.allclose(tw_part['ddpx'], tw_fw.rows[:-1]['ddpx'], atol=1e-3, rtol=0)
assert np.allclose(tw_part['ddpy'], tw_fw.rows[:-1]['ddpy'], atol=1e-3, rtol=0)
assert np.allclose(tw_part['dx'], tw_bw.rows[:-1]['dx'], atol=1e-2, rtol=0)
assert np.allclose(tw_part['dy'], tw_bw.rows[:-1]['dy'], atol=1e-2, rtol=0)
assert np.allclose(tw_part['dpx'], tw_bw.rows[:-1]['dpx'], atol=1e-3, rtol=0)
assert np.allclose(tw_part['dpy'], tw_bw.rows[:-1]['dpy'], atol=1e-3, rtol=0)
