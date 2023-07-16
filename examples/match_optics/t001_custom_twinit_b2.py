import xtrack as xt
import numpy as np

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()

# Switch on crossing angles to get some vertical dispersion
collider.vars['on_x1'] = 500
collider.vars['on_x5'] = 500
collider.vars['on_disp'] = 0

line = collider.lhcb2
assert line.twiss_default['reverse'] is True

location = 'mb.a28l5.b2_entry'

tw_full = line.twiss()
betx0 = tw_full['betx', location]
bety0 = tw_full['bety', location]
alfx0 = tw_full['alfx', location]
alfy0 = tw_full['alfy', location]
dx0 = tw_full['dx', location]
dpx0 = tw_full['dpx', location]
dy0 = tw_full['dy', location]
dpy0 = tw_full['dpy', location]
mux0 = tw_full['mux', location]
muy0 = tw_full['muy', location]
x0 = tw_full['x', location]
px0 = tw_full['px', location]
y0 = tw_full['y', location]
py0 = tw_full['py', location]

tw_init_custom = xt.TwissInit(
                        betx=betx0, bety=bety0, alfx=alfx0, alfy=alfy0,
                        dx=dx0, dpx=dpx0, dy=dy0, dpy=dpy0,
                        mux=mux0, muy=muy0, x=x0, px=px0, y=y0, py=py0,
                        element_name=location, line=line)

tw = line.twiss(ele_start=location, ele_stop='ip7', twiss_init=tw_init_custom)

# Check at starting point
assert np.isclose(tw['betx', location], betx0, atol=1e-9, rtol=0)
assert np.isclose(tw['bety', location], bety0, atol=1e-9, rtol=0)
assert np.isclose(tw['alfx', location], alfx0, atol=1e-9, rtol=0)
assert np.isclose(tw['alfy', location], alfy0, atol=1e-9, rtol=0)
assert np.isclose(tw['dx', location], dx0, atol=1e-9, rtol=0)
assert np.isclose(tw['dpx', location], dpx0, atol=1e-9, rtol=0)
assert np.isclose(tw['dy', location], dy0, atol=1e-9, rtol=0)
assert np.isclose(tw['dpy', location], dpy0, atol=1e-9, rtol=0)
assert np.isclose(tw['mux', location], mux0, atol=1e-9, rtol=0)
assert np.isclose(tw['muy', location], muy0, atol=1e-9, rtol=0)

# Check at a point in a downstream arc
loc_check = 'mb.a24l7.b2'
assert np.isclose(tw['betx', loc_check], tw_full['betx', loc_check],
                    atol=1e-8, rtol=0)
assert np.isclose(tw['bety', loc_check], tw_full['bety', loc_check],
                    atol=1e-8, rtol=0)
assert np.isclose(tw['alfx', loc_check], tw_full['alfx', loc_check],
                    atol=1e-8, rtol=0)
assert np.isclose(tw['alfy', loc_check], tw_full['alfy', loc_check],
                    atol=1e-8, rtol=0)
assert np.isclose(tw['dx', loc_check], tw_full['dx', loc_check],
                    atol=1e-8, rtol=0)
assert np.isclose(tw['dpx', loc_check], tw_full['dpx', loc_check],
                    atol=1e-8, rtol=0)
assert np.isclose(tw['dy', loc_check], tw_full['dy', loc_check],
                    atol=1e-8, rtol=0)
assert np.isclose(tw['dpy', loc_check], tw_full['dpy', loc_check],
                    atol=1e-8, rtol=0)
assert np.isclose(tw['mux', loc_check], tw_full['mux', loc_check],
                    atol=1e-9, rtol=0)
assert np.isclose(tw['muy', loc_check], tw_full['muy', loc_check],
                    atol=1e-9, rtol=0)

