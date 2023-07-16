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

tw_init_custom = xt.TwissInit(betx=betx0, bety=bety0, alfx=alfx0, alfy=alfy0,
                                dx=dx0, dpx=dpx0, dy=dy0, dpy=dpy0,
                                mux=mux0, muy=muy0, element_name=location,
                                line=line)

tw = line.twiss(ele_start=location, ele_stop='ip7', twiss_init=tw_init_custom)