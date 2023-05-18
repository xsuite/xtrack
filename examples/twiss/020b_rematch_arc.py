
import numpy as np

import xtrack as xt

# xt._print.suppress = True

# Load the line
collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_collider/collider_00_from_mad.json')
collider.build_trackers()

collider.lhcb1.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['method'] = '4d'
collider.lhcb2.twiss_default['reverse'] = True

# line = collider.lhcb1
# start_cell = 's.cell.67.b1'
# end_cell = 'e.cell.67.b1'
# start_arc = 'e.ds.r6.b1'
# end_arc = 'e.ds.l7.b1'

line = collider.lhcb2
start_cell = 's.cell.67.b2'
end_cell = 'e.cell.67.b2'
start_arc = 'e.ds.r6.b2'
end_arc = 'e.ds.l7.b2'

tw = line.twiss()

mux_arc_target = tw['mux', end_arc] - tw['mux', start_arc]
muy_arc_target = tw['muy', end_arc] - tw['muy', start_arc]

tw_cell = line.twiss(
    ele_start=start_cell,
    ele_stop=end_cell,
    twiss_init='preserve')

tw_cell_periodic = line.twiss(
    method='4d',
    ele_start=start_cell,
    ele_stop=end_cell,
    twiss_init='periodic')

twinit_start_cell = tw_cell_periodic.get_twiss_init(start_cell)

tw_to_end_arc = line.twiss(
    ele_start=start_cell,
    ele_stop=end_arc,
    twiss_init=twinit_start_cell)

tw_to_start_arc = line.twiss(
    ele_start=start_arc,
    ele_stop=start_cell,
    twiss_init=twinit_start_cell)

mux_arc_from_cell = tw_to_end_arc['mux', end_arc] - tw_to_start_arc['mux', start_arc]
muy_arc_from_cell = tw_to_end_arc['muy', end_arc] - tw_to_start_arc['muy', start_arc]

assert np.isclose(mux_arc_from_cell, mux_arc_target, rtol=1e-6)
assert np.isclose(muy_arc_from_cell, muy_arc_target, rtol=1e-6)