import time

import xtrack as xt
import xpart as xp

from cpymad.madx import Madx

# xt._print.suppress = True

# Load the line
collider = xt.Multiline.from_json(
    '../../test_data/hllhc15_collider/collider_00_from_mad.json')
collider.build_trackers()
collider.lhcb1.twiss_default['method'] = '4d'

tw = collider.lhcb1.twiss()

start_cell = 's.cell.67.b1'
end_cell = 'e.cell.67.b1'
start_arc = 'e.ds.r6.b1'
end_arc = 'e.ds.l7.b1'

mux_arc_target = tw['mux', end_arc] - tw['mux', start_arc]
muy_arc_target = tw['muy', end_arc] - tw['muy', start_arc]

tw_cell = collider.lhcb1.twiss(
    ele_start=start_cell,
    ele_stop=end_cell,
    twiss_init='preserve')

tw_cell_periodic = collider.lhcb1.twiss(
    method='4d',
    ele_start=start_cell,
    ele_stop=end_cell,
    twiss_init='periodic')

twinit_start_cell = tw_cell_periodic.get_twiss_init(start_cell)

tw_to_end_arc = collider.lhcb1.twiss(
    ele_start=start_cell,
    ele_stop=end_arc,
    twiss_init=twinit_start_cell)

tw_to_start_arc = collider.lhcb1.twiss(
    ele_start=start_arc,
    ele_stop=start_cell,
    twiss_init=twinit_start_cell)

mux_arc_from_cell = tw_to_end_arc['mux', end_arc] - tw_to_start_arc['mux', start_arc]
muy_arc_from_cell = tw_to_end_arc['muy', end_arc] - tw_to_start_arc['muy', start_arc]