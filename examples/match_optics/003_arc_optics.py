import numpy as np
import xtrack as xt

arc_name = '12'
line_name = 'lhcb1'

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file('../../../hllhc15/toolkit/macro.madx')

assert arc_name in ['12', '23', '34', '45', '56', '67', '78', '81']
assert line_name in ['lhcb1', 'lhcb2']

beam_number = line_name[-1:]
sector_start_number = arc_name[:1]
sector_end_number = arc_name[1:]
start_cell = f's.cell.{arc_name}.b{beam_number}'
end_cell = f'e.cell.{arc_name}.b{beam_number}'
start_arc = f'e.ds.r{sector_start_number}.b{beam_number}'
end_arc = f's.ds.l{sector_end_number}.b{beam_number}'

line = collider[line_name]

twinit_cell = line.twiss(
            ele_start=start_cell,
            ele_stop=end_cell,
            twiss_init='periodic',
            only_twiss_init=True)

tw_to_end_arc = line.twiss(
    ele_start=twinit_cell.element_name,
    ele_stop=end_arc,
    twiss_init=twinit_cell,
    )

tw_to_start_arc = line.twiss(
    ele_start=start_arc,
    ele_stop=twinit_cell.element_name,
    twiss_init=twinit_cell)

res = xt.TwissTable.concatenate([tw_to_start_arc, tw_to_end_arc])
