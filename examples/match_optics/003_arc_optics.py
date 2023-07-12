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

tables_to_concat = [tw_to_start_arc, tw_to_end_arc]

ind_per_table = []
for ii, tt in enumerate(tables_to_concat):

    this_ind = [0, len(tt)]
    if ii > 0:
        if tt.name[0] in tables_to_concat[ii-1].name:
            assert tt.name[0] == tables_to_concat[ii-1].name[ind_per_table[ii-1][1]-1]
            ind_per_table[ii-1][1] -= 1
    if ii < len(tables_to_concat) - 1:
        if tt.name[-1] == '_end_point':
            this_ind[1] -= 1

    ind_per_table.append(this_ind)

n_elem = sum([ind[1] - ind[0] for ind in ind_per_table])

new_data = {}
for kk in tables_to_concat[0]._col_names:
    if kk == 'W_matrix':
        new_data[kk] = np.empty(
            (n_elem, 6, 6), dtype=tables_to_concat[0][kk].dtype)
        continue
    new_data[kk] = np.empty(n_elem, dtype=tables_to_concat[0][kk].dtype)

i_start = 0
for ii, tt in enumerate(tables_to_concat):
    i_end = i_start + ind_per_table[ii][1] - ind_per_table[ii][0]
    for kk in tt._col_names:
        if kk == 'W_matrix':
            new_data[kk][i_start:i_end] = tt[kk][ind_per_table[ii][0]:ind_per_table[ii][1], :, :]
            continue
        new_data[kk][i_start:i_end] = tt[kk][ind_per_table[ii][0]:ind_per_table[ii][1]]
        if kk in ['mux', 'muy', 'dzeta', 's']:
            new_data[kk][i_start:i_end] -= new_data[kk][i_start]
            new_data[kk][i_start:i_end] += new_data[kk][i_start-1]

    i_start = i_end

new_table = xt.twiss.TwissTable(new_data)
