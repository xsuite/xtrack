import xtrack as xt

ARC_NAMES = ['12', '23', '34', '45', '56', '67', '78', '81']

def get_arc_periodic_solution(collider, line_name=None, arc_name=None):

    assert collider.lhcb1.twiss_default.get('reverse', False) is False
    assert collider.lhcb2.twiss_default['reverse'] is True

    if line_name is None or arc_name is None:
        assert line_name is None and arc_name is None
        res = {'lhcb1': {}, 'lhcb2': {}}
        for line_name in ['lhcb1', 'lhcb2']:
            res[line_name] = {}
            for arc_name in ARC_NAMES:
                res[line_name][arc_name] = get_arc_periodic_solution(
                    collider, line_name=line_name, arc_name=arc_name)
        return res


    assert arc_name in ARC_NAMES
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
    res['mux'] = res['mux'] - res['mux', start_arc]
    res['muy'] = res['muy'] - res['muy', start_arc]

    return res

