import xtrack as xt

ARC_NAMES = ['12', '23', '34', '45', '56', '67', '78', '81']

def get_arc_periodic_solution(collider, line_name=None, arc_name=None):

    assert collider.lhcb1.twiss_default.get('reverse', False) is False
    assert collider.lhcb2.twiss_default['reverse'] is True

    if arc_name is None:
        arc_name = ARC_NAMES

    if line_name is None:
        line_name = ['lhcb1', 'lhcb2']

    if line_name is None or arc_name is None:
        res = {'lhcb1': {}, 'lhcb2': {}}
        for line_name in ['lhcb1', 'lhcb2']:
            res[line_name] = {}
            for arc_name in arc_name:
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


class ActionArcPhaseAdvanceFromCell(xt.Action):

    def __init__(self, collider, line_name, arc_name):

        self.collider = collider
        self.line_name = line_name
        self.arc_name = arc_name

    def run(self):

        tw_arc = get_arc_periodic_solution(
            self.collider, line_name=self.line_name, arc_name=self.arc_name)

        return {'table': tw_arc,
                'mux': tw_arc['mux', -1] - tw_arc['mux', 0],
                'muy': tw_arc['muy', -1] - tw_arc['muy', 0]}

def match_arc_phase_advance(collider, arc_name,
                            target_mux_b1, target_muy_b1,
                            target_mux_b2, target_muy_b2,
                            solve=True):

    assert collider.lhcb1.twiss_default.get('reverse', False) is False
    assert collider.lhcb2.twiss_default['reverse'] is True

    assert arc_name in ARC_NAMES

    action_phase_b1 = ActionArcPhaseAdvanceFromCell(
                    collider=collider, line_name='lhcb1', arc_name=arc_name)
    action_phase_b2 = ActionArcPhaseAdvanceFromCell(
                        collider=collider, line_name='lhcb2', arc_name=arc_name)

    opt=collider.match(
        solve=False,
        targets=[
            action_phase_b1.target('mux', target_mux_b1),
            action_phase_b1.target('muy', target_muy_b1),
            action_phase_b2.target('mux', target_mux_b2),
            action_phase_b2.target('muy', target_muy_b2),
        ],
        vary=[
            xt.VaryList([f'kqtf.a{arc_name}b1', f'kqtd.a{arc_name}b1',
                        f'kqtf.a{arc_name}b2', f'kqtd.a{arc_name}b2',
                        f'kqf.a{arc_name}', f'kqd.a{arc_name}'
                        ]),
        ])
    if solve:
        opt.solve()

    return opt



def propagate_optics_from_beta_star(collider, ip_name, line_name,
                                    beta_star_x, beta_star_y,
                                    ele_start, ele_stop):

    assert collider.lhcb1.twiss_default.get('reverse', False) is False
    assert collider.lhcb2.twiss_default['reverse'] is True
    assert collider.lhcb1.element_names[1] == 'ip1'
    assert collider.lhcb2.element_names[1] == 'ip1.l1'
    assert collider.lhcb1.element_names[-2] == 'ip1.l1'
    assert collider.lhcb2.element_names[-2] == 'ip1'

    if ip_name == 'ip1':
        ele_stop_left = 'ip1.l1'
        ele_start_right = 'ip1'
    else:
        ele_stop_left = ip_name
        ele_start_right = ip_name

    tw_left = collider[line_name].twiss(ele_start=ele_start, ele_stop=ele_stop_left,
                    twiss_init=xt.TwissInit(line=collider[line_name],
                                            element_name=ele_stop_left,
                                            betx=beta_star_x,
                                            bety=beta_star_y))
    tw_right = collider[line_name].twiss(ele_start=ele_start_right, ele_stop=ele_stop,
                        twiss_init=xt.TwissInit(line=collider[line_name],
                                                element_name=ele_start_right,
                                                betx=beta_star_x,
                                                bety=beta_star_y))

    tw_ip = xt.TwissTable.concatenate([tw_left, tw_right])

    return tw_ip
