import xtrack as xt

ARC_NAMES = ['12', '23', '34', '45', '56', '67', '78', '81']

def get_arc_periodic_solution(collider, line_name=None, arc_name=None):

    assert collider.lhcb1.twiss_default.get('reverse', False) is False
    assert collider.lhcb2.twiss_default['reverse'] is True

    if (line_name is None or arc_name is None
        or isinstance(line_name, (list, tuple))
        or isinstance(arc_name, (list, tuple))):

        if line_name is None:
            line_name = ['lhcb1', 'lhcb2']
        if arc_name is None:
            arc_name = ARC_NAMES

        res = {'lhcb1': {}, 'lhcb2': {}}
        for ll in line_name:
            res[ll] = {}
            for aa in arc_name:
                res[ll][aa] = get_arc_periodic_solution(
                    collider, line_name=ll, arc_name=aa)
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
                            solve=True, default_tol=None):

    assert collider.lhcb1.twiss_default.get('reverse', False) is False
    assert collider.lhcb2.twiss_default['reverse'] is True

    assert arc_name in ARC_NAMES

    action_phase_b1 = ActionArcPhaseAdvanceFromCell(
                    collider=collider, line_name='lhcb1', arc_name=arc_name)
    action_phase_b2 = ActionArcPhaseAdvanceFromCell(
                        collider=collider, line_name='lhcb2', arc_name=arc_name)

    opt=collider.match(
        solve=False,
        default_tol=default_tol,
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


def compute_ats_phase_advances_for_auxiliary_irs(line_name,
        tw_sq_a81_ip1_a12, tw_sq_a45_ip5_a56,
        muxip1_l, muyip1_l, muxip1_r, muyip1_r,
        muxip5_l, muyip5_l, muxip5_r, muyip5_r,
        muxip2, muyip2, muxip4, muyip4, muxip6, muyip6, muxip8, muyip8,
        mux12, muy12, mux45, muy45, mux56, muy56, mux81, muy81
    ):

    assert line_name in ['lhcb1', 'lhcb2']
    bn = line_name[-2:]

    mux_compensate_ir2 = (tw_sq_a81_ip1_a12['mux', f's.ds.l2.{bn}'] - tw_sq_a81_ip1_a12['mux', 'ip1']
                          - muxip1_r - mux12)
    mux_ir2_target = muxip2 - mux_compensate_ir2
    muy_compensate_ir2 = (tw_sq_a81_ip1_a12['muy', f's.ds.l2.{bn}'] - tw_sq_a81_ip1_a12['muy', 'ip1']
                          - muyip1_r - muy12)
    muy_ir2_target = muyip2 - muy_compensate_ir2

    mux_compensate_ir4 = (tw_sq_a45_ip5_a56['mux', 'ip5'] - tw_sq_a45_ip5_a56['mux', f'e.ds.r4.{bn}']
                          - muxip5_l - mux45)
    mux_ir4_target = muxip4 - mux_compensate_ir4
    muy_compensate_ir4 = (tw_sq_a45_ip5_a56['muy', 'ip5'] - tw_sq_a45_ip5_a56['muy', f'e.ds.r4.{bn}']
                          - muyip5_l - muy45)
    muy_ir4_target = muyip4 - muy_compensate_ir4

    mux_compensate_ir6 = (tw_sq_a45_ip5_a56['mux', f's.ds.l6.{bn}'] - tw_sq_a45_ip5_a56['mux', 'ip5']
                          - muxip5_r - mux56)
    mux_ir6_target = muxip6 - mux_compensate_ir6
    muy_compensate_ir6 = (tw_sq_a45_ip5_a56['muy', f's.ds.l6.{bn}'] - tw_sq_a45_ip5_a56['muy', 'ip5']
                          - muyip5_r - muy56)
    muy_ir6_target = muyip6 - muy_compensate_ir6

    mux_compensate_ir8 = (tw_sq_a81_ip1_a12['mux', 'ip1.l1'] - tw_sq_a81_ip1_a12['mux', f'e.ds.r8.{bn}']
                          - muxip1_l - mux81)
    mux_ir8_target = muxip8 - mux_compensate_ir8
    muy_compensate_ir8 = (tw_sq_a81_ip1_a12['muy', 'ip1.l1'] - tw_sq_a81_ip1_a12['muy', f'e.ds.r8.{bn}']
                          - muyip1_l - muy81)
    muy_ir8_target = muyip8 - muy_compensate_ir8

    return (mux_ir2_target, muy_ir2_target, mux_ir4_target, muy_ir4_target,
            mux_ir6_target, muy_ir6_target, mux_ir8_target, muy_ir8_target)


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

class ActionPhase_23_34(xt.Action):

    def __init__(self, collider):
        self.collider = collider

    def run(self):
        try:
            tw_arc = get_arc_periodic_solution(self.collider, arc_name=['23', '34'])
        except ValueError:
            # Twiss failed
            return {
                'mux_23_34_b1': 1e100,
                'muy_23_34_b1': 1e100,
                'mux_23_34_b2': 1e100,
                'muy_23_34_b2': 1e100,
            }
        tw_23_b1 = tw_arc['lhcb1']['23']
        tw_23_b2 = tw_arc['lhcb2']['23']
        mux_23_b1 = tw_23_b1['mux', 's.ds.l3.b1'] - tw_23_b1['mux', 'e.ds.r2.b1']
        muy_23_b1 = tw_23_b1['muy', 's.ds.l3.b1'] - tw_23_b1['muy', 'e.ds.r2.b1']
        mux_23_b2 = tw_23_b2['mux', 's.ds.l3.b2'] - tw_23_b2['mux', 'e.ds.r2.b2']
        muy_23_b2 = tw_23_b2['muy', 's.ds.l3.b2'] - tw_23_b2['muy', 'e.ds.r2.b2']

        tw34_b1 = tw_arc['lhcb1']['34']
        tw34_b2 = tw_arc['lhcb2']['34']
        mux_34_b1 = tw34_b1['mux', 's.ds.l4.b1'] - tw34_b1['mux', 'e.ds.r3.b1']
        muy_34_b1 = tw34_b1['muy', 's.ds.l4.b1'] - tw34_b1['muy', 'e.ds.r3.b1']
        mux_34_b2 = tw34_b2['mux', 's.ds.l4.b2'] - tw34_b2['mux', 'e.ds.r3.b2']
        muy_34_b2 = tw34_b2['muy', 's.ds.l4.b2'] - tw34_b2['muy', 'e.ds.r3.b2']

        return {
            'mux_23_34_b1': mux_23_b1 + mux_34_b1,
            'muy_23_34_b1': muy_23_b1 + muy_34_b1,
            'mux_23_34_b2': mux_23_b2 + mux_34_b2,
            'muy_23_34_b2': muy_23_b2 + muy_34_b2
        }

class ActionPhase_67_78(xt.Action):

    def __init__(self, collider):
        self.collider = collider

    def run(self):
        try:
            tw_arc = get_arc_periodic_solution(self.collider, arc_name=['67', '78'])
        except ValueError:
            # Twiss failed
            return {
                'mux_67_78_b1': 1e100,
                'muy_67_78_b1': 1e100,
                'mux_67_78_b2': 1e100,
                'muy_67_78_b2': 1e100,
            }
        tw_67_b1 = tw_arc['lhcb1']['67']
        tw_67_b2 = tw_arc['lhcb2']['67']
        mux_67_b1 = tw_67_b1['mux', 's.ds.l7.b1'] - tw_67_b1['mux', 'e.ds.r6.b1']
        muy_67_b1 = tw_67_b1['muy', 's.ds.l7.b1'] - tw_67_b1['muy', 'e.ds.r6.b1']
        mux_67_b2 = tw_67_b2['mux', 's.ds.l7.b2'] - tw_67_b2['mux', 'e.ds.r6.b2']
        muy_67_b2 = tw_67_b2['muy', 's.ds.l7.b2'] - tw_67_b2['muy', 'e.ds.r6.b2']

        tw78_b1 = tw_arc['lhcb1']['78']
        tw78_b2 = tw_arc['lhcb2']['78']
        mux_78_b1 = tw78_b1['mux', 's.ds.l8.b1'] - tw78_b1['mux', 'e.ds.r7.b1']
        muy_78_b1 = tw78_b1['muy', 's.ds.l8.b1'] - tw78_b1['muy', 'e.ds.r7.b1']
        mux_78_b2 = tw78_b2['mux', 's.ds.l8.b2'] - tw78_b2['mux', 'e.ds.r7.b2']
        muy_78_b2 = tw78_b2['muy', 's.ds.l8.b2'] - tw78_b2['muy', 'e.ds.r7.b2']

        return {
            'mux_67_78_b1': mux_67_b1 + mux_78_b1,
            'muy_67_78_b1': muy_67_b1 + muy_78_b1,
            'mux_67_78_b2': mux_67_b2 + mux_78_b2,
            'muy_67_78_b2': muy_67_b2 + muy_78_b2
        }

def change_phase_non_ats_arcs(collider,
    d_mux_15_b1=None, d_muy_15_b1=None, d_mux_15_b2=None, d_muy_15_b2=None,
    solve=True, default_tol=None):

    assert (d_mux_15_b1 is not None or d_muy_15_b1 is not None
            or d_mux_15_b2 is not None or d_muy_15_b2 is not None), (
        "At least one of the phase advance changes must be non-zero"
            )

    action_phase_23_34 = ActionPhase_23_34(collider)
    action_phase_67_78 = ActionPhase_67_78(collider)

    phase_23_34_0 = action_phase_23_34.run()
    phase_67_78_0 = action_phase_67_78.run()

    mux_23_34_b1_target = phase_23_34_0['mux_23_34_b1']
    muy_23_34_b1_target = phase_23_34_0['muy_23_34_b1']
    mux_23_34_b2_target = phase_23_34_0['mux_23_34_b2']
    muy_23_34_b2_target = phase_23_34_0['muy_23_34_b2']
    mux_67_78_b1_target = phase_67_78_0['mux_67_78_b1']
    muy_67_78_b1_target = phase_67_78_0['muy_67_78_b1']
    mux_67_78_b2_target = phase_67_78_0['mux_67_78_b2']
    muy_67_78_b2_target = phase_67_78_0['muy_67_78_b2']

    n_contraints = 0
    targets = []
    if d_mux_15_b1 is not None:
        mux_23_34_b1_target += d_mux_15_b1
        mux_67_78_b1_target -= d_mux_15_b1
        targets.append(action_phase_23_34.target('mux_23_34_b1', mux_23_34_b1_target))
        targets.append(action_phase_67_78.target('mux_67_78_b1', mux_67_78_b1_target))
        n_contraints += 1

    if d_muy_15_b1 is not None:
        muy_23_34_b1_target += d_muy_15_b1
        muy_67_78_b1_target -= d_muy_15_b1
        targets.append(action_phase_23_34.target('muy_23_34_b1', muy_23_34_b1_target))
        targets.append(action_phase_67_78.target('muy_67_78_b1', muy_67_78_b1_target))
        n_contraints += 1

    if d_mux_15_b2 is not None:
        mux_23_34_b2_target += d_mux_15_b2
        mux_67_78_b2_target -= d_mux_15_b2
        targets.append(action_phase_23_34.target('mux_23_34_b2', mux_23_34_b2_target))
        targets.append(action_phase_67_78.target('mux_67_78_b2', mux_67_78_b2_target))
        n_contraints += 1

    if d_muy_15_b2 is not None:
        muy_23_34_b2_target += d_muy_15_b2
        muy_67_78_b2_target -= d_muy_15_b2
        targets.append(action_phase_23_34.target('muy_23_34_b2', muy_23_34_b2_target))
        targets.append(action_phase_67_78.target('muy_67_78_b2', muy_67_78_b2_target))
        n_contraints += 1

    vary = [
        xt.VaryList(['kqf.a23', 'kqd.a23', 'kqf.a34', 'kqd.a34'], weight=5),
        xt.VaryList(['kqf.a67', 'kqd.a67', 'kqf.a78', 'kqd.a78'], weight=5),
    ]
    if n_contraints > 2:
        vary += [
            xt.VaryList(['kqtf.a23b1', 'kqtd.a23b1', 'kqtf.a34b1', 'kqtd.a34b1',
                        'kqtf.a23b2', 'kqtd.a23b2', 'kqtf.a34b2', 'kqtd.a34b2']),
            xt.VaryList(['kqtf.a67b1', 'kqtd.a67b1', 'kqtf.a78b1', 'kqtd.a78b1',
                        'kqtf.a67b2', 'kqtd.a67b2', 'kqtf.a78b2', 'kqtd.a78b2']),
        ]

    opt = collider.match(
        solve=False,
        default_tol=default_tol,
        solver_options={'n_bisections': 5},
        vary=vary,
        targets=targets,
    )

    if solve:
        opt.solve()

    return opt

def rematch_ir2(collider, line_name,
                boundary_conditions_left, boundary_conditions_right,
                mux_ir2, muy_ir2, betx_ip2, bety_ip2,
                solve=True, staged_match=False, default_tol=None):

    assert line_name in ['lhcb1', 'lhcb2']
    bn = line_name[-2:]

    opt = collider[f'lhc{bn}'].match(
        solve=False,
        default_tol=default_tol,
        ele_start=f's.ds.l2.{bn}', ele_stop=f'e.ds.r2.{bn}',
        # Left boundary
        twiss_init='preserve_start', table_for_twiss_init=boundary_conditions_left,
        targets=[
            # IP optics
            xt.TargetList(('alfx', 'alfy', 'dx', 'dpx'), value=0, at='ip2',    tag='stage2'),
            xt.Target('betx', value=betx_ip2, at='ip2', tag='stage2'),
            xt.Target('bety', value=bety_ip2, at='ip2', tag='stage2'),
            # Right boundary
            xt.TargetList(('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'), tag='stage0',
                    value=boundary_conditions_right, at=f'e.ds.r2.{bn}'),
            xt.TargetRelPhaseAdvance('mux', mux_ir2, tag='stage0'),
            xt.TargetRelPhaseAdvance('muy', muy_ir2, tag='stage0'),
        ],
        vary=[
            xt.VaryList([
                f'kq9.l2{bn}', f'kq10.l2{bn}', f'kqtl11.l2{bn}', f'kqt12.l2{bn}', f'kqt13.l2{bn}',
                f'kq9.r2{bn}', f'kq10.r2{bn}', f'kqtl11.r2{bn}', f'kqt12.r2{bn}', f'kqt13.r2{bn}'],
                tag='stage0'),
            xt.VaryList(
                [f'kq4.l2{bn}', f'kq5.l2{bn}',  f'kq6.l2{bn}',    f'kq7.l2{bn}',   f'kq8.l2{bn}',
                 f'kq6.r2{bn}',    f'kq7.r2{bn}',   f'kq8.r2{bn}'],
                tag='stage1'),
            xt.VaryList([f'kq4.r2{bn}', f'kq5.r2{bn}'], tag='stage2')
        ]
    )

    if solve:
        if staged_match:
            opt.disable_all_vary()
            opt.disable_all_targets()

            opt.enable_vary(tag='stage0')
            opt.enable_targets(tag='stage0')
            opt.solve()

            opt.enable_vary(tag='stage1')
            opt.solve()

            opt.enable_vary(tag='stage2')
            opt.enable_targets(tag='stage2')
            opt.solve()
        else:
            opt.solve()

        return opt

def rematch_ir3(collider, line_name,
                boundary_conditions_left, boundary_conditions_right,
                mux_ir3, muy_ir3,
                alfx_ip3, alfy_ip3,
                betx_ip3, bety_ip3,
                dx_ip3, dpx_ip3,
                solve=True, staged_match=False, default_tol=None):

    assert line_name in ['lhcb1', 'lhcb2']
    bn = line_name[-2:]

    opt = collider[f'lhc{bn}'].match(
        solve=False,
        default_tol=default_tol,
        ele_start=f's.ds.l3.{bn}', ele_stop=f'e.ds.r3.{bn}',
        # Left boundary
        twiss_init='preserve_start', table_for_twiss_init=boundary_conditions_left,
        targets=[
            xt.Target('alfx', alfx_ip3, at='ip3', tag='stage1'),
            xt.Target('alfy', alfy_ip3, at='ip3', tag='stage1'),
            xt.Target('betx', betx_ip3, at='ip3', tag='stage1'),
            xt.Target('bety', bety_ip3, at='ip3', tag='stage1'),
            xt.Target('dx', dx_ip3, at='ip3', tag='stage1'),
            xt.Target('dpx', dpx_ip3, at='ip3', tag='stage1'),
            xt.TargetList(('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'),
                    value=boundary_conditions_right, at=f'e.ds.r3.{bn}'),
            xt.TargetRelPhaseAdvance('mux', mux_ir3),
            xt.TargetRelPhaseAdvance('muy', muy_ir3),
        ],
        vary=[
            xt.VaryList([f'kqt13.l3{bn}', f'kqt12.l3{bn}', f'kqtl11.l3{bn}',
                         f'kqtl10.l3{bn}', f'kqtl9.l3{bn}', f'kqtl8.l3{bn}',
                         f'kqtl7.l3{bn}', f'kq6.l3{bn}',
                         f'kq6.r3{bn}', f'kqtl7.r3{bn}',
                         f'kqtl8.r3{bn}', f'kqtl9.r3{bn}', f'kqtl10.r3{bn}',
                         f'kqtl11.r3{bn}', f'kqt12.r3{bn}', f'kqt13.r3{bn}'])
        ]
    )

    if solve:
        if staged_match:
            opt.disable_targets(tag='stage1')
            opt.solve()
            opt.enable_targets(tag='stage1')
            opt.solve()
        else:
            opt.solve()

    return opt

def rematch_ir4(collider, line_name,
                boundary_conditions_left, boundary_conditions_right,
                mux_ir4, muy_ir4,
                alfx_ip4, alfy_ip4,
                betx_ip4, bety_ip4,
                dx_ip4, dpx_ip4,
                solve=True, staged_match=False, default_tol=None):

    assert line_name in ['lhcb1', 'lhcb2']
    bn = line_name[-2:]

    opt = collider[f'lhc{bn}'].match(
        solve=False,
        default_tol=default_tol,
        ele_start=f's.ds.l4.{bn}', ele_stop=f'e.ds.r4.{bn}',
        # Left boundary
        twiss_init='preserve_start', table_for_twiss_init=boundary_conditions_left,
        targets=[
            xt.Target('alfx', alfx_ip4, at='ip4', tag='stage1'),
            xt.Target('alfy', alfy_ip4, at='ip4', tag='stage1'),
            xt.Target('betx', betx_ip4, at='ip4', tag='stage1'),
            xt.Target('bety', bety_ip4, at='ip4', tag='stage1'),
            xt.Target('dx', dx_ip4, at='ip4', tag='stage1'),
            xt.Target('dpx', dpx_ip4, at='ip4', tag='stage1'),
            xt.TargetList(('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'),
                    value=boundary_conditions_right, at=f'e.ds.r4.{bn}'),
            xt.TargetRelPhaseAdvance('mux', mux_ir4),
            xt.TargetRelPhaseAdvance('muy', muy_ir4),
        ],
        vary=[
            xt.VaryList([
                f'kqt13.l4{bn}', f'kqt12.l4{bn}', f'kqtl11.l4{bn}', f'kq10.l4{bn}',
                f'kq9.l4{bn}', f'kq8.l4{bn}', f'kq7.l4{bn}', f'kq6.l4{bn}',
                f'kq5.l4{bn}', f'kq5.r4{bn}', f'kq6.r4{bn}', f'kq7.r4{bn}',
                f'kq8.r4{bn}', f'kq9.r4{bn}', f'kq10.r4{bn}', f'kqtl11.r4{bn}',
                f'kqt12.r4{bn}', f'kqt13.r4{bn}'
            ])
        ]
    )

    if solve:
        if staged_match:
            opt.disable_targets(tag='stage1')
            opt.solve()
            opt.enable_targets(tag='stage1')
            opt.solve()
        else:
            opt.solve()

    return opt


def rematch_ir6(collider, line_name,
                boundary_conditions_left, boundary_conditions_right,
                mux_ir6, muy_ir6,
                alfx_ip6, alfy_ip6,
                betx_ip6, bety_ip6,
                dx_ip6, dpx_ip6,
                solve=True, staged_match=False, default_tol=None):

    assert line_name in ['lhcb1', 'lhcb2']
    bn = line_name[-2:]

    opt = collider[f'lhc{bn}'].match(
        solve=False,
        default_tol=default_tol,
        ele_start=f's.ds.l6.{bn}', ele_stop=f'e.ds.r6.{bn}',
        # Left boundary
        twiss_init='preserve_start', table_for_twiss_init=boundary_conditions_left,
        targets=[
            xt.Target('alfx', alfx_ip6, at='ip6'),
            xt.Target('alfy', alfy_ip6, at='ip6'),
            xt.Target('betx', betx_ip6, at='ip6'),
            xt.Target('bety', bety_ip6, at='ip6'),
            xt.Target('dx', dx_ip6, at='ip6'),
            xt.Target('dpx', dpx_ip6, at='ip6'),
            xt.TargetList(('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'),
                    value=boundary_conditions_right, at=f'e.ds.r6.{bn}'),
            xt.TargetRelPhaseAdvance('mux', mux_ir6),
            xt.TargetRelPhaseAdvance('muy', muy_ir6),
        ],
        vary=(
            [xt.VaryList([
                f'kqt13.l6{bn}', f'kqt12.l6{bn}', f'kqtl11.l6{bn}', f'kq10.l6{bn}',
                f'kq9.l6{bn}', f'kq8.l6{bn}', f'kq5.l6{bn}',
                f'kq5.r6{bn}', f'kq8.r6{bn}', f'kq9.r6{bn}',
                f'kq10.r6{bn}', f'kqtl11.r6{bn}', f'kqt12.r6{bn}', f'kqt13.r6{bn}'])]
            + [xt.Vary((f'kq4.r6{bn}' if bn == 'b1' else f'kq4.l6{bn}'))]
        )
    )

    # No staged match for IR6
    if solve:
        opt.solve()

    return opt

def rematch_ir7(collider, line_name,
              boundary_conditions_left, boundary_conditions_right,
              mux_ir7, muy_ir7,
              alfx_ip7, alfy_ip7,
              betx_ip7, bety_ip7,
              dx_ip7, dpx_ip7,
              solve=True, staged_match=False, default_tol=None):

    assert line_name in ['lhcb1', 'lhcb2']
    bn = line_name[-2:]

    opt = collider[f'lhc{bn}'].match(
        solve=False,
        default_tol=default_tol,
        ele_start=f's.ds.l7.{bn}', ele_stop=f'e.ds.r7.{bn}',
        # Left boundary
        twiss_init='preserve_start', table_for_twiss_init=boundary_conditions_left,
        targets=[
            xt.Target('alfx', alfx_ip7, at='ip7', tag='stage1'),
            xt.Target('alfy', alfy_ip7, at='ip7', tag='stage1'),
            xt.Target('betx', betx_ip7, at='ip7', tag='stage1'),
            xt.Target('bety', bety_ip7, at='ip7', tag='stage1'),
            xt.Target('dx', dx_ip7, at='ip7', tag='stage1'),
            xt.Target('dpx', dpx_ip7, at='ip7', tag='stage1'),
            xt.TargetList(('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'),
                    value=boundary_conditions_right, at=f'e.ds.r7.{bn}'),
            xt.TargetRelPhaseAdvance('mux', mux_ir7),
            xt.TargetRelPhaseAdvance('muy', muy_ir7),
        ],
        vary=[
            xt.VaryList([
                f'kqt13.l7{bn}', f'kqt12.l7{bn}', f'kqtl11.l7{bn}',
                f'kqtl10.l7{bn}', f'kqtl9.l7{bn}', f'kqtl8.l7{bn}',
                f'kqtl7.l7{bn}', f'kq6.l7{bn}',
                f'kq6.r7{bn}', f'kqtl7.r7{bn}',
                f'kqtl8.r7{bn}', f'kqtl9.r7{bn}', f'kqtl10.r7{bn}',
                f'kqtl11.r7{bn}', f'kqt12.r7{bn}', f'kqt13.r7{bn}'
            ])
        ]
    )

    if solve:
        if staged_match:
            opt.disable_targets(tag='stage1')
            opt.solve()
            opt.enable_targets(tag='stage1')
            opt.solve()
        else:
            opt.solve()

    return opt

def rematch_ir8(collider, line_name,
                boundary_conditions_left, boundary_conditions_right,
                mux_ir8, muy_ir8,
                alfx_ip8, alfy_ip8,
                betx_ip8, bety_ip8,
                dx_ip8, dpx_ip8,
                solve=True, staged_match=False, default_tol=None):

    assert line_name in ['lhcb1', 'lhcb2']
    bn = line_name[-2:]

    opt = collider[f'lhc{bn}'].match(
        solve=False,
        default_tol=default_tol,
        ele_start=f's.ds.l8.{bn}', ele_stop=f'e.ds.r8.{bn}',
        # Left boundary
        twiss_init='preserve_start', table_for_twiss_init=boundary_conditions_left,
        targets=[
            xt.Target('alfx', alfx_ip8, at='ip8'),
            xt.Target('alfy', alfy_ip8, at='ip8'),
            xt.Target('betx', betx_ip8, at='ip8'),
            xt.Target('bety', bety_ip8, at='ip8'),
            xt.Target('dx', dx_ip8, at='ip8'),
            xt.Target('dpx', dpx_ip8, at='ip8'),
            xt.TargetList(('betx', 'bety', 'alfx', 'alfy', 'dx', 'dpx'),
                    value=boundary_conditions_right, at=f'e.ds.r8.{bn}',
                    tag='stage2'),
            xt.TargetRelPhaseAdvance('mux', mux_ir8),
            xt.TargetRelPhaseAdvance('muy', muy_ir8),
        ],
        vary=[
            xt.VaryList([
                f'kq6.l8{bn}', f'kq7.l8{bn}',
                f'kq8.l8{bn}', f'kq9.l8{bn}', f'kq10.l8{bn}', f'kqtl11.l8{bn}',
                f'kqt12.l8{bn}', f'kqt13.l8{bn}']
            ),
            xt.VaryList([f'kq4.l8{bn}', f'kq5.l8{bn}'], tag='stage1'),
            xt.VaryList([
                f'kq4.r8{bn}', f'kq5.r8{bn}', f'kq6.r8{bn}', f'kq7.r8{bn}',
                f'kq8.r8{bn}', f'kq9.r8{bn}', f'kq10.r8{bn}', f'kqtl11.r8{bn}',
                f'kqt12.r8{bn}', f'kqt13.r8{bn}'],
                tag='stage2')
        ]
    )

    if solve:
        if staged_match:
            opt.disable_targets(tag=['stage1', 'stage2'])
            opt.disable_vary(tag=['stage1', 'stage2'])
            opt.solve()

            opt.enable_vary(tag='stage1')
            opt.solve()

            opt.enable_targets(tag='stage2')
            opt.enable_vary(tag='stage2')
            opt.solve()
        else:
            opt.solve()

    return opt