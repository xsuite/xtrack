import time

import xtrack as xt

import xtrack._temp.lhc_match as lm

default_tol = {None: 1e-8, 'betx': 1e-6, 'bety': 1e-6} # to have no rematching w.r.t. madx

collider = xt.Environment.from_json(
    "../../test_data/hllhc15_thick/hllhc15_collider_thick.json")
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

c0 = collider.copy()

h_correctors_ip1_b1 = ['acbh16.r8b1', 'acbh14.l1b1', 'acbh12.l1b1',
                       'acbh13.r1b1', 'acbh15.r1b1', 'acbh15.l2b1']
v_correctors_ip1_b1 = ['acbv15.r8b1', 'acbv15.l1b1', 'acbv13.l1b1',
                       'acbv12.r1b1', 'acbv14.r1b1', 'acbv16.l2b1']
h_correctors_ip5_b1 = ['acbh16.r4b1', 'acbh14.l5b1', 'acbh12.l5b1',
                       'acbh13.r5b1', 'acbh15.r5b1', 'acbh15.l6b1']
v_correctors_ip5_b1 = ['acbv15.r4b1', 'acbv15.l5b1', 'acbv13.l5b1',
                       'acbv12.r5b1', 'acbv14.r5b1', 'acbv16.l6b1']

h_correctors_ip1_b2 = ['acbh15.r8b2', 'acbh15.l1b2', 'acbh13.l1b2',
                       'acbh12.r1b2', 'acbh14.r1b2', 'acbh16.l2b2']
v_correctors_ip1_b2 = ['acbv16.r8b2', 'acbv14.l1b2', 'acbv12.l1b2',
                       'acbv13.r1b2', 'acbv15.r1b2', 'acbv15.l2b2']
h_correctors_ip5_b2 = ['acbh12.r5b2', 'acbh14.r5b2', 'acbh16.l6b2',
                       'acbh15.r4b2', 'acbh15.l5b2', 'acbh13.l5b2']
v_correctors_ip5_b2 = ['acbv16.r4b2', 'acbv14.l5b2', 'acbv12.l5b2',
                       'acbv13.r5b2', 'acbv15.r5b2', 'acbv15.l6b2']

correctors = {}
correctors['ip1_x_b1'] = h_correctors_ip1_b1
correctors['ip1_y_b1'] = v_correctors_ip1_b1
correctors['ip5_x_b1'] = h_correctors_ip5_b1
correctors['ip5_y_b1'] = v_correctors_ip5_b1
correctors['ip1_x_b2'] = h_correctors_ip1_b2
correctors['ip1_y_b2'] = v_correctors_ip1_b2
correctors['ip5_x_b2'] = h_correctors_ip5_b2
correctors['ip5_y_b2'] = v_correctors_ip5_b2

# Wipe corrector strengths
for kk in correctors:
    for cc in correctors[kk]:
        collider.vars[cc] = 0

acb_limits = (-800.e-6, 800e-6)

knobs_to_compensate = {
    'on_x1hl': dict(value=295, ip='ip1', plane='x'),
    'on_x1hs': dict(value=295, ip='ip1', plane='y'),
    'on_x1vl': dict(value=295, ip='ip1', plane='y'),
    'on_x1vs': dict(value=295, ip='ip1', plane='x'),
    'on_x5hl': dict(value=295, ip='ip5', plane='x'),
    'on_x5hs': dict(value=295, ip='ip5', plane='y'),
    'on_x5vl': dict(value=295, ip='ip5', plane='y'),
    'on_x5vs': dict(value=295, ip='ip5', plane='x'),
    'on_sep1h': dict(value=1, ip='ip1', plane='x'),
    'on_sep1v': dict(value=1, ip='ip1', plane='y'),
    'on_sep5h': dict(value=1, ip='ip5', plane='x'),
    'on_sep5v': dict(value=1, ip='ip5', plane='y'),
}

start_ele_before_cycle = []
for line_name in ['lhcb1', 'lhcb2']:
    line = collider[line_name]
    start_ele_before_cycle.append(line.element_names[0])
    line.cycle('ip3', inplace=True)

for kk in knobs_to_compensate:

    disp_knob_name = kk.replace('on_', 'on_d')

    plane = knobs_to_compensate[kk]['plane']
    ip = knobs_to_compensate[kk]['ip']
    ref_val = knobs_to_compensate[kk]['value']

    for line_name in ['lhcb1', 'lhcb2']:
        line = collider[line_name]
        beam_name = line_name[-2:]

        left_ip = {'ip1': 'ip8', 'ip5': 'ip4'}[ip]
        right_ip = {'ip1': 'ip2', 'ip5': 'ip6'}[ip]

        tw_ref = line.twiss()
        opt = line.match_knob(
            run=False,
            knob_name=disp_knob_name,
            knob_value_start=0, knob_value_end=ref_val,
            start=left_ip, end=right_ip,
            init=tw_ref, init_at=xt.START,
            vary=[
                xt.VaryList(correctors[f'{ip}_{plane}_{beam_name}'],
                            step=1e-10, limits=acb_limits),
                ],
            targets=[
                # Constraints on dispersion
                xt.Target('d' + plane, value=tw_ref, at=ip, tol=1e-6),
                xt.Target('d' + plane, value=tw_ref, at=right_ip, tol=1e-6),
                # Constraints on orbit
                xt.TargetSet([plane, 'p' + plane], value=tw_ref, at=f'e.ds.l{ip[-1:]}.{beam_name}'),
                xt.TargetSet([plane, 'p' + plane], value=tw_ref, at=f'e.ds.l{right_ip[-1:]}.{beam_name}'),
            ],
        )

        temp_expr = line.vars[kk]._expr
        line.vars[kk] = ref_val
        opt.solve()
        line.vars[kk] = temp_expr
        opt.generate_knob()

# cycle back
for line_name, estart  in zip(['lhcb1', 'lhcb2'], start_ele_before_cycle):
    line = collider[line_name]
    line.cycle(estart, inplace=True)