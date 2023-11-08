import time

import xtrack as xt

import xtrack._temp.lhc_match as lm

def second_order_chromaticity(line, ddelta=1e-5, **kwargs):

    tw0 = line.twiss(**kwargs)

    kwargs.pop('delta0', None)

    tw_plus = line.twiss(delta0=tw0.delta[0] + ddelta, **kwargs)
    tw_minus = line.twiss(delta0=tw0.delta[0] - ddelta, **kwargs)

    ddqx = (tw_plus.dqx - tw_minus.dqx) / (2 * ddelta)
    ddqy = (tw_plus.dqy - tw_minus.dqy) / (2 * ddelta)

    return ddqx, ddqy


default_tol = {None: 1e-8, 'betx': 1e-6, 'bety': 1e-6} # to have no rematching w.r.t. madx

collider = xt.Multiline.from_json(
    "../../test_data/hllhc15_thick/hllhc15_collider_thick.json")
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

collider.lhcb1.twiss_default['only_markers'] = True
collider.lhcb2.twiss_default['only_markers'] = True

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
v_correctors_ip1_b2 = ['acbv16.r8b2', 'acbv14.l1b2', 'acbv12.l1b2'
                       'acbv13.r1b2', 'acbv15.r1b2', 'acbv15.l2b2']
h_correctors_ip5_b2 = ['acbh12.r5b2', 'acbh14.r5b2', 'acbh16.l6b2',
                       'acbh15.r4b2', 'acbh15.l5b2', 'acbh13.l5b2']
v_correctors_ip5_b2 = ['acbv16.r4b2', 'acbv14.l5b2', 'acbv12.l5b2'
                       'acbv13.r5b2', 'acbv15.r5b2', 'acbv15.l6b2']