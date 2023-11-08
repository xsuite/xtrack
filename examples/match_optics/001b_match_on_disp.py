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

