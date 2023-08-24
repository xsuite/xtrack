import numpy as np
import xtrack as xt

line = xt.Line.from_json('fccee_p_ring_thin.json')

tt = line.get_table()

line.build_tracker()

tw_no_rad = line.twiss(method='4d')

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad_wig_off = line.twiss(eneloss_and_damping=True)



class ActionEquilibriumEmittance(xt.Action):

    def __init__(self, line):
        self.line = line

    def run(self):
        xt.general._print.suppress = True
        self.line.compensate_radiation_energy_loss()
        xt.general._print.suppress = False
        tw_rad = self.line.twiss(eneloss_and_damping=True)

        return tw_rad

line.vars['on_wiggler_v'] = 0.1
line.compensate_radiation_energy_loss()
ey_target = 1e-12
opt = line.match(
    solve=False,
    compensate_radiation_energy_loss=True,
    eneloss_and_damping=True,
    targets=[xt.Target('eq_gemitt_y', ey_target, tol=1e-15, optimize_log=True)],
    vary=xt.Vary('on_wiggler_v', step=0.01, limits=(0.1, 2))
)

opt.solve()
