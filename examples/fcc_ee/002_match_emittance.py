import numpy as np
import xtrack as xt

line = xt.Line.from_json('fccee_p_ring_thin.json')

tt = line.get_table()

line.build_tracker()

tw_no_rad = line.twiss(method='4d')

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad_wig_off = line.twiss(eneloss_and_damping=True)

line.vars['on_wiggler_v'] = 0.7
line.compensate_radiation_energy_loss()

class ActionEquilibriumEmittance(xt.Action):

    def __init__(self, line):
        self.line = line

    def run(self):
        xt.general._print.suppress = True
        self.line.compensate_radiation_energy_loss()
        xt.general._print.suppress = False
        tw_rad = self.line.twiss(eneloss_and_damping=True)

        ex = tw_rad.nemitt_x_rad / (tw_rad.gamma0 * tw_rad.beta0)
        ey = tw_rad.nemitt_y_rad / (tw_rad.gamma0 * tw_rad.beta0)
        ez = tw_rad.nemitt_zeta_rad / (tw_rad.gamma0 * tw_rad.beta0)

        return {'ex': ex, 'ey': ey, 'ez': ez,
                'log10_ex': np.log10(ex), 'log10_ey': np.log10(ey),
                'log10_ez': np.log10(ez)}

ey_target = 1e-12
action_equilibrium_emittance = ActionEquilibriumEmittance(line)
opt = line.match(
    solve=False,
    targets=[action_equilibrium_emittance.target('log10_ey', np.log10(ey_target),
                                                 tol=1e-3)],
    vary=xt.Vary('on_wiggler_v', step=0.01, limits=(0.1, 2))
)

opt.solve()
