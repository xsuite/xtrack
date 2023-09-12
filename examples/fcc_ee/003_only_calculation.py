import numpy as np
import xtrack as xt

fname = 'fccee_t'

line = xt.Line.from_json(fname + '_thin.json')

line.build_tracker()

tw_no_rad = line.twiss(method='4d')

line.configure_radiation(model='mean')
line.compensate_radiation_energy_loss()

tw_rad_wig_off = line.twiss(eneloss_and_damping=True)

line.vars['on_wiggler_v'] = 0.5
line.compensate_radiation_energy_loss()

tw_rad = line.twiss(eneloss_and_damping=True)


