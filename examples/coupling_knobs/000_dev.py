import numpy as np
import xtrack as xt

# Load a line and build tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.twiss_default['method'] = '4d'



line.vv['kqs.r3b1'] = 0.0001

tt = line.get_table(attr=True)
tw = line.twiss()
k1sl = tt['k1sl']

c_min = 1 / (2*np.pi) * np.sum(k1sl * np.sqrt(tw.betx * tw.bety)
                               * np.exp(1j * 2 * np.pi * (tw.mux - tw.muy)))