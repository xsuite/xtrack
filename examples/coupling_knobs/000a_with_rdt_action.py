import numpy as np
import xtrack as xt

# Load a line and build tracker
line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.cycle('ip1', inplace=True)
line.twiss_default['method'] = '4d'

# see Eq. 47 in https://cds.cern.ch/record/522049/files/lhc-project-report-501.pdf
class ActionCmin(xt.Action):
    def __init__(self, line):
        self.line = line
    def run(self):
        tw = self.line.twiss()
        tt = self.line.get_table(attr=True)
        k1sl = tt['k1sl']
        c_min = 1 / (2*np.pi) * np.sum(k1sl * np.sqrt(tw.betx * tw.bety)
                                * np.exp(1j * 2 * np.pi * (tw.mux - tw.muy)))
        return {'c_min_re': c_min.real, 'c_min_im': c_min.imag}

act_cmin = ActionCmin(line)

# Circuits in non-ATS arcs
vary=[xt.VaryList(['kqs.a23b1', 'kqs.a67b1'], step=5e-5),
      xt.VaryList(['kqs.l4b1', 'kqs.l8b1','kqs.r3b1', 'kqs.r7b1'], weight=2,
                  step=5e-5)]

c_min_match = 1e-4

opt_re = line.match_knob(knob_name='c_minus_re.b1',
    knob_value_start=0, knob_value_end=c_min_match,
    run=False,
    vary=vary,
    targets=[
        act_cmin.target('c_min_re', value=c_min_match, tol=1e-8),
        act_cmin.target('c_min_im', value=0, tol=1e-8),
    ])
opt_re.solve()
opt_re.generate_knob()

opt_im = line.match_knob(knob_name='c_minus_im.b1',
    knob_value_start=0, knob_value_end=c_min_match,
    run=False,
    vary=vary,
    targets=[
        act_cmin.target('c_min_re', value=0, tol=1e-8),
        act_cmin.target('c_min_im', value=c_min_match, tol=1e-8),
    ])
opt_im.solve()
opt_im.generate_knob()

line.vars['c_minus_re.b1'] = 5e-3
line.vars['c_minus_im.b1'] = 0
tt_re = line.get_table(attr=True)

line.vars['c_minus_re.b1'] = 0
line.vars['c_minus_im.b1'] = 5e-3
tt_im = line.get_table(attr=True)

# Check orthogonality
line.vars['c_minus_re.b1'] = 1e-3
line.vars['c_minus_im.b1'] = 1e-3
assert np.isclose(line.twiss().c_minus/np.sqrt(2), 1e-3, rtol=0, atol=1.5e-5)


import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)

plt.figure(2)
ax21 = plt.subplot(1, 1, 1)
plt.plot(tt_re.s, tt_re['k1sl'], '.', label='c_minus_re.b1=1e-3')
plt.plot(tt_im.s, tt_im['k1sl'], '.', label='c_minus_im.b1=1e-3')
plt.ylabel('k1s * L')
plt.xlabel('s [m]')
plt.suptitle('Xsuite')
plt.legend()

plt.show()



