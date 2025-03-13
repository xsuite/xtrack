import numpy as np
import xtrack as xt

# Load a machine model
line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1, energy0=7e12)
line.cycle('ip1', inplace=True)
line.twiss_default['method'] = '4d'

# Select circuits with appropriate weights
vary=[xt.VaryList(['kqs.a23b1', 'kqs.a67b1'], step=5e-5),
      xt.VaryList(['kqs.l4b1', 'kqs.l8b1','kqs.r3b1', 'kqs.r7b1'],
                  weight=2, step=5e-5)]

# Match c_minus_re.b1
c_min_match = 1e-4
opt_re = line.match_knob(knob_name='c_minus_re.b1',
    knob_value_start=0, knob_value_end=c_min_match,
    run=False,
    vary=vary,
    targets=[
        xt.Target('c_minus_re_0', value=c_min_match, tol=1e-8),
        xt.Target('c_minus_im_0', value=0,           tol=1e-8),
    ])
opt_re.solve()
opt_re.generate_knob()

# Match c_minus_im.b1
opt_im = line.match_knob(knob_name='c_minus_im.b1',
    knob_value_start=0, knob_value_end=c_min_match,
    run=False,
    vary=vary,
    targets=[
        xt.Target('c_minus_re_0', value=0,           tol=1e-8),
        xt.Target('c_minus_im_0', value=c_min_match, tol=1e-8),
    ])
opt_im.solve()
opt_im.generate_knob()

# Test the knob
line['c_minus_re.b1'] = 1e-3
tw = line.twiss4d()
tw.c_minus_re_0 # is 0.00099998
tw.c_minus_im_0 # is 4.05e-09

line['c_minus_re.b1'] = 0
line['c_minus_im.b1'] = 1e-3
tw = line.twiss4d()
tw.c_minus_re_0 # is 2.16e-8
tw.c_minus_im_0 # is 0.001000003
