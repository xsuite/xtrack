import numpy as np

from cpymad.madx import Madx
import xtrack as xt
import xdeps as xd

mad = Madx()
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use(sequence="lhcb1")

line = xt.Line.from_madx_sequence(mad.sequence['lhcb1'], deferred_expressions=True)
line.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, q0=1,
                                 gamma0=mad.sequence.lhcb1.beam.gamma)
line.build_tracker()

line.functions['ramp_on_x1'] = xd.FunctionPieceWiseLinear(x=[0.5, 1, 3, 5],
                                                          y=[0, 100, 100, 0])
line.vars['t_turn_s'] = 0
line.vars['on_x1'] = line.functions.ramp_on_x1(line.vars['t_turn_s'])

assert np.isclose(line.twiss()['px', 'ip1'], 0, atol=1e-7, rtol = 0)
line.vars['t_turn_s'] = 1
assert np.isclose(line.twiss()['px', 'ip1'], 100e-6, atol=1e-7, rtol = 0)
line.vars['t_turn_s'] = 2
assert np.isclose(line.twiss()['px', 'ip1'], 100e-6, atol=1e-7, rtol = 0)
line.vars['t_turn_s'] = 3
assert np.isclose(line.twiss()['px', 'ip1'], 100e-6, atol=1e-7, rtol = 0)
line.vars['t_turn_s'] = 4
assert np.isclose(line.twiss()['px', 'ip1'], 50e-6, atol=1e-7, rtol = 0)
line.vars['t_turn_s'] = 5
assert np.isclose(line.twiss()['px', 'ip1'], 0, atol=1e-7, rtol = 0)
line.vars['t_turn_s'] = 6
assert np.isclose(line.twiss()['px', 'ip1'], 0, atol=1e-7, rtol = 0)

# Test a knob involving functions
line.vars['on_x8'] = 100
line.vars['phi_ir8'] = 0
assert np.isclose(line.twiss()['px', 'ip8'], 100e-6, atol=1e-7, rtol = 0)
assert np.isclose(line.twiss()['py', 'ip8'], 0, atol=1e-7, rtol = 0)

line.vars['phi_ir8'] = 90
assert np.isclose(line.twiss()['px', 'ip8'], 0, atol=1e-7, rtol = 0)
assert np.isclose(line.twiss()['py', 'ip8'], 100e-6, atol=1e-7, rtol = 0)

line.vars['phi_ir8'] = 0

# Test json roundtrip
line.to_json('test.json')
line2 = xt.Line.from_json('test.json')
line2.build_tracker()

assert np.isclose(line2.twiss()['px', 'ip8'], 100e-6, atol=1e-7, rtol = 0)
assert np.isclose(line2.twiss()['py', 'ip8'], 0, atol=1e-7, rtol = 0)
line2.vars['phi_ir8'] = 90
assert np.isclose(line2.twiss()['px', 'ip8'], 0, atol=1e-7, rtol = 0)
assert np.isclose(line2.twiss()['py', 'ip8'], 100e-6, atol=1e-7, rtol = 0)

assert np.isclose(line2.twiss()['px', 'ip1'], 0, atol=1e-7, rtol = 0)
line2.vars['t_turn_s'] = 1
assert np.isclose(line2.twiss()['px', 'ip1'], 100e-6, atol=1e-7, rtol = 0)
line2.vars['t_turn_s'] = 2
assert np.isclose(line2.twiss()['px', 'ip1'], 100e-6, atol=1e-7, rtol = 0)
line2.vars['t_turn_s'] = 3
assert np.isclose(line2.twiss()['px', 'ip1'], 100e-6, atol=1e-7, rtol = 0)
line2.vars['t_turn_s'] = 4
assert np.isclose(line2.twiss()['px', 'ip1'], 50e-6, atol=1e-7, rtol = 0)
line2.vars['t_turn_s'] = 5
assert np.isclose(line2.twiss()['px', 'ip1'], 0, atol=1e-7, rtol = 0)
line2.vars['t_turn_s'] = 6
assert np.isclose(line2.twiss()['px', 'ip1'], 0, atol=1e-7, rtol = 0)


# Remember to test multiline2

