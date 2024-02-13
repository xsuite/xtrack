import xtrack as xt

# Load a line and build a tracker
line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')
line.build_tracker()

tw0 = line.twiss(method='4d')

# Knob optimizer for horizontal chromaticity
opt = line.match_knob('dqx.b1', knob_value_start=tw0.dqx, knob_value_end=3.0,
            run=False, method='4d',
            vary=xt.VaryList(['ksf.b1', 'ksd.b1'], step=1e-8),
            targets=xt.TargetSet(dqx=3.0, dqy=tw0, tol=1e-6))

# New terms have been added to knobs to vary
line.vars['ksf.b1']._expr # is: (0.0 + vars['ksf.b1_from_dqx.b1'])
line.vars['ksd.b1']._expr # is: (0.0 + vars['ksd.b1_from_dqx.b1'])
line.vars['ksf.b1_from_dqx.b1']._expr # is None
line.vars['ksd.b1_from_dqx.b1']._expr # is None

# optimized acts on newly created terms
opt.vary_status(); opt.target_status()
# prints:
#
# Vary status:
# id state tag name               lower_limit current_val upper_limit val_at_iter_0  step weight
#  0 ON        ksf.b1_from_dqx.b1        None           0        None             0 1e-08      1
#  1 ON        ksd.b1_from_dqx.b1        None           0        None             0 1e-08      1
# Target status:
# id state tag tol_met  residue current_val target_val description
#  0 ON          False -1.09005     1.90995          3 'dqx', val=3, tol=1e-06, weight=1
#  1 ON           True        0     1.94297    1.94297 'dqy', val=1.94297, tol=1e-06, weight=1

opt.solve() # perform optimization
opt.vary_status(); opt.target_status()
# prints:
#
# Vary status:
# id state tag name               lower_limit current_val upper_limit val_at_iter_0  step weight
#  0 ON        ksf.b1_from_dqx.b1        None  0.00130336        None             0 1e-08      1
#  1 ON        ksd.b1_from_dqx.b1        None  -0.0004024        None             0 1e-08      1
# Target status:
# id state tag tol_met     residue current_val target_val description
#  0 ON           True 7.94672e-08           3          3 'dqx', val=3, tol=1e-06, weight=1
#  1 ON           True           0     1.94297    1.94297 'dqy', val=1.94297, tol=1e-06, weight=1

# Generate the knob
opt.generate_knob()

line.vars['ksf.b1']._expr # is: (0.0 + vars['ksf.b1_from_dqx.b1'])
line.vars['ksd.b1']._expr # is: (0.0 + vars['ksd.b1_from_dqx.b1'])
line.vars['ksf.b1_from_dqx.b1']._expr
# is ((0.0011956933485755728 * vars['dqx.b1']) - 0.0022837181704350494)
line.vars['ksd.b1_from_dqx.b1']._expr # is None
# is ((-0.0003691583859286993 * vars['dqx.b1']) - -0.0007050751889840094)

# Create also vertical chromaticity knob
opt_dqy = line.match_knob('dqy.b1', knob_value_start=tw0.dqy, knob_value_end=3.0,
            run=False, method='4d',
            vary=xt.VaryList(['ksf.b1', 'ksd.b1'], step=1e-8),
            targets=xt.TargetSet(dqx=tw0, dqy=3.0, tol=1e-6))
opt_dqy.solve()
opt_dqy.generate_knob()

line.vars['ksf.b1']._expr
# is: ((0.0 + vars['ksf.b1_from_dqx.b1']) + vars['ksf.b1_from_dqy.b1'])
line.vars['ksd.b1']._expr
# is: ((0.0 + vars['ksd.b1_from_dqx.b1']) + vars['ksd.b1_from_dqy.b1'])
line.vars['ksf.b1_from_dqx.b1']._expr
# is ((0.0011956933485755728 * vars['dqx.b1']) - 0.0022837181704350494)
line.vars['ksd.b1_from_dqx.b1']._expr # is None
# is ((-0.0003691583859286993 * vars['dqx.b1']) - -0.0007050751889840094)
line.vars['ksf.b1_from_dqy.b1']._expr
# is ((0.0011956933485755728 * vars['dqy.b1']) - 0.0022837181704350494)
line.vars['ksd.b1_from_dqy.b1']._expr
# is ((-0.0003691583859286993 * vars['dqy.b1']) - -0.0007050751889840094)

# Test knobs
line.vars['dqx.b1'] = 5.
line.vars['dqy.b1'] = 6.

tw = line.twiss(method='4d')
tw.dqx # is 5.00000231
tw.dqy # is 5.99999987
