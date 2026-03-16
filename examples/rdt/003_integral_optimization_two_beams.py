import xtrack as xt

env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')

line_b1 = env['lhcb1']
line_b2 = env['lhcb2']

line_b2.twiss_default['reverse'] = False

line_b1.cycle('ip3')
line_b2.cycle('ip3')

tw_ref_b1 = line_b1.twiss4d()
tw_ref_b2 = line_b2.twiss4d()

env['mbrd.4l1.b1'].ksl[1] = -5e-5
# env['mbrd.4l1.b2'].ksl[1] = -6e-5
env['mbrd.4r1.b1'].ksl[1] = -3e-5
# env['mbrd.4r1.b2'].ksl[1] = 4e-5

env['mbrd.4l5.b1'].ksl[1] = -5e-5
# env['mbrd.4l5.b2'].ksl[1] = 6e-5
env['mbrd.4r5.b1'].ksl[1] = -3e-5
# env['mbrd.4r5.b2'].ksl[1] = 4e-5

tw_uncorr_b1 = line_b1.twiss4d()
tw_uncorr_b2 = line_b2.twiss4d()

# Local correction b1 ir1
integral_optim_ir1_b1  = xt.IntegralOptimization(
    twiss=tw_ref_b1,
    line=line_b1,
    start='e.ds.l1.b1', end='s.ds.r1.b1',
    vary=xt.VaryList(['kqsx3.r1', 'kqsx3.r1']),
    target_quantities={'f1001_b1': 'f1001'},
    generated_knob_name='on_corr_k1s_ir1'
)

# Local correction b2 ir1 (only targets here)
integral_optim_ir1_b2  = xt.IntegralOptimization(
    twiss=tw_ref_b2,
    line=line_b2,
    start='s.ds.r1.b2', end='e.ds.l1.b2',
    vary=[], # only targets here
    target_quantities={'f1001_b2': 'f1001'},
    generated_knob_name=None
)

# Build combined optimizer
opt_ir1_b1 = integral_optim_ir1_b1.get_optimizer()
opt_ir1_b2 = integral_optim_ir1_b2.get_optimizer()
opt_ir1 = opt_ir1_b1.opt.clone(add_targets=opt_ir1_b2.opt.targets)

# Optimize and generate knob
opt_ir1.step(n_steps=10)
opt_ir1_b1.generate_knob()

# Same for ir5
integral_optim_ir5_b1  = xt.IntegralOptimization(
    twiss=tw_ref_b1,
    line=line_b1,
    start='e.ds.l5.b1', end='s.ds.r5.b1',
    vary=xt.VaryList(['kqsx3.r5', 'kqsx3.r5']),
    target_quantities={'f1001_b1': 'f1001'},
    generated_knob_name='on_corr_k1s_ir5'
)

integral_optim_ir5_b2  = xt.IntegralOptimization(
    twiss=tw_ref_b2,
    line=line_b2,
    start='s.ds.r5.b2', end='e.ds.l5.b2',
    vary=[], # only targets here
    target_quantities={'f1001_b2': 'f1001'},
    generated_knob_name=None
)

opt_ir5_b1 = integral_optim_ir5_b1.get_optimizer()
opt_ir5_b2 = integral_optim_ir5_b2.get_optimizer()
opt_ir5 = opt_ir5_b1.opt.clone(add_targets=opt_ir5_b2.opt.targets)

opt_ir5.step(n_steps=10)
opt_ir5_b1.generate_knob()

# Corrections ON
env['on_corr_k1s_ir1'] = 1.0
env['on_corr_k1s_ir5'] = 1.0

tw_local_corr_b1 = line_b1.twiss4d()
tw_local_corr_b2 = line_b2.twiss4d()

# compare c_minus from twiss
print("Uncorrected:")
print("b1 c_minus:", tw_uncorr_b1.c_minus)
print("b2 c_minus:", tw_uncorr_b2.c_minus)
print("Local correction:")
print("b1 c_minus:", tw_local_corr_b1.c_minus)
print("b2 c_minus:", tw_local_corr_b2.c_minus)