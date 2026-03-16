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
env['mbrd.4l1.b2'].ksl[1] = -6e-5
env['mbrd.4r1.b1'].ksl[1] = -3e-5
env['mbrd.4r1.b2'].ksl[1] = 4e-5

env['mbrd.4l5.b1'].ksl[1] = -5e-5
env['mbrd.4l5.b2'].ksl[1] = 6e-5
env['mbrd.4r5.b1'].ksl[1] = -3e-5
env['mbrd.4r5.b2'].ksl[1] = 4e-5

tw_uncorr_b1 = line_b1.twiss4d()
tw_uncorr_b2 = line_b2.twiss4d()


# Local correction IP1 right
integral_optim_ir1_b1  = xt.IntegralOptimization(
    twiss=tw_ref_b1,
    line=line_b1,
    start='ip1', end='s.ds.r1.b1',
    vary=xt.VaryList(['kqsx3.r1'], step=1e-6),
    target_quantities={'f1001_b1': 'f1001'},
    generated_knob_name='on_corr_k1s_r1'
)

integral_optim_ir1_b2  = xt.IntegralOptimization(
    twiss=tw_ref_b2,
    line=line_b2,
    start='s.ds.r1.b2', end='ip1',
    vary=[], # only targets here
    target_quantities={'f1001_b2': 'f1001'},
    generated_knob_name=None
)

# Build combined optimizer
opt_r1_b1 = integral_optim_ir1_b1.get_optimizer()
opt_r1_b2 = integral_optim_ir1_b2.get_optimizer()
opt_r1 = opt_r1_b1.opt.clone(add_targets=opt_r1_b2.opt.targets)


# Local correction IP1
integral_optim_ir1_left_b1  = xt.IntegralOptimization(
    twiss=tw_ref_b1,
    line=line_b1,
    start='e.ds.l1.b1', end='ip1',
    vary=xt.VaryList(['kqsx3.l1'], step=1e-6),
    target_quantities={'f1001_b1': 'f1001'},
    generated_knob_name='on_corr_k1s_l1'
)
integral_optim_ir1_left_b2  = xt.IntegralOptimization(
    twiss=tw_ref_b2,
    line=line_b2,
    start='ip1', end='e.ds.l1.b2',
    vary=[], # only targets here
    target_quantities={'f1001_b2': 'f1001'},
    generated_knob_name=None
)
opt_l1_b1 = integral_optim_ir1_left_b1.get_optimizer()
opt_l1_b2 = integral_optim_ir1_left_b2.get_optimizer()
opt_l1 = opt_l1_b1.opt.clone(add_targets=opt_l1_b2.opt.targets)

opt_r1.step(n_steps=5)
opt_l1.step(n_steps=5)
opt_r1_b1.generate_knob()
opt_l1_b1.generate_knob()

# Local correction IP5 right
integral_optim_ir5_b1  = xt.IntegralOptimization(
    twiss=tw_ref_b1,
    line=line_b1,
    start='ip5', end='s.ds.r5.b1',
    vary=xt.VaryList(['kqsx3.r5'], step=1e-6),
    target_quantities={'f1001_b1': 'f1001'},
    generated_knob_name='on_corr_k1s_r5'
)
integral_optim_ir5_b2  = xt.IntegralOptimization(
    twiss=tw_ref_b2,
    line=line_b2,
    start='s.ds.r5.b2', end='ip5',
    vary=[], # only targets here
    target_quantities={'f1001_b2': 'f1001'},
    generated_knob_name=None
)
opt_r5_b1 = integral_optim_ir5_b1.get_optimizer()
opt_r5_b2 = integral_optim_ir5_b2.get_optimizer()
opt_r5 = opt_r5_b1.opt.clone(add_targets=opt_r5_b2.opt.targets)

integral_optim_ir5_left_b1  = xt.IntegralOptimization(
    twiss=tw_ref_b1,
    line=line_b1,
    start='e.ds.l5.b1', end='ip5',
    vary=xt.VaryList(['kqsx3.l5'], step=1e-6),
    target_quantities={'f1001_b1': 'f1001'},
    generated_knob_name='on_corr_k1s_l5'
)
integral_optim_ir5_left_b2  = xt.IntegralOptimization(
    twiss=tw_ref_b2,
    line=line_b2,
    start='ip5', end='e.ds.l5.b2',
    vary=[], # only targets here
    target_quantities={'f1001_b2': 'f1001'},
    generated_knob_name=None
)
opt_l5_b1 = integral_optim_ir5_left_b1.get_optimizer()
opt_l5_b2 = integral_optim_ir5_left_b2.get_optimizer()
opt_l5 = opt_l5_b1.opt.clone(add_targets=opt_l5_b2.opt.targets)

# Corrections ON
env['on_corr_k1s_r1'] = 1.0
env['on_corr_k1s_l1'] = 1.0
env['on_corr_k1s_r5'] = 1.0
env['on_corr_k1s_l5'] = 1.0

tw_local_corr_b1 = line_b1.twiss4d()
tw_local_corr_b2 = line_b2.twiss4d()


# Local with neighboring arcs
integral_optim_81_12_b1  = xt.IntegralOptimization(
    twiss=tw_ref_b1,
    line=line_b1,
    start='ip8', end='ip2',
    vary=xt.VaryList(['kqs.a81b1', 'kqs.r1b1', 'kqs.l2b1'], step=1e-6),
    target_quantities={'f1001_b1': 'f1001'},
    generated_knob_name='on_corr_k1s_ir1_long_b1'
)

integral_optim_81_12_b2  = xt.IntegralOptimization(
    twiss=tw_ref_b2,
    line=line_b2,
    start='ip2', end='ip8',
    vary=xt.VaryList(['kqs.a12b2', 'kqs.r8b2', 'kqs.l1b2'], step=1e-6),
    target_quantities={'f1001_b2': 'f1001'},
    generated_knob_name='on_corr_k1s_ir1_long_b2'
)

integral_optim_45_56_b1  = xt.IntegralOptimization(
    twiss=tw_ref_b1,
    line=line_b1,
    start='ip4', end='ip6',
    vary=xt.VaryList(['kqs.a45b1', 'kqs.r5b1', 'kqs.l6b1'], step=1e-6),
    target_quantities={'f1001_b1': 'f1001'},
    generated_knob_name='on_corr_k1s_ir5_long_b1'
)

integral_optim_45_56_b2  = xt.IntegralOptimization(
    twiss=tw_ref_b2,
    line=line_b2,
    start='ip6', end='ip4',
    vary=xt.VaryList(['kqs.a56b2', 'kqs.r6b2', 'kqs.l5b2'], step=1e-6),
    target_quantities={'f1001_b2': 'f1001'},
    generated_knob_name='on_corr_k1s_ir5_long_b2'
)

integral_optim_81_12_b1.correct(n_steps=5)
integral_optim_81_12_b2.correct(n_steps=5)
integral_optim_45_56_b1.correct(n_steps=5)
integral_optim_45_56_b2.correct(n_steps=5)

tw_local_corr_long_b1 = line_b1.twiss4d()
tw_local_corr_long_b2 = line_b2.twiss4d()

# Global correction
integral_optim_global_b1  = xt.IntegralOptimization(
    twiss=tw_ref_b1,
    line=line_b1,
    vary=xt.VaryList(
       ['kqs.r1b1', 'kqs.l2b1', 'kqs.a23b1', 'kqs.r3b1', 'kqs.l4b1',
       'kqs.a45b1', 'kqs.r5b1', 'kqs.l6b1', 'kqs.a67b1', 'kqs.r7b1',
       'kqs.l8b1', 'kqs.a81b1'], step=1e-6),
    target_quantities={'f1001_b1': 'f1001', 'f1010_b1': 'f1010'},
    generated_knob_name='on_corr_k1s_global_b1'
)

integral_optim_global_b2  = xt.IntegralOptimization(
    twiss=tw_ref_b2,
    line=line_b2,
    vary=xt.VaryList(
        ['kqs.a12b2', 'kqs.r2b2', 'kqs.l3b2', 'kqs.a34b2', 'kqs.r4b2',
         'kqs.l5b2', 'kqs.a56b2', 'kqs.r6b2', 'kqs.l7b2', 'kqs.a78b2',
         'kqs.r8b2', 'kqs.l1b2'], step=1e-6),
    target_quantities={'f1001_b2': 'f1001', 'f1010_b2': 'f1010'},
    generated_knob_name='on_corr_k1s_global_b2'
)

integral_optim_global_b1.correct(n_steps=10)
integral_optim_global_b2.correct(n_steps=10)

tw_global_corr_b1 = line_b1.twiss4d()
tw_global_corr_b2 = line_b2.twiss4d()

rdts = xt.rdt_first_order_perturbation(
    rdt=['f1001', 'f1010'],
    twiss=tw_ref_b1,
    strengths=line_b1.get_table(attr=True)
)

# compare c_minus from twiss
print("Uncorrected:")
print("b1 c_minus:", tw_uncorr_b1.c_minus)
print("b2 c_minus:", tw_uncorr_b2.c_minus)
print("Local correction:")
print("b1 c_minus:", tw_local_corr_b1.c_minus)
print("b2 c_minus:", tw_local_corr_b2.c_minus)
print("Local correction with neighboring arcs:")
print("b1 c_minus:", tw_local_corr_long_b1.c_minus)
print("b2 c_minus:", tw_local_corr_long_b2.c_minus)
print("Global correction:")
print("b1 c_minus:", tw_global_corr_b1.c_minus)
print("b2 c_minus:", tw_global_corr_b2.c_minus)

twb1 = line_b1.twiss4d(coupling_edw_teng=True)
twb2 = line_b2.twiss4d(coupling_edw_teng=True)

rdt_b1 = xt.rdt_first_order_perturbation(
    rdt=['f1001', 'f1010'],
    twiss=twb1,
    strengths=line_b1.get_table(attr=True)
)
