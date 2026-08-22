import xtrack as xt
import numpy as np

env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')

line_b1 = env['lhcb1']
line_b2 = env['lhcb2']

line_b2.twiss_default['reverse'] = False

line_b1.cycle('ip3')
line_b2.cycle('ip3')

# Twiss on clean machine
tw_ref_b1 = line_b1.twiss4d()
tw_ref_b2 = line_b2.twiss4d()

# Apply errors
env['mbrd.4l1.b1'].ksl[1] = -3e-5
env['mbrd.4l1.b2'].ksl[1] = -2e-5
env['mbrd.4r1.b1'].ksl[1] = -1e-5
env['mbrd.4r1.b2'].ksl[1] = 2e-5

env['mbrd.4l5.b1'].ksl[1] = -3e-5
env['mbrd.4l5.b2'].ksl[1] = 1e-5
env['mbrd.4r5.b1'].ksl[1] = -1e-5
env['mbrd.4r5.b2'].ksl[1] = 2e-5

# Twiss on uncorrected machine
tw_uncorr_b1 = line_b1.twiss4d()
tw_uncorr_b2 = line_b2.twiss4d()

# Local correction using common skew quadrupole correctors in IRs
for ipn in [1, 5]:

    integral_optim_ir_b1  = xt.IntegralOptimization(
        twiss=tw_ref_b1,
        line=line_b1,
        start=f'e.ds.l{ipn}.b1', end=f's.ds.r{ipn}.b1',
        vary=xt.VaryList([f'kqsx3.r{ipn}', f'kqsx3.l{ipn}'], step=1e-6),
        target_quantities={'f1001_b1': 'f1001'},
        generated_knob_name=f'on_corr_k1s_ir{ipn}_local'
    )

    integral_optim_ir_b2  = xt.IntegralOptimization(
        twiss=tw_ref_b2,
        line=line_b2,
        start=f's.ds.r{ipn}.b2', end=f'e.ds.l{ipn}.b2',
        vary=[], # only targets here
        target_quantities={'f1001_b2': 'f1001'},
        generated_knob_name=None
    )

    # Build combined optimizer
    opt_ir_b1 = integral_optim_ir_b1.get_optimizer()
    opt_ir_b2 = integral_optim_ir_b2.get_optimizer()
    opt_ir = opt_ir_b1.opt.clone(add_targets=opt_ir_b2.opt.targets)

    # impose the two correctors are powered equally
    # (they are not independed because the phase advance is pi and leaving them free
    # leads to an ill-conditioned optimization problem)
    env[f'kqsx3.l{ipn}_from_on_corr_k1s_ir{ipn}_local'] = f'kqsx3.r{ipn}_from_on_corr_k1s_ir{ipn}_local'
    opt_ir.disable(vary_name=f'kqsx3.l{ipn}_from_on_corr_k1s_ir{ipn}_local')
    opt_ir.step(n_steps=5)
    opt_ir_b1.generate_knob() # knob generation was managed through b1 (vary passaed there)

# Correction knobs switched on
env['on_corr_k1s_ir1_local'] = 1.0
env['on_corr_k1s_ir5_local'] = 1.0

# Twiss after local correction
tw_local_corr_b1 = line_b1.twiss4d()
tw_local_corr_b2 = line_b2.twiss4d()

# Global correction using all skew quadrupole correctors in arcs
opt_b1 = line_b1.match_knob(
    run=False,
    vary=xt.VaryList(
       ['kqs.r1b1', 'kqs.l2b1', 'kqs.a23b1', 'kqs.r3b1', 'kqs.l4b1',
       'kqs.a45b1', 'kqs.r5b1', 'kqs.l6b1', 'kqs.a67b1', 'kqs.r7b1',
       'kqs.l8b1', 'kqs.a81b1'], step=1e-7),
    targets=xt.TargetSet(c_minus_re_0=0, c_minus_im_0=0),
    knob_name='on_corr_k1s_global_b1'
)
opt_b1.step(5)
opt_b1.generate_knob()

opt_b2 = line_b2.match_knob(
    run=False,
    vary=xt.VaryList(
        ['kqs.a12b2', 'kqs.r2b2', 'kqs.l3b2', 'kqs.a34b2', 'kqs.r4b2',
         'kqs.l5b2', 'kqs.a56b2', 'kqs.r6b2', 'kqs.l7b2', 'kqs.a78b2',
         'kqs.r8b2', 'kqs.l1b2'], step=1e-7),
    targets=xt.TargetSet(c_minus_re_0=0, c_minus_im_0=0),
    knob_name='on_corr_k1s_global_b2'
)
opt_b2.step(5)
opt_b2.generate_knob()

env['on_corr_k1s_global_b1'] = 1.0
env['on_corr_k1s_global_b2'] = 1.0

# Twiss after global correction
tw_global_corr_b1 = line_b1.twiss4d()
tw_global_corr_b2 = line_b2.twiss4d()

# compare c_minus from twiss
print("Uncorrected:")
print("b1 c_minus:", tw_uncorr_b1.c_minus)
print("b2 c_minus:", tw_uncorr_b2.c_minus)
print("Local correction:")
print("b1 c_minus:", tw_local_corr_b1.c_minus)
print("b2 c_minus:", tw_local_corr_b2.c_minus)
print("Global correction:")
print("b1 c_minus:", tw_global_corr_b1.c_minus)
print("b2 c_minus:", tw_global_corr_b2.c_minus)

twb1 = line_b1.twiss4d(coupling_edw_teng=True, strengths=True)
twb2 = line_b2.twiss4d(coupling_edw_teng=True, strengths=True)

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(twb1.s, twb1.k1sl, label='b1')
plt.xlabel('s [m]')
plt.ylabel('k1sl [1/m]')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(twb2.s, twb2.k1sl, label='b2')
plt.xlabel('s [m]')
plt.ylabel('k1sl [1/m]')
plt.legend()

plt.figure(2)
plt.subplot(2, 1, 1)
plt.plot(twb1.s, np.abs(twb1.f1001), label='b1')
plt.xlabel('s [m]')
plt.ylabel('|f1001|')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(twb2.s, np.abs(twb2.f1001), label='b2')
plt.xlabel('s [m]')
plt.ylabel('|f1001|')
plt.legend()

plt.show()
