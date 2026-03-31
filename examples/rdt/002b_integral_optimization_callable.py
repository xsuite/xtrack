import xtrack as xt
import numpy as np
from xtrack.feed_down import feed_down

env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')

line = env['lhcb1']

tw_ref = line.twiss4d()

env['tilt12'] = 1
env['tilt56'] = 1

# Tilt a few quadrupoles in sector 12
env['mq.20r1.b1'].rot_s_rad = env.ref['tilt12'] * 10e-3
env['mq.24r1.b1'].rot_s_rad = env.ref['tilt12'] * 5e-3
env['mq.18l2.b1'].rot_s_rad = env.ref['tilt12'] * 7e-3
env['mq.22l2.b1'].rot_s_rad = env.ref['tilt12'] * 10e-3

# Tilt a few quadrupoles in sector 56
env['mq.20r5.b1'].rot_s_rad = env.ref['tilt56'] * 10e-3
env['mq.24r5.b1'].rot_s_rad = env.ref['tilt56'] * 10e-3
env['mq.18l6.b1'].rot_s_rad = env.ref['tilt56'] * 10e-3
env['mq.22l6.b1'].rot_s_rad = env.ref['tilt56'] * 10e-3

tw = line.twiss4d()

def f1001(tw, tt):

    knl = np.zeros(shape=(len(tt),6))
    ksl = np.zeros(shape=(len(tt),6))
    for ii in range(6):
        knl[:, ii] = tt[f'k{ii}l']
        ksl[:, ii] = tt[f'k{ii}sl']

    knl_eff, ksl_eff = feed_down(
            kn=knl,
            kskew=ksl,
            shift_x=tt.shift_x,
            shift_y=tt.shift_y,
            psi=tt.rot_s_rad,
            x0=0,
            y0=0,
            max_output_order=1,
        )

    betx = tw.betx
    bety = tw.bety
    mux = tw.mux
    muy = tw.muy
    k1sl = ksl_eff[:, 1]
    return k1sl * np.sqrt(betx * bety) * np.exp(1j * 2 * np.pi * (mux - muy))

def f1010(tw, tt):

    knl = np.zeros(shape=(len(tt),6))
    ksl = np.zeros(shape=(len(tt),6))
    for ii in range(6):
        knl[:, ii] = tt[f'k{ii}l']
        ksl[:, ii] = tt[f'k{ii}sl']
    knl_eff, ksl_eff = feed_down(
            kn=knl,
            kskew=ksl,
            shift_x=tt.shift_x,
            shift_y=tt.shift_y,
            psi=tt.rot_s_rad,
            x0=0,
            y0=0,
            max_output_order=1,
        )

    betx = tw.betx
    bety = tw.bety
    mux = tw.mux
    muy = tw.muy
    k1sl = ksl_eff[:, 1]
    return k1sl * np.sqrt(betx * bety) * np.exp(1j * 2 * np.pi * (mux + muy))

integral_optim_12 = xt.IntegralOptimization(
    twiss=tw,
    start='s.ds.r1.b1',
    end='e.ds.l2.b1',
    line=line,
    vary=xt.VaryList(['kqs.r1b1', 'kqs.l2b1'], step=1e-5),
    target_quantities={'coupl_1': f1001, 'coupl_2': f1010},
    generated_knob_name='on_corr_coupl_12'
)

integral_optim_56 = xt.IntegralOptimization(
    twiss=tw,
    start='s.ds.r5.b1',
    end='e.ds.l6.b1',
    line=line,
    vary=xt.VaryList(['kqs.r5b1', 'kqs.l6b1'], step=1e-5),
    target_quantities={'coupl_1': f1001, 'coupl_2': f1010},
    generated_knob_name='on_corr_coupl_56'
)

assert line.twiss4d().c_minus > 5e-2 # Because correction of 56 is not there

opt12 = integral_optim_12.correct(n_steps=10)
opt56 = integral_optim_56.correct(n_steps=10)

assert line.twiss4d().c_minus < 1e-3

# Disable correction of 56
env['on_corr_coupl_56'] = 0.0
assert line.twiss4d().c_minus > 2e-2

# Disable source in 56 (check that correction of 12 is fully local)
env['tilt56'] = 0.0
assert line.twiss4d().c_minus < 1e-3

# Disable correction of 12
env['on_corr_coupl_12'] = 0.0
assert line.twiss4d().c_minus > 2e-2

# Disable source in 12
env['tilt12'] = 0.0
assert line.twiss4d().c_minus < 1e-3

# Enable source in 56
env['tilt56'] = 1.0
assert line.twiss4d().c_minus > 2e-2

# Enable correction of 56 (check that correction of 56 is fully local)
env['on_corr_coupl_56'] = 1.0
assert line.twiss4d().c_minus < 1e-3

