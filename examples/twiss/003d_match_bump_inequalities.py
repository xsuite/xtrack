import json

import numpy as np
import xtrack as xt
import xobjects as xo

with open('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json',
            'r') as fid:
    dct = json.load(fid)
line = xt.Line.from_dict(dct)

line.build_tracker()

tw_before = line.twiss()

GreaterThan = xt.GreaterThan
LessThan = xt.LessThan

opt = line.match(
    solve=False,
    solver='jacobian',
    # Portion of the beam line to be modified and initial conditions
    ele_start='mq.33l8.b1',
    ele_stop='mq.17l8.b1',
    twiss_init='preserve',
    # Dipole corrector strengths to be varied
    vary=[
        xt.Vary(name='acbv28.l8b1', step=1e-10),
        xt.Vary(name='acbv26.l8b1', step=1e-10),
        xt.Vary(name='acbv24.l8b1', step=1e-10),
        xt.Vary(name='acbv22.l8b1', step=1e-10),
    ],
    targets=[
        xt.Target('y', GreaterThan(2.7e-3), at='mb.b26l8.b1'),
        xt.Target('y', GreaterThan(2.7e-3), at='mb.b25l8.b1'),
        xt.Target('y', at='mq.24l8.b1', value=xt.LessThan(3e-3)),
        xt.TargetSet(['y', 'py'], at='mq.17l8.b1', value='preserve'),
    ]
)
opt.solve()

tw = line.twiss()
assert tw['y', 'mb.b26l8.b1'] > 2.7e-3
assert tw['y', 'mb.b25l8.b1'] > 2.7e-3
assert tw['y', 'mq.24l8.b1'] < 3e-3 + 1e-6
assert np.isclose(tw['y', 'mq.17l8.b1'], tw_before['y', 'mq.17l8.b1'], rtol=0, atol=1e-7)
assert np.isclose(tw['py', 'mq.17l8.b1'], tw_before['py', 'mq.17l8.b1'], rtol=0, atol=1e-9)

assert isinstance(opt.targets[0].value, xt.GreaterThan)
assert isinstance(opt.targets[1].value, xt.GreaterThan)
assert isinstance(opt.targets[2].value, xt.LessThan)

assert opt.targets[0].value._value == 0
assert opt.targets[1].value._value == 0
assert opt.targets[2].value._value == 0

assert opt.targets[0].value.mode == 'step'
assert opt.targets[1].value.mode == 'step'
assert opt.targets[2].value.mode == 'step'

assert opt.targets[0].value.lower == 2.7e-3
assert opt.targets[1].value.lower == 2.7e-3
assert opt.targets[2].value.upper == 3e-3


# Test mode smooth
# Remove the bump

for kk in ['acbv28.l8b1', 'acbv26.l8b1', 'acbv24.l8b1', 'acbv22.l8b1']:
    line.vars[kk] = 0

tw_before = line.twiss()
assert tw_before['y', 'mb.b26l8.b1'] < 1e-7
assert tw_before['y', 'mb.b25l8.b1'] < 1e-7


opt = line.match(
    solve=False,
    solver='jacobian',
    # Portion of the beam line to be modified and initial conditions
    ele_start='mq.33l8.b1',
    ele_stop='mq.17l8.b1',
    twiss_init='preserve',
    # Dipole corrector strengths to be varied
    vary=[
        xt.Vary(name='acbv28.l8b1', step=1e-10),
        xt.Vary(name='acbv26.l8b1', step=1e-10),
        xt.Vary(name='acbv24.l8b1', step=1e-10),
        xt.Vary(name='acbv22.l8b1', step=1e-10),
    ],
    targets=[
        xt.Target('y', GreaterThan(2.7e-3, mode='smooth', sigma_rel=0.05), at='mb.b26l8.b1'),
        xt.Target('y', GreaterThan(2.7e-3, mode='smooth'), at='mb.b25l8.b1'),
        xt.Target('y', at='mq.24l8.b1', value=xt.LessThan(3e-3, mode='smooth', sigma_rel=0.05)),
        xt.Target('y', at='mq.26l8.b1', value=xt.LessThan(6e-3, mode='smooth')),
        xt.TargetSet(['y', 'py'], at='mq.17l8.b1', value='preserve'),
    ]
)
opt.solve()

tw = line.twiss()

assert tw['y', 'mb.b26l8.b1'] > 2.7e-3 - 1e-6
assert tw['y', 'mb.b25l8.b1'] > 2.7e-3 - 1e-6
assert tw['y', 'mq.24l8.b1'] < 3e-3 + 1e-6
assert tw['y', 'mq.26l8.b1'] < 6e-3 + 1e-6
assert np.isclose(tw['y', 'mq.17l8.b1'], tw_before['y', 'mq.17l8.b1'], rtol=0, atol=1e-7)
assert np.isclose(tw['py', 'mq.17l8.b1'], tw_before['py', 'mq.17l8.b1'], rtol=0, atol=1e-9)

assert isinstance(opt.targets[0].value, xt.GreaterThan)
assert isinstance(opt.targets[1].value, xt.GreaterThan)
assert isinstance(opt.targets[2].value, xt.LessThan)
assert isinstance(opt.targets[3].value, xt.LessThan)

assert opt.targets[0].value._value == 0
assert opt.targets[1].value._value == 0
assert opt.targets[2].value._value == 0
assert opt.targets[3].value._value == 0

assert opt.targets[0].value.mode == 'smooth'
assert opt.targets[1].value.mode == 'smooth'
assert opt.targets[2].value.mode == 'smooth'
assert opt.targets[3].value.mode == 'smooth'

assert opt.targets[0].value.lower == 2.7e-3
assert opt.targets[1].value.lower == 2.7e-3
assert opt.targets[2].value.upper == 3e-3
assert opt.targets[3].value.upper == 6e-3

assert np.isclose(opt.targets[0].value.sigma, 0.05 * 2.7e-3, atol=0, rtol=1e-10)
assert np.isclose(opt.targets[1].value.sigma, 0.01 * 2.7e-3, atol=0, rtol=1e-10)
assert np.isclose(opt.targets[2].value.sigma, 0.05 * 3e-3, atol=0, rtol=1e-10)
assert np.isclose(opt.targets[3].value.sigma, 0.01 * 6e-3, atol=0, rtol=1e-10)

# Check smooth target
i_tar = 0
tar = opt.targets[i_tar]
sigma = tar.value.sigma
x0 = tar.runeval()

edge_test = np.linspace(x0 - 3 * sigma, x0 + 3 * sigma, 100)

residue = edge_test * 0
for ii, xx in enumerate(edge_test):
    tar.value.lower = xx
    residue[ii] =  opt._err()[i_tar] / tar.weight

x_minus_edge = x0 - edge_test

x_cut_norm = 1/16 + np.sqrt(33)/16
poly = lambda x: 3 * x**3 - 2 * x**4
x_cut = x_cut_norm * sigma

# assert np.all(residue[x_transf_fun > 0] == 0)
# assert np.allclose(residue[x_transf_fun < x_cut],
#                    x_transf_fun[x_transf_fun < x_cut] - x_cut + poly(x_cut_norm),
#                    atol=0, rtol=1e-10)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(100)
plt.plot(x_minus_edge, residue)
plt.plot(x_minus_edge, -x_minus_edge - x_cut + sigma*poly(x_cut_norm), '--')
plt.axvline(x=-x_cut, color='r', linestyle='--')
plt.axvline(x=-sigma, color='g', linestyle='--')



fig = plt.figure(1, figsize=(6.4*1.2, 4.8*0.8))
ax = fig.add_subplot(111)
ax.plot(tw_before.s, tw_before.y*1000, label='y')
ax.plot(tw.s, tw.y*1000, label='y')
# Target
ax.axvline(x=line.get_s_position('mb.b26l8.b1'), color='r', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mb.b25l8.b1'), color='r', linestyle='--', alpha=0.5)
# Correctors
ax.axvline(x=line.get_s_position('mcbv.32l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.28l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.26l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.24l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.22l8.b1'), color='k', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mcbv.18l8.b1'), color='k', linestyle='--', alpha=0.5)
# Boundaries
ax.axvline(x=line.get_s_position('mq.33l8.b1'), color='g', linestyle='--', alpha=0.5)
ax.axvline(x=line.get_s_position('mq.17l8.b1'), color='g', linestyle='--', alpha=0.5)
ax.set_xlim(line.get_s_position('mq.33l8.b1') - 10,
            line.get_s_position('mq.17l8.b1') + 10)
ax.set_xlabel('s [m]')
ax.set_ylabel('y [mm]')
ax.set_ylim(-10, 10)
plt.subplots_adjust(bottom=.152, top=.9, left=.1, right=.95)
plt.show()
