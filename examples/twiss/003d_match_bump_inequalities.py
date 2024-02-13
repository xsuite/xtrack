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

tw0 = line.twiss()
opt = line.match(
    solve=False,
    solver='jacobian',
    # Portion of the beam line to be modified and initial conditions
    start='mq.33l8.b1',
    end='mq.17l8.b1',
    init=tw0, init_at=xt.START,
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
        xt.Target('y', at='mq.26l8.b1', value=xt.LessThan(6e-3)),
        xt.TargetSet(['y', 'py'], at='mq.17l8.b1', value=tw0),
    ]
)


# Check freeze
opt.step(1)
ts = opt.target_status(ret=True)
assert ts.tol_met[2] == False
assert ts.tol_met[4] == False
assert ts.tol_met[5] == False

opt.targets[2].freeze()
opt.targets[5].freeze()

ts = opt.target_status(ret=True)
assert ts.tol_met[2] == True
assert ts.tol_met[4] == False
assert ts.tol_met[5] == True

opt.targets[2].unfreeze()
opt.targets[5].unfreeze()

ts = opt.target_status(ret=True)
assert ts.tol_met[2] == False
assert ts.tol_met[4] == False
assert ts.tol_met[5] == False

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
    start='mq.33l8.b1',
    end='mq.17l8.b1',
    init=tw_before, init_at=xt.START,
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
        xt.Target('y', at='mq.24l8.b1', value=xt.LessThan(3e-3, mode='smooth', sigma_rel=0.04)),
        xt.Target('y', at='mq.26l8.b1', value=xt.LessThan(6e-3, mode='smooth')),
        xt.TargetSet(['y', 'py'], at='mq.17l8.b1', value=tw_before),
    ]
)

# Check freeze
opt.step(1)
ts = opt.target_status(ret=True)
assert ts.tol_met[0] == False
assert ts.tol_met[1] == False
assert ts.tol_met[5] == False

opt.targets[0].freeze()
opt.targets[5].freeze()

ts = opt.target_status(ret=True)
assert ts.tol_met[0] == True
assert ts.tol_met[1] == False
assert ts.tol_met[5] == True

opt.targets[0].unfreeze()
opt.targets[5].unfreeze()

ts = opt.target_status(ret=True)
assert ts.tol_met[0] == False
assert ts.tol_met[1] == False
assert ts.tol_met[5] == False

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
assert np.isclose(opt.targets[2].value.sigma, 0.04 * 3e-3, atol=0, rtol=1e-10)
assert np.isclose(opt.targets[3].value.sigma, 0.01 * 6e-3, atol=0, rtol=1e-10)

x_cut_norm = 1/16 + np.sqrt(33)/16
poly = lambda x: 3 * x**3 - 2 * x**4

# Check smooth target (GreaterThan)
i_tar_gt = 0
tar_gt = opt.targets[i_tar_gt]
sigma_gt = tar_gt.value.sigma
x0_gt = tar_gt.runeval()

edge_test_gt = np.linspace(x0_gt - 3 * sigma_gt, x0_gt + 3 * sigma_gt, 100)

residue_gt = edge_test_gt * 0
for ii, xx in enumerate(edge_test_gt):
    tar_gt.value.lower = xx
    residue_gt[ii] =  opt._err()[i_tar_gt] / tar_gt.weight

x_minus_edge_gt = x0_gt - edge_test_gt
x_cut_gt = -x_cut_norm * sigma_gt

mask_zero_gt = x_minus_edge_gt > 0
assert np.all(residue_gt[mask_zero_gt] == 0)
mask_linear_gt = x_minus_edge_gt < x_cut_gt
assert np.allclose(residue_gt[mask_linear_gt],
    -x_minus_edge_gt[mask_linear_gt] - x_cut_norm * sigma_gt + sigma_gt*poly(x_cut_norm),
    atol=0, rtol=1e-10)
mask_poly_gt = (~mask_zero_gt) & (~mask_linear_gt)
assert np.allclose(residue_gt[mask_poly_gt],
    sigma_gt * poly(-x_minus_edge_gt[mask_poly_gt]/sigma_gt),
    atol=0, rtol=1e-10)

# Check smooth target (LessThan)
i_tar_lt = 2
tar_lt = opt.targets[i_tar_lt]
sigma_lt = tar_lt.value.sigma
x0_lt = tar_lt.runeval()

edge_test_lt = np.linspace(x0_lt - 3 * sigma_lt, x0_lt + 3 * sigma_lt, 100)

residue_lt = edge_test_lt * 0
for ii, xx in enumerate(edge_test_lt):
    tar_lt.value.upper = xx
    residue_lt[ii] =  opt._err()[i_tar_lt] / tar_lt.weight

x_minus_edge_lt = x0_lt - edge_test_lt
x_cut_lt = x_cut_norm * sigma_lt

mask_zero_lt = x_minus_edge_lt < 0
assert np.all(residue_lt[mask_zero_lt] == 0)
mask_linear_lt = x_minus_edge_lt > x_cut_lt
assert np.allclose(residue_lt[mask_linear_lt],
    x_minus_edge_lt[mask_linear_lt] - x_cut_norm * sigma_lt + sigma_lt*poly(x_cut_norm),
    atol=0, rtol=1e-10)
mask_poly_lt = (~mask_zero_lt) & (~mask_linear_lt)
assert np.allclose(residue_lt[mask_poly_lt],
    sigma_lt * poly(x_minus_edge_lt[mask_poly_lt]/sigma_lt),
    atol=0, rtol=1e-10)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(100)
plt.suptitle('GreaterThan')
plt.plot(x_minus_edge_gt, residue_gt)
plt.plot(x_minus_edge_gt,
    -x_minus_edge_gt - x_cut_norm * sigma_gt + sigma_gt*poly(x_cut_norm),
    '--', color='C1')
plt.plot(x_minus_edge_gt, sigma_gt * poly(-x_minus_edge_gt/sigma_gt), '--', color='C1')
plt.axvline(x=-x_cut_gt, color='r', linestyle='--')
plt.axvline(x=0, color='r', linestyle='--')
plt.axvline(x=-sigma_gt, color='g', linestyle='--')
plt.ylim(np.array([-1, 1]) * np.max(np.abs(x_minus_edge_gt)))

plt.figure(101)
plt.suptitle('LessThan')
plt.plot(x_minus_edge_lt, residue_lt)
plt.plot(x_minus_edge_lt,
    x_minus_edge_lt - x_cut_norm * sigma_lt + sigma_lt*poly(x_cut_norm),
    '--', color='C1')
plt.plot(x_minus_edge_lt, sigma_lt * poly(x_minus_edge_lt/sigma_lt), '--', color='C1')
plt.axvline(x=x_cut_lt, color='r', linestyle='--')
plt.axvline(x=0, color='r', linestyle='--')
plt.axvline(x=sigma_lt, color='g', linestyle='--')
plt.ylim(np.array([-1, 1]) * np.max(np.abs(x_minus_edge_lt)))



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
