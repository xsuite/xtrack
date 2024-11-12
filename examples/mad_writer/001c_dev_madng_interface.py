import numpy as np
import xtrack as xt
import xobjects as xo

rdts = ["f4000", "f3100", "f2020", "f1120", 'f1001']

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

line['test_dk1'] = 0
line['mb.b32l8.b1'].knl[1] = 'test_dk1'

# sequence_name='dummy'
# mng = line.to_madng(sequence_name=sequence_name)
# mng._sequence_name = sequence_name

# line.build_tracker()
# mvars = MadngVars(mng)
# line.tracker.vars_to_update = [mvars]

# line['a'] = 3.
# assert mng.MADX.a == 3.

tw = line.madng_twiss(normal_form=False)

opt = line.match(
    solve=False,
    vary=[
        xt.Vary('on_x1', step=1e-3),
    ],
    targets=(
        tw.target('px_ng', 50e-6, at='ip1'),
        tw.target('py_ng', 0, at='ip1'),
    ),
)
opt.step(3)
tw_after = line.madng_twiss(normal_form=False)
xo.assert_allclose(tw_after['px_ng', 'ip1'], 50e-6, rtol=5e-3, atol=0)

# tw = line.madng_twiss()
# opt = line.match(
#     solve=False,
#     vary=[
#         xt.VaryList([
#             'kof.a78b1', 'kof.a81b1', 'kof.a12b1', 'kof.a23b1',
#             'kof.a34b1', 'kof.a45b1', 'kof.a56b1', 'kof.a67b1'], step=1e-2),
#         xt.VaryList(
#             ['kod.a78b1', 'kod.a81b1', 'kod.a12b1', 'kod.a23b1',
#              'kod.a34b1', 'kod.a45b1', 'kod.a56b1', 'kod.a67b1'], step=1e-2),
#     ],
#     targets=(
#         tw.target('dqxdjx_nf_ng', 1e6),
#         tw.target('dqydjy_nf_ng', 1e6),
#     ),
# )
# opt.step()


line['on_x1'] = 1.

xo.assert_allclose(line.madng_twiss()['px_ng', 'ip1'], 1e-6, rtol=5e-3, atol=0)

line['on_x1'] = -2.
xo.assert_allclose(line.madng_twiss()['px_ng', 'ip1'], -2e-6, rtol=5e-3, atol=0)

tw_rdt = line.madng_twiss(rdts=rdts)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)

plt.plot(tw_rdt.s, np.abs(tw_rdt.f4000), label='f4000')
plt.plot(tw_rdt.s, np.abs(tw_rdt.f2020), label='f2020')
plt.plot(tw_rdt.s, np.abs(tw_rdt.f1120), label='f1120')
plt.plot(tw_rdt.s, np.abs(tw_rdt.f3100), label='f3100')
plt.xlabel('s [m]')
plt.ylabel(r'|f_{jklm}|')
plt.legend()

plt.show()