import numpy as np
import xtrack as xt
import xobjects as xo

class MadngVars:

    def __init__(self, mad):
        self.mad = mad

    def __setitem__(self, key, value):
        setattr(self.mad.MADX, key.replace('.', '_'), value)
        #Expressions still to be handled, could use the following:
        # mng.send(
        #     MADX:open_env()
        #     a = 3
        #     b =\ 3 * a
        #     c =\ 4 * a
        #     MADX:close_env()
        #     ''')

def _build_madng_model(line, sequence_name='seq'):
    if line.tracker is None:
        line.build_tracker()
    mng = line.to_madng(sequence_name=sequence_name)
    mng._sequence_name = sequence_name
    line.tracker._madng = mng
    line.tracker.vars_to_update = [MadngVars(mng)]
    return mng

rdts = ["f4000", "f3100", "f2020", "f1120", 'f1001']

def _tw_ng(line, rdts=[], normal_form=True,
           mapdef_twiss=2, mapdef_normal_form=4
           ):

    tw_kwargs = locals()
    del tw_kwargs['line']
    _action = ActionTwissMadng(line, tw_kwargs)

    if not hasattr(line.tracker, '_madng'):
        line._build_madng_model()
    mng = line.tracker._madng

    tw = line.twiss(method='4d', reverse=False)
    tw._action = _action

    tw_columns = ['s', 'beta11', 'beta22', 'alfa11', 'alfa22',
                'x', 'px', 'y', 'py', 't', 'pt',
                'dx', 'dy', 'dpx', 'dpy', 'mu1', 'mu2']

    columns = tw_columns + rdts
    rdt_cmd = 'local rdts = {"' + '", "'.join(rdts) + '"}'
    send_cmd = f'py:send({{mtbl.{", mtbl.".join(columns)}}})'

    if len(rdts) > 0:
        mng_script = ('''
        local damap in MAD
        '''
        f'local seq = MADX.{mng._sequence_name}'
        '''
        -- list of RDTs
        '''
        + rdt_cmd +
        '''
        -- create phase-space damap at 4th order
        local X0 = damap {nv=6, mo=4}

        -- twiss with RDTs
        local mtbl = twiss {sequence=seq, X0=X0, trkrdt=rdts, info=2, saverdt=true}

        -- send columns to Python
        '''
        + send_cmd)
    else:
        mng_script = ('''
        local damap in MAD
        '''
        f'local seq = MADX.{mng._sequence_name}'
        '''

        -- twiss with RDTs
        local mtbl = twiss {sequence=seq, method=4,'''
        f'mapdef={mapdef_twiss}'
        ''', implicit=true, nslice=3}

        -- send columns to Python
        '''
        + send_cmd)

    mng.send(mng_script)

    out = mng.recv()
    out_dct = {k: v for k, v in zip(columns, out)}

    # Add to table
    assert len(out[0]) == len(tw) + 1
    for nn in tw_columns:
        tw[nn+'_ng'] = np.atleast_1d(np.squeeze(out_dct[nn]))[:-1]
    for nn in rdts:
        tw[nn] = np.atleast_1d(np.squeeze(out_dct[nn]))[:-1]

    if normal_form:
        mng_script_nf = (
            f'local seq = MADX.{mng._sequence_name}'
            '''
            local track in MAD  -- like "from MAD import track"
            local mytrktable, mytrkflow = MAD.track{sequence=seq, method=4,'''
            f'mapdef={mapdef_normal_form}, '
            '''nslice=3}

            local normal in MAD.gphys  -- like "from MAD.gphys import normal"
            local my_norm_for = normal(mytrkflow[1]):analyse('anh') -- anh stands for anharmonicity

            local nf = my_norm_for
            last_nf = my_norm_for
            py:send({
                    nf:q1{1}, -- qx from the normal form (fractional part)
                    nf:q2{1}, -- qy
                    nf:dq1{1}, -- dqx / d delta
                    nf:dq2{1}, -- dqy / d delta
                    nf:dq1{2}, -- d2 qx / d delta2
                    nf:dq2{2}, -- d2 qy / d delta2
                    nf:dq1{3}, -- d3 qx / d delta3
                    nf:dq2{3}, -- d3 qy / d delta3
                    nf:dq1{4}, -- d4 qx / d delta4
                    nf:dq2{4}, -- d4 qy / d delta4
                    nf:dq1{5}, -- d5 qx / d delta5
                    nf:dq2{5}, -- d5 qy / d delta5
                    nf:anhx{1, 0}, -- dqx / d(2 jx)
                    nf:anhy{0, 1}, -- dqy / d(2 jy)
                    nf:anhx{0, 1}, -- dqx / d(2 jy)
                    nf:anhy{1, 0}, -- dqy / d(2 jx)
                    })
        ''')
        mng.send(mng_script_nf)
        out_nf = mng.recv()

        dct_nf = dict(
            q1 =   out_nf[0],
            q2 =   out_nf[1],
            dq1 =  out_nf[2],
            dq2 =  out_nf[3],
            d2q1 = out_nf[4],
            d2q2 = out_nf[5],
            d3q1 = out_nf[6],
            d3q2 = out_nf[7],
            d4q1 = out_nf[8],
            d4q2 = out_nf[9],
            d5q1 = out_nf[10],
            d5q2 = out_nf[11],
            dqxdjx = out_nf[6]*2.,
            dqydjy = out_nf[7]*2.,
            dqxdjy = out_nf[8]*2.,
            dqydjx = out_nf[9]*2.,
        )
        for nn in dct_nf:
            tw[nn+'_nf_ng'] = dct_nf[nn]

    return tw

xt.Line._tw_ng = _tw_ng
xt.Line._build_madng_model = _build_madng_model

class ActionTwissMadng(xt.Action):
    def __init__(self, line, tw_kwargs):
        self.line = line
        self.tw_kwargs = tw_kwargs

    def run(self):
        return self.line._tw_ng(**self.tw_kwargs)

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

tw = line._tw_ng(normal_form=False)

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
tw_after = line._tw_ng(normal_form=False)
xo.assert_allclose(tw_after['px_ng', 'ip1'], 50e-6, rtol=5e-3, atol=0)

# tw = line._tw_ng()
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

xo.assert_allclose(line._tw_ng()['px_ng', 'ip1'], 1e-6, rtol=5e-3, atol=0)

line['on_x1'] = -2.
xo.assert_allclose(line._tw_ng()['px_ng', 'ip1'], -2e-6, rtol=5e-3, atol=0)

tw_rdt = line._tw_ng(rdts=rdts)

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