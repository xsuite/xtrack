import numpy as np
import xtrack as xt

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

sequence_name='dummy'
mng = line.to_madng(sequence_name=sequence_name)
mng._sequence_name = sequence_name


class MadngVars:

    def __init__(self, mad):
        self.mad = mad

    def __setitem__(self, key, value):
        setattr(self.mad.MADX, key.replace('.', '_'), value)
        #Expressions still to be handled

mvars = MadngVars(mng)

line.build_tracker()
line.tracker.vars_to_update = [mvars]

line['a'] = 3.
assert mng.MADX.a == 3.


prrrrr

# line._xdeps_vref._owner.mng = mng

rdts = ["f4000", "f3100", "f2020", "f1120", 'f1001']

def _tw_ng(line, rdts=[], tw=None, scalars=True):
    if tw is None:
        tw = line.twiss(method='4d')
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
        local mtbl = twiss {sequence=seq, method=4, mapdef=2, implicit=true, nslice=3}

        -- send columns to Python
        '''
        + send_cmd)

    mng.send(mng_script)

    out = mng.recv()
    out_dct = {k: v for k, v in zip(columns, out)}

    # Add to table
    assert len(out[0]) == len(tw) + 1
    for nn in tw_columns:
        tw[nn+'_ng'] = out_dct[nn][:-1]
    for nn in rdts:
        tw[nn] = out_dct[nn][:-1]

    if scalars:
        mng_script_scalar = (
            f'local seq = MADX.{mng._sequence_name}'
            '''
            local track in MAD  -- like "from MAD import track"
            local mytrktable, mytrkflow = MAD.track{sequence=seq, method=4,
                                                    mapdef=4, nslice=3}

            local normal in MAD.gphys  -- like "from MAD.gphys import normal"
            local my_norm_for = normal(mytrkflow[1]):analyse('anh') -- anh stands for anharmonicity

            local nf = my_norm_for
            py:send({
                    nf:q1{1}, -- qx from the normal form (fractional part)
                    nf:q2{1}, -- qy
                    nf:dq1{1}, -- dqx / d delta
                    nf:dq2{1}, -- dqy / d delta
                    nf:dq1{2}, -- d2 qx / d delta2
                    nf:dq2{2}, -- d2 qy / d delta2
                    nf:anhx{1, 0}, -- dqx / d(2 jx)
                    nf:anhy{0, 1}, -- dqy / d(2 jy)
                    nf:anhx{0, 1}, -- dqx / d(2 jy)
                    nf:anhy{1, 0}, -- dqy / d(2 jx)
                    })
        ''')
        mng.send(mng_script_scalar)
        out_scalar = mng.recv()

        dct_scalar = dict(
            q1 =   out_scalar[0],
            q2 =   out_scalar[1],
            dq1 =  out_scalar[2],
            dq2 =  out_scalar[3],
            d2q1 = out_scalar[4],
            d2q2 = out_scalar[5],
            dqxdjx = out_scalar[6]*2.,
            dqydjy = out_scalar[7]*2.,
            dqxdjy = out_scalar[8]*2.,
            dqydjx = out_scalar[9]*2.,
        )
        for nn in dct_scalar:
            tw[nn+'_nf_ng'] = dct_scalar[nn]

    return tw

xt.Line._tw_ng = _tw_ng

tw = line._tw_ng(rdts=rdts)

# dct = {k: v[:-1] for k, v in zip(colums, out)}
# dct['name'] = tw.name
# tng = xt.Table(dct)

import matplotlib.pyplot as plt
plt.close('all')
plt.figure(1)

plt.plot(tw.s, np.abs(tw.f4000), label='f4000')
plt.plot(tw.s, np.abs(tw.f2020), label='f2020')
plt.plot(tw.s, np.abs(tw.f1120), label='f1120')
plt.plot(tw.s, np.abs(tw.f3100), label='f3100')
plt.xlabel('s [m]')
plt.ylabel(r'|f_{jklm}|')
plt.legend()

plt.show()