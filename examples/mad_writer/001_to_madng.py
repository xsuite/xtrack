import numpy as np
import xtrack as xt

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tw = line.twiss(method='4d')

mng = line.to_madng(sequence_name='lhcb1')

mng["mytwtable", 'mytwflow'] = mng.twiss(
    sequence=mng.lhcb1, method=4, mapdef=2, implicit=True, nslice=3, save="'atbody'")

print(mng["mytwtable"].mu1[-1])
assert np.isclose(mng["mytwtable"].mu1[-1][0], 62.31, atol=1e-6, rtol=0)


mng.send('''
    local track in MAD  -- like "from MAD import track"
    local mytrktable, mytrkflow = MAD.track{sequence=MADX.lhcb1, method=4,
                                            mapdef=4, nslice=3}

        -- print(MAD.typeid.is_damap(mytrkflow[1]))

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

out = mng.recv()

print(f'''
    q1: {out[0]}
    q2: {out[1]}
    dq1: {out[2]}
    dq2: {out[3]}
    d2q1: {out[4]}
    d2q2: {out[5]}
    anhx: {out[6]}
    anhy: {out[7]}
    anhx: {out[8]}
    anhy: {out[9]}
''')


