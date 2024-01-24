import numpy as np
from cpymad.madx import Madx
from pymadng import MAD
import xtrack as xt
import uuid

madx1 = Madx()
madx1.call("../../test_data/hllhc15_thick/lhc.seq")
madx1.call("../../test_data/hllhc15_thick/hllhc_sequence.madx")
madx1.call("../../test_data/hllhc15_thick/opt_round_150_1500.madx")
madx1.beam(particle='proton', energy=7000e9)
madx1.use("lhcb1")
madx1.input('save, sequence=lhcb1, file="formadn.seq";')

mng1 = MAD()
mng1.MADX.load('"formadn.seq"', f"'mad1.madng'")
mng1["lhcb1"] = mng1.MADX.lhcb1
mng1.lhcb1.beam = mng1.beam()
mng1["mytwtable", 'mytwflow'] = mng1.twiss(
    sequence=mng1.lhcb1, method=4, mapdef=2, implicit=True, nslice=3, save="'atbody'")


line = xt.Line.from_madx_sequence(sequence=madx1.sequence.lhcb1,
                                    deferred_expressions=True)
line.particle_ref = xt.Particles(p0c=7000e9, mass0=xt.PROTON_MASS_EV)

tw = line.twiss(method='4d')

mng2 = line.to_madng(sequence_name='lhcb1')

mng2["lhcb1"] = mng2.MADX.lhcb1
mng2.lhcb1.beam = mng2.beam(particle="'proton'", energy=7000)
mng2["mytwtable", 'mytwflow'] = mng2.twiss(
    sequence=mng2.lhcb1, method=4, mapdef=2, implicit=True, nslice=3, save="'atbody'")

print(mng2["mytwtable"].mu1[-1])
assert np.isclose(mng2["mytwtable"].mu1[-1][0], 62.31, atol=1e-6, rtol=0)


mng2.send('''
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
            nf:anhx{1, 0}, -- dqx / djx
            nf:anhy{0, 1}, -- dqy / djy
            nf:anhx{0, 1}, -- dqx / djy
            nf:anhy{1, 0}, -- dqy / djx
            })
''')

out = mng2.recv()

madx2 = Madx()
madx2.call("xsuite_to_mad.madx")
madx2.beam(particle='proton', energy=7000)
madx2.use("lhcb1")
tmx2 = madx2.twiss()

dp = 1e-5
twp = line.twiss(method='4d', delta0=dp)
twm = line.twiss(method='4d', delta0=-dp)

ddqx = (twp.dqx - twm.dqx) / (2*dp)


