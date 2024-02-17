from pymadng import MAD
import matplotlib.pyplot as plt

from cpymad.madx import Madx
madx = Madx()
madx.call("../../test_data/sps_thick/sps.seq")
madx.call("../../test_data/sps_thick/lhc_q20.str")
madx.beam()
madx.use("sps")
twmadx = madx.twiss()

mad = MAD()

# with MAD() as mad:
if True:
    mad.MADX.load('"../../test_data/sps_thick/sps.seq"', "'sps.mad'")
    mad.MADX.load(f"'../../test_data/sps_thick/lhc_q20.str'", f"'lhc_q20.mad'")
    mad["sps"] = mad.MADX.sps
    mad.sps.beam = mad.beam()
    mad["mytwtable", 'mytwflow'] = mad.twiss(sequence=mad.sps, method=4, mapdef=2, implicit=True, nslice=3, save="'atbody'")

    dq1 = mad.mytwtable.dq1
    # plt.plot(mad.mytwtable.s, mad.mytwtable.beta11)

    mad.send('''
        local track in MAD  -- like "from MAD import track"
        local mytrktable, mytrkflow = MAD.track{sequence=MADX.sps, method=4,
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

    out = mad.recv()

    # mad["mytrktable", "mytrkflow"] = mad.track(sequence=mad.sps, method=4)
    # mad.gphys.normal(mad["mytrkflow"][0])



# plt.show()

# doc
# mad.show(mad.twiss)