import numpy as np
from cpymad.madx import Madx
import xtrack as xt
import os

madx1 = Madx()
madx1.call("../../test_data/hllhc15_thick/lhc.seq")
madx1.call("../../test_data/hllhc15_thick/hllhc_sequence.madx")
madx1.call("../../test_data/hllhc15_thick/opt_round_150_1500.madx")
madx1.beam(particle='proton', energy=7000e9)
madx1.use("lhcb1")
madx1.input('save, sequence=lhcb1, file="formadn.seq";')

from pymadng import MAD
mad = MAD()
mad.MADX.load('"formadn.seq"', f"'mad1.madng'")
mad["lhcb1"] = mad.MADX.lhcb1
mad.lhcb1.beam = mad.beam()
mad["mytwtable", 'mytwflow'] = mad.twiss(
    sequence=mad.lhcb1, method=4, mapdef=2, implicit=True, nslice=3, save="'atbody'")

prrrrr

# tt = line.get_table()
# for nn in tt.rows[tt.element_type=='SRotation'].name:
#     tt.angle = 0

# mad_seq = line.to_madx_sequence(sequence_name='myseq')

# mad2 = Madx()
# mad2.input(mad_seq)
# mad2.beam()
# mad2.use('myseq')

# temp_fname = 'temp4madng'
# with open(temp_fname+'.madx', 'w') as fid:
#     fid.write(mad_seq)

# # MAD-NG stuff




# mad.send('''
#         local track in MAD  -- like "from MAD import track"
#         local mytrktable, mytrkflow = MAD.track{sequence=MADX.myseq, method=4,
#                                                 mapdef=4, nslice=3}

#          -- print(MAD.typeid.is_damap(mytrkflow[1]))

#         local normal in MAD.gphys  -- like "from MAD.gphys import normal"
#         local my_norm_for = normal(mytrkflow[1]):analyse() -- 'anh') -- anh stands for anharmonicity

#         local nf = my_norm_for
#         py:send({
#                 nf:q1{1}, -- qx from the normal form (fractional part)
#                 nf:q2{1}, -- qy
#                 nf:dq1{1}, -- dqx / d delta
#                 nf:dq2{1}, -- dqy / d delta
#                 nf:dq1{2}, -- d2 qx / d delta2
#                 nf:dq2{2}, -- d2 qy / d delta2
#                  -- nf:anhx{1, 0}, -- dqx / djx
#                  -- nf:anhy{0, 1}, -- dqy / djy
#                  -- nf:anhx{0, 1}, -- dqx / djy
#                  -- nf:anhy{1, 0}, -- dqy / djx
#              })
#     ''')

# out = mad.recv()


