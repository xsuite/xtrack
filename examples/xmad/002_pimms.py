from cpymad.madx import Madx
from xtrack.xmad.writer import XMadWriter
from xtrack.xmad.xmad import Parser
import xtrack as xt

test_data_folder = '../../test_data/'
mad = Madx(stdout=False)

mad.call(test_data_folder + 'pimms/PIMMS.seq')
mad.call(test_data_folder + 'pimms/betatron.str')
mad.beam(particle='proton', gamma=1.21315778) # 200 MeV
mad.use('pimms')
seq = mad.sequence.pimms
def_expr = True

line = xt.Line.from_madx_sequence(seq, deferred_expressions=def_expr)

line.to_json('out_pimms.json')

writer = XMadWriter('pimms', line)

with open('out_pimms.xmad', 'w') as f:
    writer.write(stream=f)

p = Parser()
out = p.parse_file('out_pimms.xmad')
