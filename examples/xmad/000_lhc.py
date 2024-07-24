from contextlib import contextmanager

import xtrack as xt
from xtrack.sequence.writer import XMadWriter
from xtrack.sequence.xmad import Parser
from time import time

@contextmanager
def how_long(what):
    t0 = time()
    yield
    elapsed = time() - t0
    print(f'{what.capitalize()} took {elapsed} s')


with how_long('reading json'):
    line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

writer = XMadWriter('lhcb1', line)

with how_long('writing'):
    with open('out.xmad', 'w') as f:
        writer.write(stream=f)

with how_long('reading'):
    p = Parser()
    out = p.parse_file('out.xmad')
