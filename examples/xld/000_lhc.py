from contextlib import contextmanager

import xtrack as xt
from xtrack.sequence.writer import XldWriter
from xtrack.sequence.parser import Parser
from time import time

@contextmanager
def how_long(what):
    t0 = time()
    yield
    elapsed = time() - t0
    print(f'{what.capitalize()} took {elapsed} s')


with how_long('reading json'):
    line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

writer = XldWriter(line, 'lhcb1')

with how_long('writing'):
    with open('out.xld', 'w') as f:
        writer.write(stream=f)

with how_long('reading'):
    p = Parser()
    out = p.parse_file('out.xld')
