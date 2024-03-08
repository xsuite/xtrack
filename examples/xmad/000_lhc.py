from contextlib import contextmanager

import xtrack as xt
from xtrack.xmad.writer import XMadWriter
from xtrack.xmad.xmad import parse_file
from time import time

@contextmanager
def how_long(what):
    t0 = time()
    yield
    elapsed = time() - t0
    print(f'{what.capitalize()} took {elapsed} s')


line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

writer = XMadWriter('lhcb1', line)

with how_long('writing'):
    with open('out.xmad', 'w') as f:
        writer.write(stream=f)

with how_long('reading'):
    out = parse_file('out.xmad')
