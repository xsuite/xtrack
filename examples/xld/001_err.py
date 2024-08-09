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


with how_long('reading'):
    p = Parser()
    out = p.parse_file('out_err.xld')
