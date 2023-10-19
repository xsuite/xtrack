import gc

import pytest
import xobjects as xo


@pytest.fixture(scope="function", autouse=True)
def cleanup(capsys):
    yield

    gc.collect()

    with capsys.disabled():
        no_buffers = sum(b.alive for b in xo.context_default.buffers)
        capacity = sum(b.peek()[0].capacity - b.peek()[0].get_free() for b in xo.context_default.buffers if b.alive)

        print(f"""
*********************************
* Buffers on default ctx: {no_buffers:5} *
* Total capacity: {capacity:13} *
*********************************
""")
