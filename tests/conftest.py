import gc
import pytest

import xobjects as xo


@pytest.fixture(scope="function", autouse=True)
def cleanup():
    yield

    gc.collect()

    alive_buffer_count = sum(b.alive for b in xo.context_default.buffers)
    if alive_buffer_count > 0:
        pytest.fail(f"There were {alive_buffer_count} active buffers after a "
                    f"test run, which points to a memory leak during the test "
                    f"session.")
