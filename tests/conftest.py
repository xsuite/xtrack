import contextlib
import gc
import pytest

import xobjects as xo


def assert_context_empty(context):
    """Assert that there are no active buffers in the given context.

    An active buffer is a buffer that has not been freed yet.
    """
    gc.collect()
    alive_buffer_count = len(context._buffers)
    if alive_buffer_count > 0:
        pytest.fail(f"There were {alive_buffer_count} active buffers after a "
                    f"test run, which points to a memory leak during the test "
                    f"session.")


@pytest.fixture(scope="function", autouse=True)
def cleanup(request):
    """Assert that there are no active buffers after each test run. If the test
    fails, then don't bother though.
    """
    tests_failed_before = request.session.testsfailed
    yield
    tests_failed_after = request.session.testsfailed
    test_did_fail = tests_failed_after > tests_failed_before
    if not test_did_fail:
        assert_context_empty(xo.context_default)


@pytest.fixture(scope="function")
def temp_context_default_func(mocker):
    """Temporarily set the default context to a new context in function scope.

    This is useful if a test uses a fixture that allocates buffers in the
    default context, and as a result the buffers are not freed after the test
    run. This context manager can be used to temporarily set the default
    context to a new context, which will have the effect of bypassing the
    default `cleanup` fixture.

    Unfortunately, due to the fact that pytest keeps the value yielded by a
    fixture in a cache longer than the duration of the test, we cannot
    assert_context_empty on the new temporary context here, as it will not
    yet be empty.
    """
    temp_context = xo.ContextCpu()
    mocker.patch.object(xo.typeutils, "context_default", temp_context)
    mocker.patch.object(xo, "context_default", temp_context)
    yield


@pytest.fixture(scope="module")
def temp_context_default_mod(module_mocker):
    """Module scope version of `temp_context_default_func` fixture."""
    module_mocker.patch.object(xo.typeutils, "context_default", xo.ContextCpu())
    yield
