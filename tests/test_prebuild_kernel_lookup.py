# copyright ################################# #
# This file is part of the Xtrack Package.    #
# Copyright (c) CERN, 2026.                   #
# ########################################### #
import builtins
import json
import types

import pytest

import xobjects as xo
import xobjects.context_cpu as context_cpu
import xtrack as xt
import xsuite.prebuild_kernels as pk


class DummyStruct(xo.Struct):
    value = xo.Float64


class OptOutStruct(xo.Struct):
    allow_no_prebuilt_kernel = True
    value = xo.Float64


def _versions():
    return {
        'xtrack': pk.xt.__version__,
        'xfields': pk.xf.__version__,
        'xcoll': pk.xc.__version__,
        'xobjects': pk.xo.__version__,
    }


def _write_metadata(
    location,
    module_name='test_kernel_cpu_serial',
    base_module_name='test_kernel',
    context='serial',
    config=None,
    versions=None,
    tracker_element_classes=None,
    classes=None,
):
    metadata = {
        'base_module_name': base_module_name,
        'context': context,
        'config': {} if config is None else config,
        'tracker_element_classes': (
            [] if tracker_element_classes is None else tracker_element_classes
        ),
        'classes': [] if classes is None else classes,
        'versions': _versions() if versions is None else versions,
    }

    with (location / f'{module_name}.json').open('w') as fid:
        json.dump(metadata, fid)


@pytest.fixture(autouse=True)
def _restore_no_prebuilt_kernel_flags(monkeypatch):
    monkeypatch.delenv('XSUITE_ALLOW_NO_PREBUILT_KERNELS', raising=False)
    monkeypatch.setattr(context_cpu, 'allow_no_prebuilt_kernel', False)


@pytest.fixture
def kernel_location(monkeypatch, tmp_path):
    monkeypatch.setattr(pk, 'XSK_PREBUILT_KERNELS_LOCATION', tmp_path)
    monkeypatch.setattr(pk, 'kernel_definitions', [('test_kernel', {})])
    return tmp_path


@pytest.fixture
def missing_xsuite(monkeypatch):
    original_import = builtins.__import__

    def import_without_xsuite(name, *args, **kwargs):
        if name == 'xsuite' or name.startswith('xsuite.'):
            raise ImportError('xsuite intentionally unavailable')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', import_without_xsuite)


def _patch_add_kernels(monkeypatch, context):
    calls = []

    def add_kernels(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(context, 'add_kernels', add_kernels)
    return calls


def test_get_suitable_kernel_success(kernel_location):
    module_name = 'test_kernel_cpu_serial'
    _write_metadata(kernel_location, module_name=module_name)
    pk._kernel_binary_file(module_name, kernel_location).touch()

    kernel_info = pk.get_suitable_kernel(
        config={},
        tracker_element_classes=[],
        classes=[],
        context=xo.ContextCpu(),
    )

    assert kernel_info == {
        'module_name': module_name,
        'tracker_element_classes': [],
    }


def test_get_suitable_kernel_raises_when_no_cached_kernels(kernel_location):
    with pytest.raises(pk.PrebuiltKernelNotFoundError) as err:
        pk.get_suitable_kernel({}, [], [], context=xo.ContextCpu())

    assert 'no cached kernels were found' in str(err.value)
    assert 'XSUITE_ALLOW_NO_PREBUILT_KERNELS' in str(err.value)


def test_get_suitable_kernel_raises_when_binary_is_missing(kernel_location):
    _write_metadata(kernel_location)

    with pytest.raises(pk.PrebuiltKernelNotFoundError) as err:
        pk.get_suitable_kernel({}, [], [], context=xo.ContextCpu())

    assert 'no compiled cached kernels were found' in str(err.value)


def test_get_suitable_kernel_raises_on_version_mismatch(kernel_location):
    module_name = 'test_kernel_cpu_serial'
    other_module_name = 'test_kernel_extra_cpu_serial'
    versions = _versions()
    versions['xtrack'] = '0.0.bad'
    _write_metadata(kernel_location, module_name=module_name, versions=versions)
    _write_metadata(
        kernel_location,
        module_name=other_module_name,
        versions=versions,
    )
    pk._kernel_binary_file(module_name, kernel_location).touch()
    pk._kernel_binary_file(other_module_name, kernel_location).touch()

    with pytest.raises(pk.PrebuiltKernelNotFoundError) as err:
        pk.get_suitable_kernel({}, [], [], context=xo.ContextCpu())

    message = str(err.value)
    assert 'package versions do not match' in message
    assert message.count(
        'cached kernels need xtrack==0.0.bad, but the current environment has'
    ) == 1
    assert module_name not in message
    assert other_module_name not in message


def test_get_suitable_kernel_raises_on_config_mismatch(kernel_location):
    module_name = 'test_kernel_cpu_serial'
    farther_module_name = 'test_kernel_farther_cpu_serial'
    _write_metadata(kernel_location, module_name=module_name, config={'a': 1})
    _write_metadata(
        kernel_location,
        module_name=farther_module_name,
        config={'a': 1, 'b': 1},
    )
    pk._kernel_binary_file(module_name, kernel_location).touch()
    pk._kernel_binary_file(farther_module_name, kernel_location).touch()

    with pytest.raises(pk.PrebuiltKernelNotFoundError) as err:
        pk.get_suitable_kernel({'a': 2}, [], [], context=xo.ContextCpu())

    message = str(err.value)
    assert 'no cached kernel matches' in message
    assert 'Closest cached kernel' in message
    assert 'different configuration' in message
    assert module_name in message
    assert farther_module_name not in message


def test_get_suitable_kernel_raises_on_context_mismatch(kernel_location):
    module_name = 'test_kernel_cpu_openmp'
    _write_metadata(
        kernel_location,
        module_name=module_name,
        context='openmp',
    )
    pk._kernel_binary_file(module_name, kernel_location).touch()

    with pytest.raises(pk.PrebuiltKernelNotFoundError) as err:
        pk.get_suitable_kernel({}, [], [], context=xo.ContextCpu())

    assert 'context `openmp`' in str(err.value)


def test_get_suitable_kernel_raises_on_missing_class(kernel_location):
    class Requested:
        pass

    class RequestedStruct:
        _DressingClass = Requested

    module_name = 'test_kernel_cpu_serial'
    _write_metadata(kernel_location, module_name=module_name)
    pk._kernel_binary_file(module_name, kernel_location).touch()

    with pytest.raises(pk.PrebuiltKernelNotFoundError) as err:
        pk.get_suitable_kernel(
            {},
            [],
            [RequestedStruct],
            context=xo.ContextCpu(),
        )

    assert 'missing requested class(es): Requested' in str(err.value)


def test_get_suitable_kernel_returns_none_with_class_opt_out(kernel_location):
    class Requested:
        allow_no_prebuilt_kernel = True

    class RequestedStruct:
        _DressingClass = Requested

    module_name = 'test_kernel_cpu_serial'
    _write_metadata(kernel_location, module_name=module_name)
    pk._kernel_binary_file(module_name, kernel_location).touch()

    assert (
        pk.get_suitable_kernel(
            {},
            [],
            [RequestedStruct],
            context=xo.ContextCpu(),
        )
        is None
    )


def test_closest_kernel_prefers_missing_class_over_config_mismatch(
    kernel_location,
):
    class Requested:
        pass

    class RequestedStruct:
        _DressingClass = Requested

    module_name = 'test_kernel_cpu_serial'
    config_mismatch_module_name = 'test_kernel_config_mismatch_cpu_serial'
    _write_metadata(kernel_location, module_name=module_name)
    _write_metadata(
        kernel_location,
        module_name=config_mismatch_module_name,
        config={'a': 1},
    )
    pk._kernel_binary_file(module_name, kernel_location).touch()
    pk._kernel_binary_file(config_mismatch_module_name, kernel_location).touch()

    with pytest.raises(pk.PrebuiltKernelNotFoundError) as err:
        pk.get_suitable_kernel(
            {},
            [],
            [RequestedStruct],
            context=xo.ContextCpu(),
        )

    message = str(err.value)
    assert 'Closest cached kernel' in message
    assert 'missing requested class(es): Requested' in message
    assert module_name in message
    assert config_mismatch_module_name not in message
    assert 'different configuration' not in message


def test_get_suitable_kernel_returns_none_with_environment_opt_out(
    monkeypatch,
    kernel_location,
):
    monkeypatch.setenv('XSUITE_ALLOW_NO_PREBUILT_KERNELS', '1')

    assert pk.get_suitable_kernel({}, [], [], context=xo.ContextCpu()) is None


def test_get_suitable_kernel_returns_none_with_context_cpu_opt_out(
    monkeypatch,
    kernel_location,
):
    monkeypatch.setattr(context_cpu, 'allow_no_prebuilt_kernel', True)

    assert pk.get_suitable_kernel({}, [], [], context=xo.ContextCpu()) is None


def test_get_suitable_kernel_returns_none_for_openmp_context(kernel_location):
    assert (
        pk.get_suitable_kernel(
            {},
            [],
            [],
            context=xo.ContextCpu(omp_num_threads='auto'),
        )
        is None
    )


def test_missing_xsuite_raises_actionable_error(missing_xsuite):
    context = xo.ContextCpu()

    with pytest.raises(ImportError) as err:
        DummyStruct.compile_class_kernels(context, only_if_needed=True)

    message = str(err.value)
    assert 'pip install xsuite' in message
    assert 'XSUITE_ALLOW_NO_PREBUILT_KERNELS' in message


def test_tracker_missing_xsuite_raises_actionable_error(missing_xsuite):
    tracker = xt.Tracker.__new__(xt.Tracker)
    buffer = type('Buffer', (), {'context': xo.ContextCpu()})()
    tracker_data = type(
        'TrackerData',
        (),
        {'_buffer': buffer, 'line_element_classes': []},
    )()
    tracker._tracker_data_cache = {None: tracker_data}
    tracker.line = type('Line', (), {'config': {}})()
    tracker.use_prebuilt_kernels = True

    with pytest.raises(ImportError) as err:
        tracker._build_kernel(compile=True)

    message = str(err.value)
    assert 'pip install xsuite' in message
    assert 'XSUITE_ALLOW_NO_PREBUILT_KERNELS' in message


def test_missing_xsuite_allows_jit_with_environment_opt_out(
    monkeypatch,
    missing_xsuite,
):
    monkeypatch.setenv('XSUITE_ALLOW_NO_PREBUILT_KERNELS', '1')
    context = xo.ContextCpu()
    add_kernel_calls = _patch_add_kernels(monkeypatch, context)

    DummyStruct.compile_class_kernels(context, only_if_needed=False)

    assert len(add_kernel_calls) == 1


def test_missing_xsuite_allows_jit_with_context_cpu_opt_out(
    monkeypatch,
    missing_xsuite,
):
    monkeypatch.setattr(context_cpu, 'allow_no_prebuilt_kernel', True)
    context = xo.ContextCpu()
    add_kernel_calls = _patch_add_kernels(monkeypatch, context)

    DummyStruct.compile_class_kernels(context, only_if_needed=False)

    assert len(add_kernel_calls) == 1


def test_missing_xsuite_allows_jit_with_class_opt_out(
    missing_xsuite,
    monkeypatch,
):
    context = xo.ContextCpu()
    add_kernel_calls = _patch_add_kernels(monkeypatch, context)

    OptOutStruct.compile_class_kernels(context, only_if_needed=False)

    assert len(add_kernel_calls) == 1


def test_missing_xsuite_allows_jit_for_openmp_context(missing_xsuite, monkeypatch):
    context = xo.ContextCpu(omp_num_threads='auto')
    add_kernel_calls = _patch_add_kernels(monkeypatch, context)

    DummyStruct.compile_class_kernels(context, only_if_needed=False)

    assert len(add_kernel_calls) == 1


def test_tracker_missing_xsuite_allows_jit_with_class_opt_out(
    missing_xsuite,
    monkeypatch,
):
    tracker = xt.Tracker.__new__(xt.Tracker)
    context = xo.ContextCpu()
    out_kernel = object()
    build_kernel_calls = []

    def build_kernels(*args, **kwargs):
        build_kernel_calls.append((args, kwargs))
        return {'track_line': out_kernel}

    monkeypatch.setattr(context, 'build_kernels', build_kernels)
    buffer = type('Buffer', (), {'context': context})()
    tracker_data = type(
        'TrackerData',
        (),
        {
            '_buffer': buffer,
            'line_element_classes': [OptOutStruct],
            'kernel_element_classes': [],
        },
    )()
    tracker._tracker_data_cache = {None: tracker_data}
    tracker.line = type('Line', (), {'config': {}})()
    tracker.use_prebuilt_kernels = True
    tracker.extra_headers = []
    tracker.track_flags = type('TrackFlags', (), {'c_header_flag_mapping': ''})()
    tracker.get_kernel_descriptions = types.MethodType(
        lambda self, kernel_element_classes: {'track_line': xo.Kernel(args=[])},
        tracker,
    )
    tracker._config_to_headers = types.MethodType(lambda self: [], tracker)

    assert tracker._build_kernel(compile=True) is out_kernel
    assert len(build_kernel_calls) == 1
