# copyright ################################# #
# This file is part of the Xtrack Package.    #
# Copyright (c) CERN, 2026.                   #
# ########################################### #
import builtins
import json

import pytest

import xobjects as xo
import xobjects.context_cpu as context_cpu
import xsuite.prebuild_kernels as pk


class DummyStruct(xo.Struct):
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
    versions = _versions()
    versions['xtrack'] = '0.0.bad'
    _write_metadata(kernel_location, module_name=module_name, versions=versions)
    pk._kernel_binary_file(module_name, kernel_location).touch()

    with pytest.raises(pk.PrebuiltKernelNotFoundError) as err:
        pk.get_suitable_kernel({}, [], [], context=xo.ContextCpu())

    message = str(err.value)
    assert 'package versions do not match' in message
    assert 'xtrack==0.0.bad' in message


def test_get_suitable_kernel_raises_on_config_mismatch(kernel_location):
    module_name = 'test_kernel_cpu_serial'
    _write_metadata(kernel_location, module_name=module_name, config={'a': 1})
    pk._kernel_binary_file(module_name, kernel_location).touch()

    with pytest.raises(pk.PrebuiltKernelNotFoundError) as err:
        pk.get_suitable_kernel({'a': 2}, [], [], context=xo.ContextCpu())

    message = str(err.value)
    assert 'no cached kernel matches' in message
    assert 'different configuration' in message


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


def test_missing_xsuite_raises_actionable_error(missing_xsuite):
    context = xo.ContextCpu()

    with pytest.raises(ImportError) as err:
        DummyStruct.compile_class_kernels(context, only_if_needed=True)

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
