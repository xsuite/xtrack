# copyright ################################# #
# This file is part of the Xtrack Package.    #
# Copyright (c) CERN, 2024.                   #
# ########################################### #
import json
import os

import cffi
import pytest

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import skip_if_forbid_compile


@pytest.fixture
def with_verbose():
    old_verbose = os.environ.get('XSUITE_VERBOSE', None)
    os.environ['XSUITE_VERBOSE'] = '1'

    yield

    if old_verbose is None:
        del os.environ['XSUITE_VERBOSE']
    else:
        os.environ['XSUITE_VERBOSE'] = old_verbose


def test_prebuild_kernels(mocker, tmp_path, temp_context_default_func, capsys, with_verbose):

    skip_if_forbid_compile()

    # Set up the temporary kernels directory
    kernel_defs = [
        ("111_test_module", {
            "config": {
                "XTRACK_MULTIPOLE_NO_SYNRAD": True,
                "XTRACK_GLOBAL_XY_LIMIT": 1.0,
                "XFIELDS_BB3D_NO_BEAMSTR": True,
                "XFIELDS_BB3D_NO_BHABHA": True,
            },
            "classes": [
                xt.Drift,
                xt.Cavity,
                xt.Translation,
                xt.DriftSlice,
                xt.DriftSliceCavity,
                xt.MultiElementMonitor,
                xt.ParticlesMonitor,
                xt.ThickSliceCavity,
                xt.ThinSliceCavity,
            ],
        }),
        ("000_test_module", {
            "config": {
                "XTRACK_MULTIPOLE_NO_SYNRAD": True,
                "XTRACK_GLOBAL_XY_LIMIT": 1.0,
                "XFIELDS_BB3D_NO_BEAMSTR": True,
                "XFIELDS_BB3D_NO_BHABHA": True,
            },
            "classes": [
                xt.Drift,
                xt.Translation,
                xt.DriftSlice,
                xt.MultiElementMonitor,
                xt.ParticlesMonitor,
                xt.ThickSliceCavity,
                xt.ThinSliceCavity,
            ],
        }),
    ]

    all_classes = [xt.Cavity, xt.Drift, xt.DriftSlice, xt.DriftSliceCavity,
                   xt.MultiElementMonitor, xt.ParticlesMonitor, xt.ThickSliceCavity,
                   xt.ThinSliceCavity, xt.Translation, xt.Particles, xt.RandomNormal]
    NAME_CLASS_MAP = {cls.__name__: cls for cls in all_classes}


    # Override the definitions with the temporary ones
    mocker.patch('xsuite.kernel_definitions.kernel_definitions', kernel_defs)
    mocker.patch('xsuite.prebuild_kernels.kernel_definitions', kernel_defs)
    # Override the NAME_CLASS_MAP definition as well
    mocker.patch('xsuite.kernel_definitions.NAME_CLASS_MAP', NAME_CLASS_MAP)
    mocker.patch('xsuite.prebuild_kernels.NAME_CLASS_MAP', NAME_CLASS_MAP)
    # We need to change the default location so that loading the kernels works
    mocker.patch('xsuite.prebuild_kernels.XSK_PREBUILT_KERNELS_LOCATION',
                 tmp_path)
    mocker.patch('xsuite.XSK_PREBUILT_KERNELS_LOCATION',
                 tmp_path)

    # Try regenerating the kernels
    from xsuite.prebuild_kernels import regenerate_kernels
    regenerate_kernels(kernels=['111_test_module', '000_test_module'], location=tmp_path)

    # Check if the expected files were created
    so_file0, = tmp_path.glob('000_test_module_cpu_serial.*.so')
    assert so_file0.exists()
    assert (tmp_path / '000_test_module_cpu_serial.c').exists()
    assert (tmp_path / '000_test_module_cpu_serial.json').exists()

    so_file1, = tmp_path.glob('111_test_module_cpu_serial.*.so')
    assert so_file1.exists()
    assert (tmp_path / '111_test_module_cpu_serial.c').exists()
    assert (tmp_path / '111_test_module_cpu_serial.json').exists()

    # Test that reloading the kernel works, and it's the right one
    cffi_compile = mocker.patch.object(cffi.FFI, 'compile')

    line = xt.Line(elements=[xt.Drift(length=2.0)])

    # Build the tracker on a fresh context, so that the kernel comes from a file
    line.build_tracker(_context=xo.ContextCpu())

    p = xt.Particles(p0c=1e9, px=3e-6)
    line.track(p)

    assert p.x == 6e-6
    assert p.y == 0.0
    cffi_compile.assert_not_called()

    captured = capsys.readouterr()
    assert 'Found suitable prebuilt kernel `111_test_module_cpu_serial`' in captured.out

def test_per_element_prebuild_kernels(mocker, tmp_path, temp_context_default_func):

    skip_if_forbid_compile()

    # Set up the temporary kernels directory
    kernel_defs = [
        ("test_module", {
            "config": {},
            "classes": [
                xt.Drift,
                xt.Cavity,
                xt.Translation,
                xt.DriftSlice,
                xt.DriftSliceCavity,
                xt.MultiElementMonitor,
                xt.ParticlesMonitor,
                xt.ThickSliceCavity,
                xt.ThinSliceCavity,
            ],
            'extra_classes': [xt.Particles]
        }),
        ("test_module_rand", {
            "config": {},
            "classes": [],
            "extra_classes": [
                xt.RandomNormal,
                xt.Particles,
            ],
        }),
    ]

    all_classes = [xt.Cavity, xt.Drift, xt.DriftSlice, xt.DriftSliceCavity,
                   xt.MultiElementMonitor, xt.ParticlesMonitor, xt.ThickSliceCavity,
                   xt.ThinSliceCavity, xt.Translation, xt.Particles, xt.RandomNormal]
    NAME_CLASS_MAP = {cls.__name__: cls for cls in all_classes}

    # Override the definitions with the temporary ones
    mocker.patch('xsuite.kernel_definitions.kernel_definitions', kernel_defs)
    mocker.patch('xsuite.prebuild_kernels.kernel_definitions', kernel_defs)
    # Override the NAME_CLASS_MAP definition as well
    mocker.patch('xsuite.kernel_definitions.NAME_CLASS_MAP', NAME_CLASS_MAP)
    mocker.patch('xsuite.prebuild_kernels.NAME_CLASS_MAP', NAME_CLASS_MAP)
    # We need to change the default location so that loading the kernels works
    mocker.patch('xsuite.prebuild_kernels.XSK_PREBUILT_KERNELS_LOCATION',
                 tmp_path)
    mocker.patch('xsuite.XSK_PREBUILT_KERNELS_LOCATION',
                 tmp_path)

    # Try regenerating the kernels
    from xsuite.prebuild_kernels import regenerate_kernels
    regenerate_kernels(kernels=['test_module', 'test_module_rand'], location=tmp_path)

    # Check if the expected files were created
    so_file_exists = False
    for path in tmp_path.iterdir():
        if not path.name.startswith('test_module_cpu_serial.'):
            continue
        if path.suffix not in ('.so', '.dll', '.dylib', '.pyd'):
            continue
        so_file_exists = True
    assert so_file_exists

    assert (tmp_path / 'test_module_cpu_serial.c').exists()
    assert (tmp_path / 'test_module_cpu_serial.json').exists()

    # Test that reloading the kernel works
    cffi_compile = mocker.patch.object(cffi.FFI, 'compile')

    drift = xt.Drift(length=2.0)

    p = xt.Particles(p0c=1e9, px=3e-6)
    drift.track(p)

    assert p.x == 6e-6
    assert p.y == 0.0

    rng = xt.RandomNormal()
    n_samples = 100
    samples = rng.generate(n_samples=n_samples, n_seeds=n_samples)
    assert len(samples) == n_samples

    cffi_compile.assert_not_called()


def test_context_specific_prebuilt_kernel_selection(mocker, tmp_path):

    kernel_defs = [
        ('test_module', {'config': {}, 'classes': [xt.Drift], 'extra_classes': [xt.Particles]}),
    ]
    drift_tracker_class = xt.Drift._XoStruct
    name_class_map = {
        drift_tracker_class._DressingClass.__name__: drift_tracker_class,
        xt.Particles.__name__: xt.Particles,
    }

    mocker.patch('xsuite.kernel_definitions.kernel_definitions', kernel_defs)
    mocker.patch('xsuite.prebuild_kernels.kernel_definitions', kernel_defs)
    mocker.patch('xsuite.kernel_definitions.NAME_CLASS_MAP', name_class_map)
    mocker.patch('xsuite.prebuild_kernels.NAME_CLASS_MAP', name_class_map)
    mocker.patch('xsuite.prebuild_kernels.XSK_PREBUILT_KERNELS_LOCATION', tmp_path)
    mocker.patch('xsuite.XSK_PREBUILT_KERNELS_LOCATION', tmp_path)

    versions = {
        'xtrack': xt.__version__,
        'xfields': __import__('xfields').__version__,
        'xcoll': __import__('xcoll').__version__,
        'xobjects': xo.__version__,
    }
    metadata_template = {
        'base_module_name': 'test_module',
        'config': {},
        'tracker_element_classes': [drift_tracker_class._DressingClass.__name__],
        'classes': ['Particles'],
        'versions': versions,
    }

    with (tmp_path / 'test_module_cpu_serial.json').open('w') as fd:
        json.dump({**metadata_template, 'context': 'serial'}, fd)
    with (tmp_path / 'test_module_cpu_openmp.json').open('w') as fd:
        json.dump({**metadata_template, 'context': 'omp'}, fd)

    from xsuite.prebuild_kernels import get_suitable_kernel

    serial_info = get_suitable_kernel(
        config={},
        tracker_element_classes=[drift_tracker_class],
        classes=[xt.Particles],
        context=xo.ContextCpu(),
    )
    omp_info = get_suitable_kernel(
        config={},
        tracker_element_classes=[drift_tracker_class],
        classes=[xt.Particles],
        context=xo.ContextCpu(omp_num_threads='auto'),
    )

    assert serial_info['module_name'] == 'test_module_cpu_serial'
    assert omp_info['module_name'] == 'test_module_cpu_openmp'


def test_clear_kernels_preserves_other_context(tmp_path):

    for filename in (
        'test_module_cpu_serial.json',
        'test_module_cpu_serial.c',
        'test_module_cpu_serial.cpython-311-darwin.so',
        'test_module_cpu_openmp.json',
        'test_module_cpu_openmp.c',
        'test_module_cpu_openmp.cpython-311-darwin.so',
    ):
        (tmp_path / filename).write_text('')

    from xsuite.prebuild_kernels import clear_kernels

    clear_kernels(location=tmp_path, context='omp')

    assert (tmp_path / 'test_module_cpu_serial.json').exists()
    assert (tmp_path / 'test_module_cpu_serial.c').exists()
    assert (tmp_path / 'test_module_cpu_serial.cpython-311-darwin.so').exists()
    assert not (tmp_path / 'test_module_cpu_openmp.json').exists()
    assert not (tmp_path / 'test_module_cpu_openmp.c').exists()
    assert not (tmp_path / 'test_module_cpu_openmp.cpython-311-darwin.so').exists()
