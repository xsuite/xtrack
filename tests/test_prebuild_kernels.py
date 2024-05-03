# copyright ################################# #
# This file is part of the Xtrack Package.    #
# Copyright (c) CERN, 2024.                   #
# ########################################### #
import os

import cffi
import pytest

import xobjects as xo
import xtrack as xt


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
                xt.XYShift,
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
                xt.Cavity,
                xt.XYShift,
                xt.Bend,
            ],
        }),
    ]

    # Override the definitions with the temporary ones
    mocker.patch('xsuite.kernel_definitions.kernel_definitions', kernel_defs)
    mocker.patch('xsuite.prebuild_kernels.kernel_definitions', kernel_defs)
    # We need to change the default location so that loading the kernels works
    mocker.patch('xsuite.prebuild_kernels.XSK_PREBUILT_KERNELS_LOCATION',
                 tmp_path)
    mocker.patch('xsuite.XSK_PREBUILT_KERNELS_LOCATION',
                 tmp_path)

    # Try regenerating the kernels
    from xsuite.prebuild_kernels import regenerate_kernels
    regenerate_kernels(kernels=['111_test_module', '000_test_module'], location=tmp_path)

    # Check if the expected files were created
    so_file0, = tmp_path.glob('000_test_module.*.so')
    assert so_file0.exists()
    assert (tmp_path / '000_test_module.c').exists()
    assert (tmp_path / '000_test_module.json').exists()

    so_file1, = tmp_path.glob('111_test_module.*.so')
    assert so_file1.exists()
    assert (tmp_path / '111_test_module.c').exists()
    assert (tmp_path / '111_test_module.json').exists()

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
    assert 'Found suitable prebuilt kernel `111_test_module`' in captured.out


def test_per_element_prebuild_kernels(mocker, tmp_path, temp_context_default_func):
    # Set up the temporary kernels directory
    kernel_defs = [
        ("test_module", {
            "config": {},
            "classes": [
                xt.Drift,
                xt.Cavity,
                xt.XYShift,
            ]
        }),
        ("test_module_rand", {
            "config": {},
            "classes": [],
            "extra_classes": [
                xt.RandomNormal,
            ]
        }),
    ]

    # Override the definitions with the temporary ones
    mocker.patch('xsuite.kernel_definitions.kernel_definitions', kernel_defs)
    mocker.patch('xsuite.prebuild_kernels.kernel_definitions', kernel_defs)
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
        if not path.name.startswith('test_module.'):
            continue
        if path.suffix not in ('.so', '.dll', '.dylib', '.pyd'):
            continue
        so_file_exists = True
    assert so_file_exists

    assert (tmp_path / 'test_module.c').exists()
    assert (tmp_path / 'test_module.json').exists()

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
