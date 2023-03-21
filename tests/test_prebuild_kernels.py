# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2022.                   #
# ########################################### #
import json

import cffi

import xpart as xp
import xtrack as xt
from xtrack.prebuild_kernels import regenerate_kernels


def test_prebuild_kernels(mocker, tmp_path):
    # Set up the temporary kernels directory
    metadata = {
        "test_module": {
            "config": {
                "XTRACK_MULTIPOLE_NO_SYNRAD": True,
                "XFIELDS_BB3D_NO_BEAMSTR": True,
            },
            "classes": [
                "Drift",
                "Cavity",
                "XYShift",
            ]
        },
    }

    with (tmp_path /
          xt.prebuild_kernels.XT_PREBUILT_KERNELS_METADATA).open('w+') as fp:
        json.dump(metadata, fp)

    mocker.patch('xtrack.prebuild_kernels.XT_PREBUILT_KERNELS_LOCATION',
                 tmp_path)
    mocker.patch('xtrack.tracker.XT_PREBUILT_KERNELS_LOCATION', tmp_path)

    # Try regenerating the kernels
    regenerate_kernels()

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

    line=xt.Line(elements=[xt.Drift(length=2.0)])
    line.build_tracker()

    p = xp.Particles(p0c=1e9, px=3e-6)
    line.track(p)

    assert p.x == 6e-6
    assert p.y == 0.0
    cffi_compile.assert_not_called()
