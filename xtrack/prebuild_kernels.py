# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #
import os
from importlib import import_module
from importlib.util import find_spec
from typing import Dict, Hashable, List, Tuple

import xtrack as xt

DEFAULT_CONFIG = {
    'XTRACK_MULTIPOLE_NO_SYNRAD': True,
}

PREBUILT_KERNELS = {
    'xtrack.lib.drifts_and_multipoles': (
        ['Drift', 'Multipole', 'ParticlesMonitor'],
        DEFAULT_CONFIG,
    ),
}


class PrebuiltKernelNotFound(Exception):
    pass


def get_ffi_module_for_configuration(
        desired_element_classes: List[str],
        desired_config: Dict[str, Hashable],
) -> Tuple[str, List[xt.BeamElement]]:
    prebuilt_kernels = {}
    prebuilt_kernels.update(PREBUILT_KERNELS)
    try:
        from xfields.prebuild_kernels import PREBUILT_KERNELS as XF_KERNELS
        prebuilt_kernels.update(XF_KERNELS)
    except ModuleNotFoundError:
        pass

    for module_name, (classes, config) in prebuilt_kernels.items():
        if (set(desired_element_classes) < set(classes)
                and desired_config == config):
            if find_spec(module_name):
                return module_name, classes

    raise PrebuiltKernelNotFound(
        f'There is no prebuilt kernel for classes {desired_element_classes} '
        f'and config {desired_config}.'
    )


def precompile_single_kernel(name, elements, config):
    dummy_tracker = xt.Tracker(
        line=xt.Line(elements),
        compile=False,
    )
    dummy_tracker.config.update(config)
    dummy_tracker._build_kernel(
        compile='force',
        built_ffi_module_name=name,
    )


def precompile_kernels():
    precompile_single_kernel(
        'xtrack.lib.drifts_and_multipoles',
        [xt.Drift(), xt.Multipole()],
        {}
    )
