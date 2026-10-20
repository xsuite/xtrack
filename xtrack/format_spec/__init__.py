# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

"""Stable, versioned TFS output presets.

Pass a preset to ``table.to_tfs(file, format_spec=...)``. Presets are immutable;
``load(preset)`` returns a mutable copy for customization or JSON serialization.
``load`` also accepts user dictionaries and JSON/TOML paths.
"""

from importlib.resources import files
from types import MappingProxyType
import tomllib

from .._tfs_format import load_format_spec as load


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _preset(name):
    with files(__package__).joinpath(name + '.toml').open('rb') as fh:
        return _freeze(load(tomllib.load(fh)))


optics_orbit_v1 = _preset('optics_orbit_v1')
optics_orbit_strengths_v1 = _preset('optics_orbit_strengths_v1')
madx_twiss_v1 = _preset('madx_twiss_v1')

__all__ = ['load', 'optics_orbit_v1', 'optics_orbit_strengths_v1', 'madx_twiss_v1']
