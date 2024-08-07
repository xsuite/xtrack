# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from enum import Enum
from pathlib import Path
from xobjects.general import _print  # noqa: F401

_pkg_root = Path(__file__).parent.absolute()


class _LOC:
    def __init__(self, name=None):
        self.name = name

    def __repr__(self):
        return self.name


START = _LOC('START')
END = _LOC('END')
