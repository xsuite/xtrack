# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from .bend import Bend

class CombinedFunctionMagnet:

    def __init__(self, *args, **kwargs):
        raise TypeError('`CombinedFunctionMagnet` is supported anymore. '
                        'Use `Bend` instead.')

    @classmethod
    def from_dict(cls, dct):
        return Bend(**dct)
