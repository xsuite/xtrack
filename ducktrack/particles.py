# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import xpart as xp
from .mathlibs import MathlibDefault

from .temp_pyparticles import Pyparticles

class TestParticles(Pyparticles):

    _m = MathlibDefault
