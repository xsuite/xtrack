# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

from .masses import PROTON_MASS_EV, ELECTRON_MASS_EV, MUON_MASS_EV, Pb208_MASS_EV
from .particles import Particles, reference_from_pdg_id, LAST_INVALID_STATE


def enable_pyheadtail_interface():
    import xpart.pyheadtail_interface.pyhtxtparticles as pp
    import xpart as xp
    import xtrack as xt
    xp.Particles = pp.PyHtXtParticles
    xt.Particles = pp.PyHtXtParticles


def disable_pyheadtail_interface():
    import xpart as xp
    import xtrack as xt
    xp.Particles = Particles
    xt.Particles = Particles
