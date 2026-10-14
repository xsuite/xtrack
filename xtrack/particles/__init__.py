# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

from warnings import warn

from ..general import DEPRECATION_INFO_PREP_1_0
from .masses import PROTON_MASS_EV, ELECTRON_MASS_EV, MUON_MASS_EV, Pb208_MASS_EV
from .particles import (Particles, reference_from_pdg_id, LAST_INVALID_STATE,
                        _update_kwargs0_from_pdg_id)

_PYHT_DEPRECATION_MSG = (
    'The PyHEADTAIL interface (`enable_pyheadtail_interface()` / '
    '`disable_pyheadtail_interface()`) is deprecated and will be removed in a '
    'future version. Please use the `xwakes` package for wakefields, '
    'impedances and transverse dampers.' + DEPRECATION_INFO_PREP_1_0)


def enable_pyheadtail_interface():
    warn(_PYHT_DEPRECATION_MSG, FutureWarning, stacklevel=2)
    import xpart.pyheadtail_interface.pyhtxtparticles as pp
    import xpart as xp
    import xtrack as xt
    xp.Particles = pp.PyHtXtParticles
    xt.Particles = pp.PyHtXtParticles


def disable_pyheadtail_interface():
    warn(_PYHT_DEPRECATION_MSG, FutureWarning, stacklevel=2)
    import xpart as xp
    import xtrack as xt
    xp.Particles = Particles
    xt.Particles = Particles
