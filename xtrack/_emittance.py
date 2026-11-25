# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #


def _geometric_emittance(nemitt, gemitt, beta_gamma, plane, default=None):
    """Resolve one plane, preserving the caller's default when omitted."""
    if nemitt is not None and gemitt is not None:
        raise ValueError(f'Cannot provide both nemitt_{plane} and gemitt_{plane}.')
    if nemitt is not None:
        return nemitt / beta_gamma
    return default if gemitt is None else gemitt
