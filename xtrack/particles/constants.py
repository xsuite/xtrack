# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import scipy.constants as sc

PROTON_MASS_EV = sc.m_p *sc.c**2 /sc.e
ELECTRON_MASS_EV = sc.m_e * sc.c**2 /sc.e
MUON_MASS_EV = sc.physical_constants['muon mass'][0] * sc.c**2 /sc.e
Pb208_MASS_EV = 193729024900.
U_MASS_EV = 931494102.42
