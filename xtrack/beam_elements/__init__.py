# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from .elements import *
from .exciter import Exciter
from .apertures import *
from .magnets import Magnet
from .beam_interaction import BeamInteraction, ParticlesInjectionSample
from .slice_elements import (ThinSliceQuadrupole, ThinSliceSextupole,
                             ThinSliceOctupole, ThinSliceBend,
                             ThinSliceRBend,
                             ThinSliceBendEntry, ThinSliceBendExit,
                             ThinSliceRBendEntry, ThinSliceRBendExit,
                             ThinSliceQuadrupoleEntry, ThinSliceQuadrupoleExit,
                             ThinSliceSextupoleEntry, ThinSliceSextupoleExit,
                             ThinSliceOctupoleEntry, ThinSliceOctupoleExit)
from .slice_elements_thick import (ThickSliceBend, ThickSliceRBend,
                                   ThickSliceQuadrupole, ThickSliceSextupole,
                                   ThickSliceOctupole, ThickSliceSolenoid,
                                   DriftSliceOctupole, DriftSliceSextupole,
                                   DriftSliceQuadrupole, DriftSliceBend,
                                   DriftSliceRBend, DriftSlice)
from .rft_element import RFT_Element
from ..base_element import BeamElement

element_classes = tuple(v for v in globals().values() if isinstance(v, type) and issubclass(v, BeamElement))
