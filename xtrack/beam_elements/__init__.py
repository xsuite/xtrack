# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #


from ..base_element import BeamElement
from ..general import DEPRECATION_INFO_PREP_1_0
from typing import List
from numbers import Number
from ..internal_record import RecordIndex
import copy
from scipy.special import factorial
import numpy as np
from warnings import warn
import xobjects as xo
import xtrack as xt
from ..random import (
    RandomExponential,
    RandomNormal,
    RandomUniformAccurate,
)
from ._common import (
    DEFAULT_MULTIPOLE_ORDER as DEFAULT_MULTIPOLE_ORDER,
    ElectronCoolerRecord as ElectronCoolerRecord,
    SynchrotronRadiationRecord as SynchrotronRadiationRecord,
    ThinSliceNotNeededError as ThinSliceNotNeededError,
)

from .splineboris import Spline4, SplineBoris
from .reference_energy_increase import ReferenceEnergyIncrease
from .reference_energy_change import ReferenceEnergyChange
from .marker import Marker
from .drift import Drift
from .drift_exact import DriftExact
from .cavity import Cavity
from .crab_cavity import CrabCavity
from .xy_shift import XYShift
from .translation import Translation
from .elens import Elens
from .nonlinear_lens import NonLinearLens
from .wire import Wire
from .rotation import Rotation
from .s_rotation import SRotation
from .x_rotation import XRotation
from .y_rotation import YRotation
from .zeta_shift import ZetaShift
from .time_delay import TimeDelay
from .misalignment import Misalignment
from .multipole import Multipole
from .simple_thin_quadrupole import SimpleThinQuadrupole
from .bend import Bend
from .rbend import RBend
from .sextupole import Sextupole
from .octupole import Octupole
from .quadrupole import Quadrupole
from .uniform_solenoid import UniformSolenoid
from .variable_solenoid import VariableSolenoid
from .temp_rf import TempRF
from .solenoid import Solenoid
from .magnet import Magnet
from .magnet_edge import MagnetEdge
from .combined_function_magnet import CombinedFunctionMagnet
from .dipole_fringe import DipoleFringe
from .wedge import Wedge
from .simple_thin_bend import SimpleThinBend
from .rf_multipole import RFMultipole
from .dipole_edge import DipoleEdge
from .multipole_edge import MultipoleEdge
from .line_segment_map import LineSegmentMap
from .first_order_taylor_map import FirstOrderTaylorMap
from .second_order_taylor_map import SecondOrderTaylorMap
from .electron_cooler import ElectronCooler
from ._aperture_common import UNLIMITED
from .limit_rect import LimitRect
from .limit_racetrack import LimitRacetrack
from .limit_ellipse import LimitEllipse
from .limit_polygon import LimitPolygon
from .limit_rect_ellipse import LimitRectEllipse
from .longitudinal_limit_rect import LongitudinalLimitRect

from .acdipole import ACDipole
from .exciter import Exciter
from .beam_interaction import BeamInteraction, ParticlesInjectionSample
from .slice_elements_thin import (ThinSliceQuadrupole, ThinSliceSextupole,
                             ThinSliceOctupole, ThinSliceBend,
                             ThinSliceRBend, ThinSliceCavity,
                             ThinSliceCrabCavity, ThinSliceMultipole)
from .slice_elements_edge import (
                             ThinSliceBendEntry, ThinSliceBendExit,
                             ThinSliceRBendEntry, ThinSliceRBendExit,
                             ThinSliceQuadrupoleEntry, ThinSliceQuadrupoleExit,
                             ThinSliceSextupoleEntry, ThinSliceSextupoleExit,
                             ThinSliceOctupoleEntry, ThinSliceOctupoleExit,
                             ThinSliceUniformSolenoidEntry,
                             ThinSliceUniformSolenoidExit)
from .slice_elements_thick import (ThickSliceBend, ThickSliceRBend,
                                   ThickSliceQuadrupole, ThickSliceSextupole,
                                   ThickSliceOctupole, ThickSliceUniformSolenoid,
                                   ThickSliceCavity, ThickSliceCrabCavity,
                                   ThickSliceMultipole)
from .slice_elements_drift import (DriftSliceOctupole, DriftSliceSextupole,
                                   DriftSliceQuadrupole, DriftSliceBend,
                                   DriftSliceRBend, DriftSlice, DriftSliceCavity,
                                   DriftSliceCrabCavity, DriftSliceMultipole,
                                   DriftExactSlice)

from .rft_element import RFT_Element
from ..base_element import BeamElement

element_classes = tuple(v for v in globals().values() if isinstance(v, type) and issubclass(v, BeamElement))


__all__ = tuple(cls.__name__ for cls in element_classes) + (
    'BeamInteraction',
    'CombinedFunctionMagnet',
    'DEFAULT_MULTIPOLE_ORDER',
    'ElectronCoolerRecord',
    'ParticlesInjectionSample',
    'RFT_Element',
    'Spline4',
    'SynchrotronRadiationRecord',
    'ThinSliceNotNeededError',
    'element_classes',
)
