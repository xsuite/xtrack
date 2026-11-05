# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #


import copy
from numbers import Number
from typing import List
from warnings import warn

import numpy as np
import xobjects as xo
from scipy.special import factorial

import xtrack as xt

from ..base_element import BeamElement
from ..general import DEPRECATION_INFO_PREP_1_0
from ..internal_record import RecordIndex
from ..random import RandomExponential, RandomNormal, RandomUniformAccurate
from ..synctime import SyncTime
from ._aperture_common import UNLIMITED
from ._common import DEFAULT_MULTIPOLE_ORDER as DEFAULT_MULTIPOLE_ORDER
from ._common import ElectronCoolerRecord as ElectronCoolerRecord
from ._common import SynchrotronRadiationRecord as SynchrotronRadiationRecord
from ._common import ThinSliceNotNeededError as ThinSliceNotNeededError
from .acdipole import ACDipole
from .beam_interaction import BeamInteraction, ParticlesInjectionSample
from .bend import Bend
from .bfield_expansion import BFieldExpansion
from .cavity import Cavity
from .combined_function_magnet import CombinedFunctionMagnet
from .crab_cavity import CrabCavity
from .device import Device
from .dipole_edge import DipoleEdge
from .dipole_fringe import DipoleFringe
from .drift import Drift
from .drift_exact import DriftExact
from .electron_cooler import ElectronCooler
from .elens import Elens
from .exciter import Exciter
from .first_order_taylor_map import FirstOrderTaylorMap
from .limit_ellipse import LimitEllipse
from .limit_polygon import LimitPolygon
from .limit_racetrack import LimitRacetrack
from .limit_rect import LimitRect
from .limit_rect_ellipse import LimitRectEllipse
from .line_segment_map import LineSegmentMap
from .longitudinal_limit_rect import LongitudinalLimitRect
from .magnet import Magnet
from .magnet_edge import MagnetEdge
from .marker import Marker
from .misalignment import Misalignment
from .multipole import Multipole
from .multipole_edge import MultipoleEdge
from .nonlinear_lens import NonLinearLens
from .octupole import Octupole
from .quadrupole import Quadrupole
from .rbend import RBend
from .reference_energy_change import ReferenceEnergyChange
from .reference_energy_increase import ReferenceEnergyIncrease
from .rf_multipole import RFMultipole
from .rft_element import RFT_Element
from .rotation import Rotation
from .s_rotation import SRotation
from .second_order_taylor_map import SecondOrderTaylorMap
from .sextupole import Sextupole
from .simple_thin_bend import SimpleThinBend
from .simple_thin_quadrupole import SimpleThinQuadrupole
from .slice_elements_drift import (
    DriftExactSlice,
    DriftSlice,
    DriftSliceBend,
    DriftSliceCavity,
    DriftSliceCrabCavity,
    DriftSliceMultipole,
    DriftSliceOctupole,
    DriftSliceQuadrupole,
    DriftSliceRBend,
    DriftSliceSextupole,
)
from .slice_elements_edge import (
    ThinSliceBendEntry,
    ThinSliceBendExit,
    ThinSliceMultipoleEntry,
    ThinSliceMultipoleExit,
    ThinSliceOctupoleEntry,
    ThinSliceOctupoleExit,
    ThinSliceQuadrupoleEntry,
    ThinSliceQuadrupoleExit,
    ThinSliceRBendEntry,
    ThinSliceRBendExit,
    ThinSliceSextupoleEntry,
    ThinSliceSextupoleExit,
    ThinSliceUniformSolenoidEntry,
    ThinSliceUniformSolenoidExit,
)
from .slice_elements_thick import (
    ThickSliceBend,
    ThickSliceBFieldExpansion,
    ThickSliceCavity,
    ThickSliceCrabCavity,
    ThickSliceDevice,
    ThickSliceMultipole,
    ThickSliceOctupole,
    ThickSliceQuadrupole,
    ThickSliceRBend,
    ThickSliceSextupole,
    ThickSliceUniformSolenoid,
)
from .slice_elements_thin import (
    ThinSliceBend,
    ThinSliceCavity,
    ThinSliceCrabCavity,
    ThinSliceMultipole,
    ThinSliceOctupole,
    ThinSliceQuadrupole,
    ThinSliceRBend,
    ThinSliceSextupole,
)
from .solenoid import Solenoid
from .splineboris import Spline4, SplineBoris
from .temp_rf import TempRF
from .time_delay import TimeDelay
from .translation import Translation
from .uniform_solenoid import UniformSolenoid
from .variable_solenoid import VariableSolenoid
from .wedge import Wedge
from .wire import Wire
from .x_rotation import XRotation
from .xy_shift import XYShift
from .y_rotation import YRotation
from .zeta_shift import ZetaShift

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
