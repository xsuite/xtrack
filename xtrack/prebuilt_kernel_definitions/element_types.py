# copyright ############################### #
# This file is part of the Xtrack package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..beam_elements import *
from ..monitors import *
from ..random import *
from ..multisetter import MultiSetter

ONLY_XTRACK_ELEMENTS = [
    Drift,
    Multipole,
    Bend,
    RBend,
    Quadrupole,
    Sextupole,
    Octupole,
    Magnet,
    SecondOrderTaylorMap,
    Marker,
    ReferenceEnergyIncrease,
    Cavity,
    CrabCavity,
    Elens,
    Wire,
    Solenoid,
    VariableSolenoid,
    UniformSolenoid,
    RFMultipole,
    DipoleEdge,
    MultipoleEdge,
    SimpleThinBend,
    SimpleThinQuadrupole,
    LineSegmentMap,
    FirstOrderTaylorMap,
    NonLinearLens,
    DriftExact,
    # Drift Slices
    DriftSlice,
    DriftExactSlice,
    DriftSliceBend,
    DriftSliceRBend,
    DriftSliceOctupole,
    DriftSliceQuadrupole,
    DriftSliceSextupole,
    DriftSliceCavity,
    DriftSliceCrabCavity,
    DriftSliceMultipole,
    # Thick slices
    ThickSliceBend,
    ThickSliceRBend,
    ThickSliceOctupole,
    ThickSliceQuadrupole,
    ThickSliceSextupole,
    ThickSliceUniformSolenoid,
    ThickSliceCavity,
    ThickSliceCrabCavity,
    ThickSliceMultipole,
    # Thin slices
    ThinSliceBend,
    ThinSliceRBend,
    ThinSliceOctupole,
    ThinSliceQuadrupole,
    ThinSliceSextupole,
    ThinSliceCavity,
    ThinSliceCrabCavity,
    ThinSliceMultipole,
    # Edge slices
    ThinSliceBendEntry,
    ThinSliceBendExit,
    ThinSliceRBendEntry,
    ThinSliceRBendExit,
    ThinSliceQuadrupoleEntry,
    ThinSliceQuadrupoleExit,
    ThinSliceSextupoleEntry,
    ThinSliceSextupoleExit,
    ThinSliceUniformSolenoidEntry,
    ThinSliceUniformSolenoidExit,
    ThinSliceOctupoleEntry,
    ThinSliceOctupoleExit,

    # Transformations
    XYShift,
    ZetaShift,
    XRotation,
    SRotation,
    YRotation,
    # Apertures
    LimitEllipse,
    LimitRectEllipse,
    LimitRect,
    LimitRacetrack,
    LimitPolygon,
    LongitudinalLimitRect,
    # Monitors
    BeamPositionMonitor,
    BeamSizeMonitor,
    BeamProfileMonitor,
    LastTurnsMonitor,
    ParticlesMonitor,
]

NO_SYNRAD_ELEMENTS = [
    Exciter,
]

NON_TRACKING_ELEMENTS = [
    RandomUniform,
    RandomExponential,
    RandomNormal,
    RandomRutherford,
    MultiSetter,
]
