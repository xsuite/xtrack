# copyright ################################# #
# This file is part of the Xobjects Package.  #
# Copyright (c) CERN, 2023.                   #
# ########################################### #
import logging

from xtrack.beam_elements import *

LOGGER = logging.getLogger(__name__)

BASE_CONFIG = {
    'XTRACK_MULTIPOLE_NO_SYNRAD': True,
    'XFIELDS_BB3D_NO_BEAMSTR': True,
    'XFIELDS_BB3D_NO_BHABHA': True,
    'XTRACK_GLOBAL_XY_LIMIT': 1.0,
}

FREEZE_ENERGY = {
    'FREEZE_VAR_delta': True,
    'FREEZE_VAR_ptau': True,
    'FREEZE_VAR_rpp': True,
    'FREEZE_VAR_rvv': True,
}

FREEZE_LONGITUDINAL = {
    **FREEZE_ENERGY,
    'FREEZE_VAR_zeta': True,
}

ONLY_XTRACK_ELEMENTS = [
    Drift,
    Multipole,
    Marker,
    ReferenceEnergyIncrease,
    Cavity,
    XYShift,
    Elens,
    Wire,
    SRotation,
    RFMultipole,
    DipoleEdge,
    SimpleThinBend,
    SimpleThinQuadrupole,
    LineSegmentMap,
    LinearTransferMatrix,
    NonLinearLens,
    LimitEllipse,
    LimitRectEllipse,
    LimitRect,
    LimitRacetrack,
    LimitPolygon
]

NO_SYNRAD_ELEMENTS = [
    Bend,
    Quadrupole,
    Sextupole,
    CombinedFunctionMagnet,
    Solenoid,
]

# These will be enumerated in order of appearance in the dict, so in this case
# (for optimization purposes) the order is important.
kernel_definitions = {
    'default_only_xtrack': {
        'config': BASE_CONFIG,
        'classes': ONLY_XTRACK_ELEMENTS + NO_SYNRAD_ELEMENTS,
    },
    'default_only_xtrack_backtrack': {
        'config': {**BASE_CONFIG, 'XSUITE_BACKTRACK': True},
        'classes': ONLY_XTRACK_ELEMENTS + NO_SYNRAD_ELEMENTS,
    },
    'only_xtrack_frozen_longitudinal': {
        'config': {**BASE_CONFIG, **FREEZE_LONGITUDINAL},
        'classes': ONLY_XTRACK_ELEMENTS + NO_SYNRAD_ELEMENTS,
    },
    'only_xtrack_frozen_energy': {
        'config': {**BASE_CONFIG, **FREEZE_ENERGY},
        'classes': ONLY_XTRACK_ELEMENTS + NO_SYNRAD_ELEMENTS,
    },
    'only_xtrack_backtrack_frozen_energy': {
        'config': {**BASE_CONFIG, **FREEZE_ENERGY, 'XSUITE_BACKTRACK': True},
        'classes': ONLY_XTRACK_ELEMENTS + NO_SYNRAD_ELEMENTS,
    },
    'only_xtrack_taper': {
        'config': {
            **BASE_CONFIG,
            'XTRACK_MULTIPOLE_NO_SYNRAD': False,
            'XTRACK_MULTIPOLE_TAPER': True,
            'XTRACK_DIPOLEEDGE_TAPER': True,
        },
        'classes': ONLY_XTRACK_ELEMENTS,
    },
    'only_xtrack_with_synrad': {
        'config': {**BASE_CONFIG, 'XTRACK_MULTIPOLE_NO_SYNRAD': False},
        'classes': ONLY_XTRACK_ELEMENTS,
    },
    'only_xtrack_with_synrad_kick_as_co': {
        'config': {**BASE_CONFIG, 'XTRACK_MULTIPOLE_NO_SYNRAD': False,
                   'XTRACK_SYNRAD_KICK_SAME_AS_FIRST': True},
        'classes': ONLY_XTRACK_ELEMENTS,
    }
}

try:
    import xfields as xf
    DEFAULT_BB3D_ELEMENTS = [
        *ONLY_XTRACK_ELEMENTS,
        xf.BeamBeamBiGaussian2D,
        xf.BeamBeamBiGaussian3D,
    ]

    kernel_definitions['default_bb3d'] = {
        'config': BASE_CONFIG,
        'classes': [*DEFAULT_BB3D_ELEMENTS, LineSegmentMap],
    }
except ImportError:
    LOGGER.warning('Xfields not installed, skipping BB3D elements')
