# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ...base_element import BeamElement
from ...general import DEPRECATION_INFO_PREP_1_0
from typing import List
from warnings import warn
import xobjects as xo
from ...random import (
    RandomExponential,
    RandomUniformAccurate,
)
from ._common import (
    DEFAULT_MULTIPOLE_ORDER,
    SynchrotronRadiationRecord,
    _HasKnlKsl,
)

class Solenoid(_HasKnlKsl, BeamElement):
    """Solenoid element.

    .. warning:: The Solenoid element is deprecated, use VariableSolenoid or UniformSolenoid instead.

    Parameters
    ----------
    length : float
        Length of the element in meters.
    ks : float
        Strength of the solenoid component in rad / m.
    ksi : float
        Integrated strength of the solenoid component in rad. Only to be
        specified when the element is thin, i.e. when `length` is 0.
    order : int, optional
        Maximum order of multipole expansion for this magnet. Defaults to 5.
    knl : list of floats, optional
        Normal multipole integrated strengths. If not provided, defaults to zeroes.
    ksl : list of floats, optional
        Skew multipole integrated strengths. If not provided, defaults to zeroes.
    num_multipole_kicks : int, optional
        The number of kicks to be used in thin kick splitting. The default value
        of zero implies a single kick in the middle of the element.
    radiation_flag : int, optional
        Whether to enable radiation. See ``Magnet`` for details.
    mult_rot_x_rad : float, optional
        Rotation around the x-axis of the embedded multipolar field, in radians.
    mult_rot_y_rad : float, optional
        Rotation around the y-axis of the embedded multipolar field, in radians.
    mult_shift_x : float, optional
        Offset of the embedded multipolar field along the x-axis, in metres.
    mult_shift_y : float, optional
        Offset of the embedded multipolar field along the y-axis, in metres.
    mult_shift_s : float, optional
        Offset of the embedded multipolar field along s, in metres.
    """
    isthick = True
    has_backtrack = True
    allow_loss_refinement = True

    _xofields = {
        'length': xo.Float64,
        'ks': xo.Float64,
        'ksi': xo.Float64,
        'radiation_flag': xo.Int64,
        'num_multipole_kicks': xo.Int64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'mult_rot_x_rad': xo.Float64,
        'mult_rot_y_rad': xo.Float64,
        'mult_shift_x': xo.Float64,
        'mult_shift_y': xo.Float64,
        'mult_shift_s': xo.Float64,
    }

    _skip_in_to_dict = ['_order', 'inv_factorial_order']  # defined by knl, etc.

    _rename = {
        'order': '_order',
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/legacy_solenoid.h"',
    ]

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _internal_record_class = SynchrotronRadiationRecord

    def __init__(self, order=None, knl: List[float] = None, ksl: List[float] = None, **kwargs):
        warn(
            'The `Solenoid` element is deprecated. Use `VariableSolenoid` or `UniformSolenoid` instead.'
            + DEPRECATION_INFO_PREP_1_0,
            FutureWarning
        )

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if kwargs.get('ksi', 0) != 0:
            # Fail when trying to create a thin solenoid, as these are not
            # tested yet
            raise NotImplementedError('Thin solenoids are not implemented yet.')
            # self.isthick = False

        if kwargs.get('ksi') and kwargs.get('length'):
            raise ValueError(
                "The parameter `ksi` can only be specified when `length` == 0."
            )

        order = order or DEFAULT_MULTIPOLE_ORDER
        multipolar_kwargs = self._prepare_multipolar_params(order, knl=knl, ksl=ksl)
        kwargs.update(multipolar_kwargs)

        self.xoinitialize(**kwargs)
