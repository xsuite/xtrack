# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2025.                 #
# ######################################### #

from ..base_element import BeamElement
import xobjects as xo
from ..random import (
    RandomExponential,
    RandomUniformAccurate,
)
from ._common import (
    SynchrotronRadiationRecord,
    _BendCommon,
    _EDGE_MODEL_TO_INDEX,
    _HasKnlKsl,
    _INDEX_TO_EDGE_MODEL,
)

class Magnet(_BendCommon, BeamElement):
    """General transverse field magnet with curvature and fringe fields.

    A beam element representing a magnet with transverse fields, curvature, and
    edge and fringe-field effects. Optional ``integrator`` and ``model``
    parameters can be used to specify the integration scheme and drift model to
    be used in the kick-splitting scheme. Default value is ``adaptive`` for
    both, which aims to provide best results in the general case (``rot-kick-rot``
    using the polar/exact, drift depending on h, for the model, and ``yoshida4``
    for the integration scheme).

    Parameters
    ----------
    length : float, optional
        Length of the element in meters along the reference trajectory.
    k0 : float, optional
        Strength of the horizontal dipolar component in units of m^-1.
    k1 : float, optional
        Strength of the horizontal quadrupolar component in units of m^-2.
    k2 : float, optional
        Strength of the horizontal sextupolar component in units of m^-3.
    k3 : float, optional
        Strength of the horizontal octupolar component in units of m^-4.
    k0s : float, optional
        Strength of the skew dipolar component in units of m^-1.
    k1s : float, optional
        Strength of the skew quadrupolar component in units of m^-2.
    k2s : float, optional
        Strength of the skew sextupolar component in units of m^-3.
    k3s : float, optional
        Strength of the skew octupolar component in units of m^-4.
    h : float, optional
        Curvature of the reference trajectory in units of m^-1 (= 1 / radius).
        Will imply the value of ``k0`` if ``k0_from_h`` is set.
    k0_from_h : bool, optional
        If true, the value of ``k0`` will be pinned to the value of ``h``.
    order : int, optional
        Maximum order of multipole expansion for this magnet. Defaults to 5.
    knl : list of floats, optional
        Normal multipole integrated strengths. If not provided, defaults to zeroes.
    ksl : list of floats, optional
        Skew multipole integrated strengths. If not provided, defaults to zeroes.
    model : str, optional
        Drift model to be used in the kick-splitting scheme. The options are:

            - ``adaptive``: default option, same as ``rot-kick-rot``.
            - ``full``: kept for backward compatibility, same as ``rot-kick-rot``.
            - ``bend-kick-bend``: use a thick (curved, if ``h`` non-zero) exact
                bend map for ``k0``, ``h``, and handle the other strengths in
                the kicks.
            - ``rot-kick-rot-low-order``: use an exact drift map (polar,
                if ``h`` non-zero) and handle all strengths in the kicks.
            - ``rot-kick-rot``: nested integration scheme, alternating: 1. Yoshida-4
                slices with exact drift maps (polar, if ``h`` non-zero) and k0-only
                kicks; 2. kicks for the remaining strengths.
            -   ``rot-kick-rot-high-order``: nested integration scheme, alternating:
                1. Yoshida-6 slices with exact drift maps (polar, if ``h`` non-zero)
                and k0-only kicks; 2. kicks for the remaining strengths.
            - ``mat-kick-mat``: use an expanded combined-function magnet map
                for ``k0``, ``k1``, ``h``, and handle the other strengths in
                the kicks.
            - ``drift-kick-drift-exact``: use an exact drift map with no curvature,
                and handle all strengths in the kicks.
            - ``drift-kick-drift-expanded``: use an expanded drift map with no
                curvature, and handle all strengths in the kicks.

        These will not be applied if the length is zero.
    integrator : str, optional
        Integration scheme to be used. The options are:

            - ``adaptive``: default option, same as ``yoshida4``.
            - ``teapot``: use the Teapot integration scheme.
            - ``yoshida4``: use the Yoshida 4 integration scheme. The number of
                kicks will be implicitly rounded up to the nearest multiple of 7,
                as required by the scheme.
            - ``uniform``: slice uniformly.

        The integration scheme setting will be ignored if the length is zero, or
        if the strength and the curvature settings imply no need for applying
        thin kicks.
    num_multipole_kicks : int, optional
        The number of kicks to be used in thin kick splitting. If zero, and if
        the model selection implies that there are kicks that need to be
        performed, the value will be guessed according to a heuristic: one kick
        in the middle for straight magnets, or ~2 kicks/mrad otherwise.
    edge_entry_active : bool, optional
        Whether to include the edge effect at entry. Enabled by default.
    edge_exit_active : bool, optional
        Whether to include the edge effect at exit. Enabled by default.
    edge_entry_model : str, optional
        Edge model at magnet entry. The options are:

            - ``linear``: use a linear model for the edge.
            - ``full``: include all multipolar terms.
            - ``dipole-only``: ``full`` but includes only the dipolar terms.
            - ``suppressed``: ignore the edge effect.
    edge_exit_model : str, optional
        Edge model at magnet exit. See ``edge_entry_model`` for the options.
    edge_entry_angle : float, optional
        The angle of the entry edge in radians. Default is 0.
    edge_exit_angle : float, optional
        Same as `edge_entry_angle`, but for the exit.
    edge_entry_angle_fdown : float, optional
        Term added to the entry angle only for the ``linear`` mode and only in
        the vertical plane to account for non-zero angle in the closed orbit
        when entering the fringe field (feed down effect). Default is 0.
    edge_exit_angle_fdown : float, optional
        Same as ``edge_entry_angle_fdown``, but for the exit. Default is 0.
    edge_entry_fint: float, optional
        Fringe integral value at entry. Default is 0.
    edge_exit_fint : float, optional
        Same as ``edge_entry_fint``, but for the exit. Default is 0.
    edge_entry_hgap : float, optional
        Equivalent gap at entry in meters. Default is 0.
    edge_exit_hgap : float, optional
        Same as ``edge_entry_hgap``, but for the exit.
    radiation_flag : int, optional
        Flag indicating if synchrotron radiation effects are enabled.
        If zero, no radiation effects are simulated; if 1, the ``mean``
        model is used; if 2, the ``quantum`` model is used and the
        emitted photons are stored in the internal radiation record; if 3,
        the ``quantum-kick`` model is used and only the total radiation kick
        is generated.
    delta_taper : float, optional
        A value added to delta for the purposes of tapering. Default is 0.
    """
    isthick = True
    has_backtrack = True

    _xofields = {
        'length': xo.Float64,
        'order': xo.Int64,
        'inv_factorial_order': xo.Float64,
        'num_multipole_kicks': xo.Int64,
        'knl': xo.Float64[:],
        'ksl': xo.Float64[:],
        'k0': xo.Float64,
        'k1': xo.Float64,
        'k2': xo.Float64,
        'k3': xo.Float64,
        'k0s': xo.Float64,
        'k1s': xo.Float64,
        'k2s': xo.Float64,
        'k3s': xo.Float64,
        'angle': xo.Float64,
        'h': xo.Float64,
        'k0_from_h': xo.Field(xo.UInt64, default=1),
        'edge_entry_active': xo.Field(xo.Int64, default=1),
        'edge_exit_active': xo.Field(xo.Int64, default=1),
        'edge_entry_model': xo.Int64,
        'edge_exit_model': xo.Int64,
        'edge_entry_angle': xo.Float64,
        'edge_exit_angle': xo.Float64,
        'edge_entry_angle_fdown': xo.Float64,
        'edge_exit_angle_fdown': xo.Float64,
        'edge_entry_fint': xo.Float64,
        'edge_exit_fint': xo.Float64,
        'edge_entry_hgap': xo.Float64,
        'edge_exit_hgap': xo.Float64,
        'model': xo.Int64,
        'integrator': xo.Int64,
        'radiation_flag': xo.Int64,
        'delta_taper': xo.Float64,
    }

    _rename = {
        'order': '_order',
        'model': '_model',
        'edge_entry_model': '_edge_entry_model',
        'edge_exit_model': '_edge_exit_model',
        'k0': '_k0',
        'k0_from_h': '_k0_from_h',
        'angle': '_angle',
        'length': '_length',
        'h': '_h',
        'integrator': '_integrator',
    }

    _extra_c_sources = [
        '#include "xtrack/beam_elements/elements_src/magnet.h"',
    ]

    _depends_on = [RandomUniformAccurate, RandomExponential]

    _internal_record_class = SynchrotronRadiationRecord

    def __init__(self, **kwargs):

        if '_xobject' in kwargs and kwargs['_xobject'] is not None:
            self.xoinitialize(**kwargs)
            return

        if 'h' in kwargs:
            raise ValueError("Setting `h` directly is not allowed. "
                                "Set `length` and `angle` instead.")

        to_be_set_with_properties = []
        for nn in ['length', 'angle', 'k0_from_h', 'edge_entry_model',
                   'edge_exit_model', 'k0']:
            if nn in kwargs:
                to_be_set_with_properties.append((nn, kwargs.pop(nn)))

        _HasKnlKsl.__init__(self, **kwargs)

        for nn, val in to_be_set_with_properties:
            setattr(self, nn, val)

    @property
    def edge_entry_model(self):
        return _INDEX_TO_EDGE_MODEL[self._edge_entry_model]

    @edge_entry_model.setter
    def edge_entry_model(self, value):
        try:
            self._edge_entry_model = _EDGE_MODEL_TO_INDEX[value]
        except KeyError:
            raise ValueError(f'Invalid edge model: {value}')

    @property
    def edge_exit_model(self):
        return _INDEX_TO_EDGE_MODEL[self._edge_exit_model]

    @edge_exit_model.setter
    def edge_exit_model(self, value):
        try:
            self._edge_exit_model = _EDGE_MODEL_TO_INDEX[value]
        except KeyError:
            raise ValueError(f'Invalid edge model: {value}')
