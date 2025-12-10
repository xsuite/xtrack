import numpy as np
import xobjects as xo

import xtrack as xt


class ACDipole(xt.BeamElement):
    """
    ACDipole is a thin element that applies an oscillating kick to the beam in the x or y direction.
    It is used for beam excitations in circular machines for optics measurements. The kick is
    a sinusoidal function defined by a voltage amplitude (with ramped up and down), a fixed frequency
    (small compared to the revolution frequency) and fixed phase lag.

    If the dipole is not in twiss mode (typically used for tracking simulations):
        The kick is applied as a function of the turn number, and it can be ramped up and down
        to avoid emittance growth. The transverse momentum in the vertical plane is changed by
        `(0.3 * volt/p0c) * sin(2π * freq * turn + lag)`.

    If the dipole is in twiss mode:
        It approximates the effect of an AC dipole, simulating it as a thin gradient error
        see (Miyamoto, R., Kopp, S., Jansson, A., & Syphers, M. (2008). Parametrization of
        the driven betatron oscillation. Phys. Rev. ST Accel. Beams, 11, 084002) for more details.
        It applies a beta and tune shift to the beam in the horizontal plane,
        depending on the natural and driven tunes.

        If any of the parameters `natural_q` or `beta_at_acdipole` are not provided,
        the effective gradient (`eff_grad`) during twiss mode is set to zero, meaning
        it will have no effect on the twiss computation.

    Parameters
    ----------
    volt : float | None
        The voltages applied to control the peak of the kick in tracking mode.
        If `None`, no kick is applied.
    freq : float | None
        The driven frequency of the AC dipole, in units of 2π per turn. This is
        equivalent to the fractional driven tune. If `None`, freq is set to zero.
        Note that freq must be small compared to the revolution frequency.
        _This is the only parameter that is used in _both_ tracking and twiss modes._
    lag : float | None
        The phase lag of the AC dipole, in units of radians. This is only used in
        tracking mode and shifts the phase of the sinusoidal kick.
        If `None`, lag is set to zero.
    ramp : list of int
        The ramp settings for the AC dipole, defining the turns for ramping up and
        down the kick in tracking mode.
        The list should contain four integers: [ramp1, ramp2, ramp3, ramp4].
        - `ramp1`: Starting turn of amplitude ramp-up.
        - `ramp2`: Last turn of amplitude ramp-up.
        - `ramp3`: Starting turn of amplitude ramp-down.
        - `ramp4`: Last turn of amplitude ramp-down.
        If not provided, no kick is applied in tracking mode.
    plane : str | None
        The plane in which the AC dipole acts, either `'h'` or `'v'` (lowercase).
        If `None`, the ACDipole is turned off and has no effect in twiss or
        tracking simulations.
    twiss_mode : bool | None
        If `True`, the element is in twiss mode, and the effective gradient is computed
        from `freq`, `natural_q` and `beta_at_acdipole`. If `None` or `False`, the element is in
        tracking mode and applies kicks based on `volt`, `freq`, `lag`, and `ramp`.
    beta_at_acdipole : float | None
        The beta function at the location of the AC dipole, in meters. This is only
        required if the element is in twiss mode. If not provided, the effective
        gradient is set to zero.
    natural_q : float | None
        The natural tune of the machine in the specified plane.
        This is only required if the element is in twiss mode.
        If not provided, the effective gradient is set
        to zero.

    """

    _xofields = {
        # Voltage defines the strength of the kick
        "volt": xo.Float64,
        # Frequency defines strength depending on delta to tune
        "freq": xo.Float64,
        # Lag
        "lag": xo.Float64,
        # Ramping parameters
        "ramp": xo.UInt32[4],
        # kick plane
        "plane": xo.UInt8,
        # Twiss mode flag
        "twiss_mode": xo.UInt64,
        # Effective gradient
        "eff_grad": xo.Float64,
    }

    _extra_c_sources = [
        "#include <beam_elements/elements_src/acdipole.h>",
    ]

    _plane_to_int = {None: 0, "h": 1, "v": 2, "x": 1, "y": 2}
    _int_to_plane = {0: None, 1: "h", 2: "v"}

    _rename = {
        "freq": "_freq",
        "plane": "_plane",
        "twiss_mode": "_twiss_mode",
    }

    def __init__(
        self,
        *,
        volt=None,
        freq=None,
        lag=None,
        ramp=None,
        plane=None,
        twiss_mode=None,
        beta_at_acdipole=None,
        natural_q=None,
        _xobject=None,
        **kwargs,
    ):
        if _xobject is not None:
            super().__init__(_xobject=_xobject)
            return
        # The default is needed, as we wish to be able to instantiate
        # elements without properties (empty elements can be templates).
        if volt is None:
            volt = 0
        if freq is None:
            freq = 0
        if lag is None:
            lag = 0
        if ramp is None:
            ramp = [0, 0, 0, 0]
        if twiss_mode is None:
            twiss_mode = 0
        if beta_at_acdipole is None:
            beta_at_acdipole = 0
        if natural_q is None:
            natural_q = 0

        if not (
            len(ramp) == 4
            and all(v == int(v) and v >= 0 for v in ramp)
        ):
            raise ValueError(
                "The ramp parameter must be a sequence of four positive integers:"
                "[ramp_up_start_turn, ramp_up_end_turn, ramp_down_start_turn, ramp_down_end_turn]."
            )

        super().__init__(
            volt=volt,
            freq=freq,
            lag=lag,
            ramp=ramp,
            plane=0,  # Temporary placeholder
            twiss_mode=int(bool(twiss_mode)),
            eff_grad=0,  # Temporary placeholder
            _xobject=_xobject,
            **kwargs,
        )

        # Set Python attributes for twiss mode parameters
        self._beta_at_acdipole = beta_at_acdipole
        self._natural_q = natural_q
        self._set_eff_grad()

        # Use the setter to set the kick_plane
        self.plane = plane

    def _set_eff_grad(self):
        if self._beta_at_acdipole == 0:
            self.eff_grad = 0
            return
        nat_q = self._natural_q % 1
        driven_q = self._freq % 1
        self.eff_grad = (
            2
            * (np.cos(2 * np.pi * driven_q) - np.cos(2 * np.pi * nat_q))
            / (self._beta_at_acdipole * np.sin(2 * np.pi * nat_q))
        )

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, value):
        self._freq = value
        self._set_eff_grad()

    @property
    def beta_at_acdipole(self):
        return self._beta_at_acdipole

    @beta_at_acdipole.setter
    def beta_at_acdipole(self, value):
        self._beta_at_acdipole = value
        self._set_eff_grad()

    @property
    def natural_q(self):
        return self._natural_q

    @natural_q.setter
    def natural_q(self, value):
        self._natural_q = value
        self._set_eff_grad()

    @property
    def plane(self):
        return self._int_to_plane[self._plane]

    @plane.setter
    def plane(self, value):
        try:
            self._plane = self._plane_to_int[value]
        except KeyError as e:
            raise ValueError(
                "The plane parameter must be either 'h', 'v', or None."
            ) from e

    @property
    def twiss_mode(self):
        return bool(self._twiss_mode)

    @twiss_mode.setter
    def twiss_mode(self, value):
        self._twiss_mode = int(bool(value))
