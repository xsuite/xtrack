import numpy as np
import xobjects as xo

import xtrack as xt


class ACDipoleThickVertical(xt.BeamElement):
    """
    ACDipoleTrack is a thin element that applies an oscillating kick to the beam in the x and y directions.
    It is used for beam excitations in circular machines for optics measurements. The kick is
    defined by the voltages, frequencies, and phase lags in the x and y directions.
    The kick is applied as a function of the turn number, and it can be ramped up and down
    to avoid emittance growth. The transverse momentum in the vertical plane is changed by
    `(0.3 * volt/p0c) * sin(2π * freq * turn + lag)`.

    Since, this acts as a kicker that has a frequency over a number of turns, it is
    only useful when running a tracking simulation with a fixed number of turns. If
    you need to run a twiss for a single turn, you should use the `ACDipoleThinHorizontal`
    element instead.

    Parameters
    ----------
    volt : list of float
        The voltages applied in the y direction,
    freq : list of float
        The frequencies of the AC dipole in the y direction, in units of Hz.
    lag : list of float
        The phase lag of the AC dipole in the y direction, in units of radians.
    ramp : list of int
        The ramp settings for the AC dipole, defining the turns for ramping up and down the kick.
        The list should contain four integers: [ramp1, ramp2, ramp3, ramp4].
        - `ramp1`: Starting turn of amplitude ramp-up.
        - `ramp2`: Last turn of amplitude ramp-up.
        - `ramp3`: Starting turn of amplitude ramp-down.
        - `ramp4`: Last turn of amplitude ramp-down.
    """

    def __init__(
        self, *, volt=None, freq=None, lag=None, ramp=None, _xobject=None, **kwargs
    ):
        if _xobject is not None:
            super().__init__(_xobject=_xobject)
        else:
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
            elif not (
                (isinstance(ramp, (list, tuple)) or hasattr(ramp, "__iter__"))
                and len(ramp) == 4
                and all(isinstance(int(x), int) for x in ramp)
            ):
                raise ValueError(
                    "The ramp parameter must be a list of four integers: [ramp1, ramp2, ramp3, ramp4]."
                )

            super().__init__(
                volt=volt, freq=freq, lag=lag, ramp=ramp, _xobject=_xobject, **kwargs
            )

    _xofields = {
        # Voltage defines the strength of the kick
        "volt": xo.Float64,
        # Frequency defines strength depending on delta to tune
        "freq": xo.Float64,
        # Lag
        "lag": xo.Float64,
        # Ramping parameters
        "ramp": xo.UInt16[:],
    }

    _extra_c_sources = [
        "#include <beam_elements/elements_src/acdipole_vertical.h>",
    ]


class ACDipoleThickHorizontal(xt.BeamElement):
    """
    ACDipoleTrack is a thin element that applies an oscillating kick to the beam in the x and y directions.
    It is used for beam excitations in circular machines for optics measurements. The kick is
    defined by the voltages, frequencies, and phase lags in the x and y directions.
    The kick is applied as a function of the turn number, and it can be ramped up and down
    to avoid emittance growth. The transverse momentum in the horizontal plane is changed by
    `(0.3 * volt/p0c) * sin(2π * freq * turn + lag)`.

    Since, this acts as a kicker that has a frequency over a number of turns, it is
    only useful when running a tracking simulation with a fixed number of turns. If
    you need to run a twiss for a single turn, you should use the `ACDipoleThinHorizontal`
    element instead.

    Parameters
    ----------
    volt : list of float
        The voltages applied in the x direction,
    freq : list of float
        The frequencies of the AC dipole in the x direction, in units of Hz.
    lag : list of float
        The phase lag of the AC dipole in the x direction, in units of radians.
    ramp : list of int
        The ramp settings for the AC dipole, defining the turns for ramping up and down the kick.
        The list should contain four integers: [ramp1, ramp2, ramp3, ramp4].
        - `ramp1`: Starting turn of amplitude ramp-up.
        - `ramp2`: Last turn of amplitude ramp-up.
        - `ramp3`: Starting turn of amplitude ramp-down.
        - `ramp4`: Last turn of amplitude ramp-down.
    """

    def __init__(
        self, *, volt=None, freq=None, lag=None, ramp=None, _xobject=None, **kwargs
    ):
        if _xobject is not None:
            super().__init__(_xobject=_xobject)
        else:
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
            elif not (
                (isinstance(ramp, (list, tuple)) or hasattr(ramp, "__iter__"))
                and len(ramp) == 4
                and all(int(x) == x for x in ramp)
            ):
                raise ValueError(
                    "The ramp parameter must be a list of four integers: [ramp1, ramp2, ramp3, ramp4]."
                )

            super().__init__(
                volt=volt, freq=freq, lag=lag, ramp=ramp, _xobject=_xobject, **kwargs
            )

    _xofields = {
        # Voltage defines the strength of the kick
        "volt": xo.Float64,
        # Frequency defines strength depending on delta to tune
        "freq": xo.Float64,
        # Lag
        "lag": xo.Float64,
        # Ramping parameters
        "ramp": xo.UInt16[:],
    }

    _extra_c_sources = [
        "#include <beam_elements/elements_src/acdipole_horizontal.h>",
    ]


class ACDipoleThinVertical(xt.BeamElement):
    """
    ACDipoleThinVertical is a thin element that approximates the effect of an AC dipole,
    simulating it as a thin gradient error see Miyamoto, R., Kopp, S., Jansson, A., & Syphers, M. (2008).
    Parametrization of the driven betatron oscillation. Phys. Rev. ST Accel. Beams, 11, 084002 for more details.
    It applies a beta and tune shift to the beam in the vertical plane, depending on the natural and driven tunes.

    If any of the parameters `natural_qy`, `driven_qy`, or `bety_at_acdipole` are not provided,
    the effective gradient (`eff_grad`) is set to zero, meaning no kick is applied.

    Parameters
    ----------
    natural_qy : float
        The natural vertical tune of the machine.
    driven_qy : float
        The driven vertical tune desired for the AC dipole.
    bety_at_acdipole : float
        The beta function at the location of the AC dipole, in meters.
    """

    def __init__(
        self,
        *,
        natural_qy=None,
        driven_qy=None,
        bety_at_acdipole=None,
        _xobject=None,
        **kwargs,
    ):
        if _xobject is not None:
            super().__init__(_xobject=_xobject)
        else:
            if natural_qy is None or driven_qy is None or bety_at_acdipole is None:
                eff_grad = 0
            else:
                eff_grad = (
                    2
                    * (np.cos(2 * np.pi * driven_qy) - np.cos(2 * np.pi * natural_qy))
                    / (bety_at_acdipole * np.sin(2 * np.pi * natural_qy))
                )
            super().__init__(eff_grad=eff_grad, _xobject=_xobject, **kwargs)

    _xofields = {
        # Effective gradient of the AC dipole.
        "eff_grad": xo.Float64,
    }

    _extra_c_sources = [
        "#include <beam_elements/elements_src/thin_vacdipole.h>",
    ]


class ACDipoleThinHorizontal(xt.BeamElement):
    """
    ACDipoleThinHorizontal is a thin element that approximates the effect of an AC dipole,
    simulating it as a thin gradient error see Miyamoto, R., Kopp, S., Jansson, A., & Syphers, M. (2008).
    Parametrization of the driven betatron oscillation. Phys. Rev. ST Accel. Beams, 11, 084002 for more details.
    It applies a beta and tune shift to the beam in the horizontal plane, depending on the natural and driven tunes.

    If any of the parameters `natural_qx`, `driven_qx`, or `betx_at_acdipole` are not provided,
    the effective gradient (`eff_grad`) is set to zero, meaning no kick is applied.

    Parameters
    ----------
    natural_qx : float
        The natural horizontal tune of the machine.
    driven_qx : float
        The driven horizontal tune desired for the AC dipole.
    betx_at_acdipole : float
        The beta function at the location of the AC dipole, in meters.
    """

    def __init__(
        self,
        *,
        natural_qx=None,
        driven_qx=None,
        betx_at_acdipole=None,
        _xobject=None,
        **kwargs,
    ):
        if _xobject is not None:
            super().__init__(_xobject=_xobject)
        else:
            if natural_qx is None or driven_qx is None or betx_at_acdipole is None:
                eff_grad = 0
            else:
                # Ensure that the tunes are in the range [0, 1)
                natural_qx = natural_qx % 1
                driven_qx = driven_qx % 1
                eff_grad = (
                    2
                    * (np.cos(2 * np.pi * driven_qx) - np.cos(2 * np.pi * natural_qx))
                    / (betx_at_acdipole * np.sin(2 * np.pi * natural_qx))
                )
            super().__init__(eff_grad=eff_grad, _xobject=_xobject, **kwargs)

    _xofields = {
        # Effective gradient of the AC dipole.
        "eff_grad": xo.Float64,
    }

    _extra_c_sources = [
        "#include <beam_elements/elements_src/thin_hacdipole.h>",
    ]
