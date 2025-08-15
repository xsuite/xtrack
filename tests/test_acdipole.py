from collections import namedtuple

import pytest
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt


# =====================
# Utility Functions
# =====================
def _create_test_particles(at_turn: int) -> xp.Particles:
    """Create test particles and verify initial conditions."""
    particles = xp.Particles(at_turn=at_turn)
    assert particles.x[0] == 0.0, f"Expected initial x=0, but got x={particles.x[0]}"
    assert particles.y[0] == 0.0, f"Expected initial y=0, but got y={particles.y[0]}"
    assert particles.py[0] == 0.0, (
        f"Expected initial py=0, but got py={particles.py[0]}"
    )
    assert particles.px[0] == 0.0, (
        f"Expected initial px=0, but got px={particles.px[0]}"
    )
    return particles


def get_acdipole_results(
    test_context,
    acdipole_class: type[xt.BeamElement],
    turn: int,
    test_voltage: float = 1.5,
    test_freq: float = 0.25,
    test_lag: float = 0.0,
) -> tuple:
    """Track particles through an ACDipole and return coordinates."""
    particles = _create_test_particles(at_turn=turn)

    acdipole = acdipole_class(
        volt=test_voltage,
        freq=test_freq,
        lag=test_lag,
        ramp=[0, 10, 100, 110],
        _context=test_context,
    )

    acdipole.track(particles)
    return particles.x[0], particles.px[0], particles.y[0], particles.py[0]


def assert_acdipole_kick(
    *,
    test_context,
    acdipole_class,
    test_turn,
    test_volt,
    test_freq,
    test_lag,
    kick_attr,
    expected_kick,
):
    """
    Assert that only the coordinate corresponding to `kick_attr` receives the expected kick,
    and all other coordinates remain zero.
    All arguments must be passed as keyword arguments for clarity.
    """
    x, px, y, py = get_acdipole_results(
        test_context, acdipole_class, test_turn, test_volt, test_freq, test_lag
    )
    vals = {"x": x, "px": px, "y": y, "py": py}
    for coord in vals:
        if coord == kick_attr:
            assert abs(vals[coord] - expected_kick) < 1e-10, (
                f"Turn {test_turn}: Expected {coord}={expected_kick}, but got {coord}={vals[coord]}"
            )
        else:
            assert vals[coord] == 0.0, (
                f"Turn {test_turn}: Expected {coord}=0, but got {coord}={vals[coord]}"
            )


# =====================
# Flattop Test Parameters and Helper
# =====================
flattop_cases = [
    (2.25, 45, 0.25, 0.0, "flattop, 2.25V, freq=0.25, lag=0.0"),
    (1.5, 46, 1 / 3, -1 / 3, "flattop, 1.5V, freq=1/3, lag=-0.25"),
    (1.5, 47, 1 / 3, 1 / 12, "flattop, 1.5V, freq=1/3, lag=+0.25"),
]
flattop_params = [(v, t, f, lag) for v, t, f, lag, _ in flattop_cases]
flattop_ids = [desc for _, _, _, _, desc in flattop_cases]


def _calculate_flattop_kick(test_volt, test_turn) -> float:
    """Compute expected kick for flattop tests."""
    if test_turn == 45:
        return test_volt * 300e-3
    if test_turn == 46:
        return 0
    if test_turn == 47:
        return -test_volt * 300e-3
    raise ValueError(
        f"Unexpected test_turn={test_turn} in flattop tests. Expected 45, 46, or 47."
    )


# =====================
# Flattop Tests
# =====================
@for_all_test_contexts
@pytest.mark.parametrize(
    "test_volt, test_turn, test_freq, test_lag",
    flattop_params,
    ids=flattop_ids,
)
def test_vacdipole_flattop(
    test_context, test_volt, test_turn, test_freq, test_lag
) -> None:
    """
    Test vertical ACDipole flattop for three specific cases.
    """
    expected_kick = _calculate_flattop_kick(test_volt, test_turn)
    assert_acdipole_kick(
        test_context=test_context,
        acdipole_class=xt.ACDipoleThickVertical,
        test_turn=test_turn,
        test_volt=test_volt,
        test_freq=test_freq,
        test_lag=test_lag,
        kick_attr="py",
        expected_kick=expected_kick,
    )


@for_all_test_contexts
@pytest.mark.parametrize(
    "test_volt, test_turn, test_freq, test_lag",
    flattop_params,
    ids=flattop_ids,
)
def test_hacdipole_flattop(
    test_context, test_volt, test_turn, test_freq, test_lag
) -> None:
    """
    Test horizontal ACDipole flattop for three specific cases.
    """
    expected_kick = _calculate_flattop_kick(test_volt, test_turn)
    assert_acdipole_kick(
        test_context=test_context,
        acdipole_class=xt.ACDipoleThickHorizontal,
        test_turn=test_turn,
        test_volt=test_volt,
        test_freq=test_freq,
        test_lag=test_lag,
        kick_attr="px",
        expected_kick=expected_kick,
    )


# =====================
# Ramp Test Parameters and Helper
# =====================
AcdipoleRampCase = namedtuple(
    "AcdipoleRampCase", ["volt", "turn", "freq", "lag", "desc"]
)
acdipole_ramp_cases = [
    AcdipoleRampCase(1.5, 5, 0.25, 0.0, "First ramp up, quarter period, no lag"),
    AcdipoleRampCase(
        1.5, 105, 1.25, 0.0, "Ramp down, after 100 turns, freq > 1, no lag"
    ),
    AcdipoleRampCase(2.25, 6, 1 / 3, -0.25, "Early ramp, third period, negative lag"),
    AcdipoleRampCase(
        1.5, 107, 1 / 3, 1 / 12, "Late ramp, third period, small positive lag"
    ),
]
acdipole_ramp_params = [(c.volt, c.turn, c.freq, c.lag) for c in acdipole_ramp_cases]
acdipole_ramp_ids = [c.desc for c in acdipole_ramp_cases]


def _calculate_ramp_kick(test_volt, test_turn):
    """
    Compute expected kick for ramp tests.
    - For turns <= 100, the kick ramps up linearly.
    - For turns > 100, the kick ramps down linearly.
    - The sign alternates based on the turn number.
    """
    kick_sign = (-1) ** (test_turn % 5 > 0)
    if test_turn > 100:
        return kick_sign * test_volt * 300e-3 * (1 - (test_turn - 100) / 10)
    return kick_sign * test_volt * 300e-3 * (test_turn / 10)


# =====================
# Ramp Tests
# =====================
@for_all_test_contexts
@pytest.mark.parametrize(
    "test_volt, test_turn, test_freq, test_lag",
    acdipole_ramp_params,
    ids=acdipole_ramp_ids,
)
def test_vacdipole_ramp(
    test_context, test_volt, test_turn, test_freq, test_lag
) -> None:
    """
    Test vertical ACDipole ramp:
    - Only py should receive the expected kick, all other coordinates should remain zero.
    - Each test case is described in the test ID.
    """
    expected_kick = _calculate_ramp_kick(test_volt, test_turn)
    assert_acdipole_kick(
        test_context=test_context,
        acdipole_class=xt.ACDipoleThickVertical,
        test_turn=test_turn,
        test_volt=test_volt,
        test_freq=test_freq,
        test_lag=test_lag,
        kick_attr="py",
        expected_kick=expected_kick,
    )


@for_all_test_contexts
@pytest.mark.parametrize(
    "test_volt, test_turn, test_freq, test_lag",
    acdipole_ramp_params,
    ids=acdipole_ramp_ids,
)
def test_hacdipole_ramp(
    test_context, test_volt, test_turn, test_freq, test_lag
) -> None:
    """
    Test horizontal ACDipole ramp:
    - Only px should receive the expected kick, all other coordinates should remain zero.
    - Each test case is described in the test ID.
    """
    expected_kick = _calculate_ramp_kick(test_volt, test_turn)
    assert_acdipole_kick(
        test_context=test_context,
        acdipole_class=xt.ACDipoleThickHorizontal,
        test_turn=test_turn,
        test_volt=test_volt,
        test_freq=test_freq,
        test_lag=test_lag,
        kick_attr="px",
        expected_kick=expected_kick,
    )
