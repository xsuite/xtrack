"""
Test suite for ACDipole elements in Xtrack.

This module tests the behavior of the ACDipole
elements, focusing on their kick effects during ramp-up, flattop, and ramp-down phases.
Tests verify that only the appropriate coordinate (py for vertical, px for horizontal)
receives the expected kick while others remain zero.
"""

from collections import namedtuple
from typing import Any

import pytest
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt
import xobjects as xo


# Constants
KICK_FACTOR = 300e-3  # Conversion factor for kick strength (mrad/V)
TOLERANCE = 1e-10  # Numerical tolerance for kick comparisons

RAMP_LENGTH = 10.0  # Number of turns for ramp up/down phases
FLATTOP_START = 100  # Turn number when flattop phase begins

# Turn numbers defining ramp phases: [start, end_ramp_up, start_ramp_down, end]:
RAMP_SCHEDULE = [0, RAMP_LENGTH, FLATTOP_START, FLATTOP_START + RAMP_LENGTH]

PLANES = ["x", "y"]


def get_acdipole_results(
    test_context: Any,
    turn: int,
    plane: str,
    test_voltage: float = 1.5,
    test_freq: float = 0.25,
    test_lag: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Track particles through an ACDipole and return final coordinates.

    Args:
        test_context: The computational context for the simulation.
        acdipole_class: The ACDipole class to instantiate (vertical or horizontal).
        turn: The turn number for tracking.
        test_voltage: Voltage setting for the ACDipole.
        test_freq: Frequency setting for the ACDipole.
        test_lag: Phase lag setting for the ACDipole.

    Returns:
        Tuple of (x, px, y, py) coordinates after tracking.
    """
    particles = xt.Particles(at_turn=turn, _context=test_context)  # Must be 0.

    acdipole = xt.ACDipole(
        volt=test_voltage,
        freq=test_freq,
        lag=test_lag,
        ramp=RAMP_SCHEDULE,
        _context=test_context,
        plane="h",
    )
    # Test the setter
    acdipole.plane = plane

    acdipole.track(particles)
    return particles.x[0], particles.px[0], particles.y[0], particles.py[0]


def assert_acdipole_kick(
    *,
    test_context: Any,
    test_turn: int,
    test_plane: str,
    test_volt: float,
    test_freq: float,
    test_lag: float,
    expected_kick: float,
) -> None:
    """
    Assert that only the specified coordinate receives the expected kick.

    Tracks particles through the ACDipole and verifies that only the coordinate
    corresponding to kick_attr receives the expected kick, while all other
    coordinates remain zero.

    Args:
        test_context: The computational context for the simulation.
        acdipole_class: The ACDipole class to test.
        test_turn: Turn number for the test.
        test_volt: Voltage setting.
        test_freq: Frequency setting.
        test_lag: Phase lag setting.
        kick_attr: The coordinate that should receive the kick ("px" or "py").
        expected_kick: The expected kick value for the specified coordinate.

    Raises:
        AssertionError: If the kick is not applied correctly.
    """
    x, px, y, py = get_acdipole_results(
        test_context, test_turn, test_plane, test_volt, test_freq, test_lag
    )
    vals = {"x": x, "px": px, "y": y, "py": py}  # Map coordinate names to values
    for coord in vals:
        if coord == f"p{test_plane}":
            xo.assert_allclose(vals[coord], expected_kick, atol=TOLERANCE, rtol=0)
        else:
            assert vals[coord] == 0.0, (
                f"Turn {test_turn}: Expected {coord}=0, but got {coord}={vals[coord]}"
            )


# =====================
# Flattop Test Parameters and Helper
# =====================
FlattopCase = namedtuple("FlattopCase", ["volt", "turn", "freq", "lag", "desc"])
FLATTOP_CASES = [
    FlattopCase(2.25, 45, 0.25, 0.0, "flattop, 2.25V, freq=0.25, lag=0.0"),
    FlattopCase(1.5, 46, 1 / 3, -1 / 3, "flattop, 1.5V, freq=0.333..., lag=-0.333..."),
    FlattopCase(1.5, 47, 1 / 3, 1 / 12, "flattop, 1.5V, freq=0.333..., lag=0.0833..."),
]


def _calculate_flattop_kick(test_volt: float, test_turn: int) -> float:
    """
    Compute the expected kick during the flattop phase.

    During flattop (turns 45-47), the kick depends on the turn:
    - Turn 45: Positive kick
    - Turn 46: Zero kick (phase cancellation)
    - Turn 47: Negative kick

    Args:
        test_volt: The voltage setting.
        test_turn: The turn number.

    Returns:
        The expected kick value.

    Raises:
        ValueError: If test_turn is not 45, 46, or 47.
    """
    if test_turn == 45:
        return test_volt * KICK_FACTOR
    if test_turn == 46:
        return 0
    if test_turn == 47:
        return -test_volt * KICK_FACTOR
    raise ValueError(
        f"Unexpected test_turn={test_turn} in flattop tests. Expected 45, 46, or 47."
    )


# =====================
# Flattop Tests
# =====================
@for_all_test_contexts
@pytest.mark.parametrize("plane", PLANES, ids=lambda o: o.upper())
@pytest.mark.parametrize("case", FLATTOP_CASES, ids=lambda c: c.desc)
def test_acdipole_flattop(
    test_context: Any,
    case: FlattopCase,
    plane: str,
) -> None:
    """
    Test ACDipole behavior during flattop phase for both orientations.

    Verifies that the ACDipole applies the correct kick during the flattop
    phase (constant amplitude) for vertical and horizontal orientations.
    The test is parametrized over different voltage, frequency, and lag settings.
    """
    expected_kick = _calculate_flattop_kick(case.volt, case.turn)
    assert_acdipole_kick(
        test_context=test_context,
        test_turn=case.turn,
        test_plane=plane,
        test_volt=case.volt,
        test_freq=case.freq,
        test_lag=case.lag,
        expected_kick=expected_kick,
    )


# =====================
# Ramp Test Parameters and Helper
# =====================
AcdipoleRampCase = namedtuple(
    "AcdipoleRampCase", ["volt", "turn", "freq", "lag", "desc"]
)
ACDIPOLE_RAMP_CASES = [
    AcdipoleRampCase(1.5, 5, 0.25, 0.0, "First ramp up, quarter period, no lag"),
    AcdipoleRampCase(
        1.5, 105, 1.25, 0.0, "Ramp down, after 100 turns, freq > 1, no lag"
    ),
    AcdipoleRampCase(2.25, 6, 1 / 3, -0.25, "Early ramp, third period, negative lag"),
    AcdipoleRampCase(
        1.5, 107, 1 / 3, 1 / 12, "Late ramp, third period, small positive lag"
    ),
]


def _calculate_ramp_kick(test_volt: float, test_turn: int) -> float:
    """
    Compute the expected kick during ramp phases.

    The ACDipole has three phases based on turn number:
    - Ramp up: turns 0-100, kick increases linearly
    - Flattop: turns 100-110, constant kick (handled separately)
    - Ramp down: turns >100, kick decreases linearly

    The kick sign alternates based on turn number modulo 5.

    Args:
        test_volt: The voltage setting.
        test_turn: The turn number.

    Returns:
        The expected kick value.
    """
    # Alternating sign based on turn number (simulates AC oscillation)
    kick_sign = (-1) ** (test_turn % 5 > 0)
    if test_turn > FLATTOP_START:
        # Ramp down phase: kick decreases from max to zero over RAMP_LENGTH turns
        return (
            kick_sign
            * test_volt
            * KICK_FACTOR
            * (1 - (test_turn - FLATTOP_START) / RAMP_LENGTH)
        )
    # Ramp up phase: kick increases linearly from zero to max over RAMP_LENGTH turns
    return kick_sign * test_volt * KICK_FACTOR * (test_turn / RAMP_LENGTH)


# =====================
# Ramp Tests
# =====================
@for_all_test_contexts
@pytest.mark.parametrize("plane", PLANES, ids=lambda o: o.upper())
@pytest.mark.parametrize("case", ACDIPOLE_RAMP_CASES, ids=lambda c: c.desc)
def test_acdipole_ramp(
    test_context: Any,
    case: AcdipoleRampCase,
    plane: str,
) -> None:
    """
    Test ACDipole behavior during ramp phases for both orientations.

    Verifies that the ACDipole applies linearly increasing/decreasing kicks
    during ramp-up and ramp-down phases, with alternating signs to simulate
    AC field oscillations. Only the appropriate coordinate receives the kick.
    """
    expected_kick = _calculate_ramp_kick(case.volt, case.turn)
    assert_acdipole_kick(
        test_context=test_context,
        test_turn=case.turn,
        test_plane=plane,
        test_volt=case.volt,
        test_freq=case.freq,
        test_lag=case.lag,
        expected_kick=expected_kick,
    )
