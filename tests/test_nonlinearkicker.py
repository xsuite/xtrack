import numpy as np
import pytest

import xobjects as xo
import xtrack as xt
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts


# This decorator ensures the test runs across ContextCpu, ContextCuda, and ContextPyopencl automatically
@for_all_test_contexts
def test_nonlinear_kicker_vs_multiple_wires(test_context):

    # ----------------------------------------------------
    # 1. Physical Parameter Setup
    # ----------------------------------------------------
    p0c = 7000e9  # 7 TeV
    l_phy = 1.5
    l_int = 1.5

    # Define 3 wires with different currents and positions
    # Wire 1: High current, positive offset
    # Wire 2: Negative current, negative offset
    # Wire 3: Low current, near center
    currents = [250.0, -150.0, 50.0]
    x_pos = [-8e-3, 10e-3, 2.1e-3]
    y_pos = [-10e-3, 5e-3, 4e-3]

    # ----------------------------------------------------
    # 2. Case A: Using the new NonlinearKicker element
    # ----------------------------------------------------
    # All wires are computed within this single element
    kicker = xt.NonlinearKicker(
        _context=test_context,  # Crucial: ensures allocation on the correct device memory
        L_phy=l_phy,
        L_int=l_int,
        current=currents,
        xma=x_pos,
        yma=y_pos,
    )

    # ----------------------------------------------------
    # 3. Case B: Using standard xt.Wire (Ground Truth Reference)
    # ----------------------------------------------------
    # Create 3 independent Wire elements to simulate the same effect
    wire_elements = []
    for i in range(len(currents)):
        w = xt.Wire(
            _context=test_context,
            L_phy=l_phy,
            L_int=l_int,
            current=currents[i],
            xma=x_pos[i],
            yma=y_pos[i],
        )
        wire_elements.append(w)

    # ----------------------------------------------------
    # 4. Construct Test Particles
    # ----------------------------------------------------
    # Create a distribution of particles covering various transverse positions
    x_test = np.array([0.0, 1e-3, -5e-3, 2e-3])
    y_test = np.array([0.0, -2e-3, 4e-3, 1e-3])

    # Particle Set 1: For testing NonlinearKicker
    p_kicker = xp.Particles(
        _context=test_context, p0c=p0c, x=x_test.copy(), y=y_test.copy()
    )

    # Particle Set 2: For testing Wire sequence (Reference)
    # Must be a deep copy to ensure identical initial states
    p_wires = p_kicker.copy()

    # ----------------------------------------------------
    # 5. Execute Tracking
    # ----------------------------------------------------

    # Case A: Single pass through NonlinearKicker
    kicker.track(p_kicker)

    # Case B: Sequential passes through the 3 Wires
    for w in wire_elements:
        w.track(p_wires)

    # ----------------------------------------------------
    # 6. Validation
    # ----------------------------------------------------
    # Use xo.assert_allclose to automatically handle GPU->CPU data transfers.
    # We expect results to be nearly identical (double-precision floating point error margin).

    xo.assert_allclose(p_kicker.px, p_wires.px, rtol=1e-13, atol=1e-13)

    xo.assert_allclose(p_kicker.py, p_wires.py, rtol=1e-13, atol=1e-13)


def test_input_validation():
    # Pure CPU test to check if the __init__ method catches errors correctly

    with pytest.raises(ValueError, match="Dimension mismatch"):
        # Intentionally passing inconsistent array lengths
        xt.NonlinearKicker(
            current=[100, 200],  # Length 2
            xma=[0.1],  # Length 1 (Incorrect!)
            yma=[0.1, 0.2],
        )
