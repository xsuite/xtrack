from typing import Any

import pytest
import xobjects as xo
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt


@for_all_test_contexts
@pytest.mark.parametrize("qx_shift", [-0.015, 0.035], ids=lambda v: f"qx_shift={v}")
@pytest.mark.parametrize("qy_shift", [0.015, -0.02], ids=lambda v: f"qy_shift={v}")
def test_ac_dipole_twiss(test_context: Any, qx_shift: float, qy_shift: float):
    """Test the effect of a thin AC dipole on the tune shift."""
    n = 4  # Number of FODO cells
    fodo = [
        xt.Multipole(length=0.2, knl=[0, +0.2], ksl=[0, 0]),
        xt.Drift(length=1.0),
        xt.Multipole(length=0.2, knl=[0, -0.2], ksl=[0, 0]),
        xt.Drift(length=1.0),
    ]
    line = xt.Line(elements=n * fodo)
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=1e9)
    line.build_tracker(_context=test_context)
    base_tws = line.twiss(method="4d")
    nat_qx, nat_qy = base_tws["qx"], base_tws["qy"]
    e5_pos = line.get_s_position("e5")
    e5_betx = base_tws.rows["e5"]["betx"].item()
    e5_bety = base_tws.rows["e5"]["bety"].item()

    drv_qx, drv_qy = nat_qx + qx_shift, nat_qy + qy_shift

    # Define AC dipole elements
    line.env.elements["e5_hacd"] = xt.ACDipole(
        natural_q=nat_qx,
        freq=drv_qx,
        beta_at_acdipole=e5_betx,
        plane="h",
        twiss_mode=True,
    )
    line.env.elements["e5_vacd"] = xt.ACDipole(
        natural_q=nat_qy,
        freq=drv_qy,
        beta_at_acdipole=e5_bety,
        plane="v",
    )

    line.insert("e5_hacd", at=e5_pos)
    line.insert("e5_vacd", at=e5_pos)

    # Test the setter of twiss_mode property
    line.env.elements["e5_vacd"].twiss_mode = True

    line.build_tracker(_context=test_context)
    tws_both = line.twiss(method="4d")

    xo.assert_allclose(tws_both["qx"], drv_qx, rtol=1e-10, atol=1e-15)
    xo.assert_allclose(tws_both["qy"], drv_qy, rtol=1e-10, atol=1e-15)

    del line
