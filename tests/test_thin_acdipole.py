import pytest
import xobjects as xo
import xpart as xp
from xobjects.test_helpers import for_all_test_contexts

import xtrack as xt


def _create_fodo_line(test_context) -> xt.Line:
    """Helper function to create a FODO line for testing."""
    n = 6  # Number of FODO cells
    fodo = [
        xt.Multipole(length=0.2, knl=[0, +0.2], ksl=[0, 0]),
        xt.Drift(length=1.0),
        xt.Multipole(length=0.2, knl=[0, -0.2], ksl=[0, 0]),
        xt.Drift(length=1.0),
    ]
    line = xt.Line(elements=n * fodo)
    line.build_tracker(_context=test_context)
    line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=1e9)
    return line


@for_all_test_contexts
@pytest.mark.parametrize(
    "qx_shift", [-0.015, 0.035], ids=lambda v: f"qx_shift={v}"
)
@pytest.mark.parametrize(
    "qy_shift", [0.015,-0.02], ids=lambda v: f"qy_shift={v}"
)
def test_thin_ac_dipole(test_context, qx_shift, qy_shift):
    """Test the effect of a thin AC dipole on the tune shift."""
    line = _create_fodo_line(test_context)
    base_tws = line.twiss(method="4d")

    nat_qx, nat_qy = base_tws["qx"], base_tws["qy"]
    drv_qx, drv_qy = nat_qx + qx_shift, nat_qy + qy_shift

    e5_pos = line.get_s_position("e5")
    e5_betx = base_tws.rows["e5"]["betx"].item()
    e5_bety = base_tws.rows["e5"]["bety"].item()

    # Define AC dipole elements
    line.env.elements["e5_hacd"] = xt.ACDipoleThinHorizontal(
        natural_qx=nat_qx, driven_qx=drv_qx, betx_at_acdipole=e5_betx
    )
    line.env.elements["e5_vacd"] = xt.ACDipoleThinVertical(
        natural_qy=nat_qy, driven_qy=drv_qy, bety_at_acdipole=e5_bety
    )

    line.insert("e5_hacd", at=e5_pos)
    line.insert("e5_vacd", at=e5_pos)
    line.build_tracker(_context=test_context)
    tws_both = line.twiss(method="4d")

    xo.assert_allclose(tws_both["qx"], drv_qx, rtol=1e-10, atol=1e-15)
    xo.assert_allclose(tws_both["qy"], drv_qy, rtol=1e-10, atol=1e-15)
