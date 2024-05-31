import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import for_all_test_contexts

@for_all_test_contexts
def test_amplitude_detuning(test_context):
    line = xt.Line(elements=[xt.LineSegmentMap(qx=62.31, qy=60.32,
                            det_xx=1000, det_xy=10, det_yx=20, det_yy=2000)])
    line.particle_ref = xt.Particles(p0c=7e9)
    line.build_tracker(_context=test_context)

    det = line.get_amplitude_detuning_coefficients()

    xo.assert_allclose(det['det_xx'], 1000, atol=1e-1, rtol=0)
    xo.assert_allclose(det['det_yy'], 2000, atol=1e-1, rtol=0)
    xo.assert_allclose(det['det_xy'], 10, atol=1e-1, rtol=0)
    xo.assert_allclose(det['det_yx'], 20, atol=1e-1, rtol=0)
