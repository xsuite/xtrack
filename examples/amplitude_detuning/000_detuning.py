import numpy as np
import xtrack as xt


# Load a line and build a tracker
# line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

line = xt.Line(elements=[xt.LineSegmentMap(qx=62.31, qy=60.32,
                        det_xx=1000, det_xy=10, det_yx=20, det_yy=2000)])
line.particle_ref = xt.Particles(p0c=7e9)
line.build_tracker()

nemitt_x = 2.5e-6
nemitt_y = 2.5e-6

det = line.get_amplitude_detuning_coefficients()

assert np.isclose(det['det_xx'], 1000, atol=1e-1, rtol=0)
assert np.isclose(det['det_yy'], 2000, atol=1e-1, rtol=0)
assert np.isclose(det['det_xy'], 10, atol=1e-1, rtol=0)
assert np.isclose(det['det_yx'], 20, atol=1e-1, rtol=0)

