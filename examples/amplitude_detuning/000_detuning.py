import numpy as np
import xtrack as xt
import NAFFlib as nl


# Load a line and build a tracker
# line = xt.Line.from_json('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

line = xt.Line(elements=[xt.LineSegmentMap(qx=62.31, qy=60.32,
                        detx_x=1000, detx_y=10, dety_x=20, dety_y=2000)])
line.particle_ref = xt.Particles(p0c=7e9)
line.build_tracker()

nemitt_x = 2.5e-6
nemitt_y = 2.5e-6

det = line.get_amplitude_detuning_coefficients()
