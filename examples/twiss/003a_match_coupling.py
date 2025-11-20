import xtrack as xt
import xobjects as xo

line = xt.load('../../test_data/hllhc14_no_errors_with_coupling_knobs/line_b1.json')

target_qx = 62.315
target_qy = 60.325

tw = line.twiss()

# Try to measure and match coupling
line['cmrskew'] = 1e-3
line['cmiskew'] = 1e-3

# Match coupling
line.match(
    vary=[
        xt.Vary(name='cmrskew', limits=[-0.5e-2, 0.5e-2], step=1e-5),
        xt.Vary(name='cmiskew', limits=[-0.5e-2, 0.5e-2], step=1e-5)],
    targets=[
        # xt.Target('c_r1_avg', 0, tol=2e-2),
        # xt.Target('c_r2_avg', 0, tol=2e-2),
        xt.Target('c_minus', 0, tol=1e-4),
        ])
