import numpy as np
import xtrack as xt
import xobjects as xo

line = xt.Line.from_json(
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tt = line.get_table()
tt_quads = tt.rows[tt.element_type=='Quadrupole']

# Introduce misalignments on all quadrupoles
tt = line.get_table()
tt_quad = tt.rows['mq\..*']
rgen = np.random.RandomState(1) # fix seed for random number generator
                                # (to have reproducible results)
shift_x = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
shift_y = rgen.randn(len(tt_quad)) * 0.01e-3 # 0.01 mm rms shift on all quads
rot_s = rgen.randn(len(tt_quad)) * 1e-3 # 1 mrad rms rotation on all quads

for nn_quad, sx, sy, rr in zip(tt_quad.name, shift_x, shift_y, rot_s):
    line[nn_quad].shift_x = sx
    line[nn_quad].shift_y = sy
    line[nn_quad].rot_s_rad = rr

line['mq.15l7.b1'].knl[2] = 0.5

tw = line.madng_twiss()

xo.assert_allclose(tw.x, tw.x_ng, atol=5e-4*tw.x.std(), rtol=0)
xo.assert_allclose(tw.y, tw.y_ng, atol=5e-4*tw.y.std(), rtol=0)
xo.assert_allclose(tw.betx2, tw.beta12_ng, atol=0, rtol=2e-3)
xo.assert_allclose(tw.bety1, tw.beta21_ng, atol=0, rtol=2e-3)
xo.assert_allclose(tw.wx_chrom, tw.wx_ng, atol=0, rtol=1e-3)
