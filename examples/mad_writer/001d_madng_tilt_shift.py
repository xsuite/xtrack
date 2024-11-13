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
for nn_quad, sx, sy in zip(tt_quad.name, shift_x, shift_y):
    line[nn_quad].shift_x = sx
    line[nn_quad].shift_y = sy

tw = line.madng_twiss()

mng = line.tracker._madng
import time
t0 = time.perf_counter()
for nn in tt_quads.name:
    nn_ng = nn.replace('.', '_')
    mng.send(
             f'MADX.{nn_ng}.dx = 0\n'
             f'MADX.{nn_ng}.dy = 0\n'
             f'MADX.{nn_ng}.misalign'
             ' = {'
             f'dx={line[nn].shift_x}, dy={line[nn].shift_y}'
                '}')
t1 = time.perf_counter()
print(f'Elapsed time: {t1-t0}')

tw1 = line.madng_twiss()

