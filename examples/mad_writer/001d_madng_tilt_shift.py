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
commands = []
for nn in line.element_names:
    if not hasattr(line[nn], 'shift_x'):
        continue
    nn_ng = nn.replace('.', '_')
    commands.append(
             f'MADX.{nn_ng}.dx = 0\n'
             f'MADX.{nn_ng}.dy = 0\n'
             f'MADX.{nn_ng}.misalign'
             ' = {'
             f'dx={line[nn].shift_x}, dy={line[nn].shift_y}'
                '}')
t1 = time.perf_counter()
mng.send('\n'.join(commands))
t2 = time.perf_counter()
print(f'Elapsed time loop: {t1-t0}')
print(f'Elapsed time send: {t2-t1}')

tw1 = line.madng_twiss()

xo.assert_allclose(tw1.x, tw1.x_ng, atol=2e-4*tw1.x.std(), rtol=0)
xo.assert_allclose(tw1.y, tw1.y_ng, atol=2e-4*tw1.y.std(), rtol=0)
