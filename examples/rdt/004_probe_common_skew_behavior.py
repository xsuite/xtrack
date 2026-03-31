import xtrack as xt
import numpy as np

env = xt.load(
    '../../test_data/hllhc15_thick/hllhc15_collider_thick.json')

line_b1 = env['lhcb1']
line_b2 = env['lhcb2']

line_b2.twiss_default['reverse'] = False

env['kqsx3.r1'] = 5e-5

tw1 = line_b1.twiss4d(coupling_edw_teng=True)
tw2 = line_b2.twiss4d(coupling_edw_teng=True)

print('kqsx3.r1 = ', env['kqsx3.r1'], 'kqsx3.l1 = ', env['kqsx3.l1'])
print(f'c_minus b1 = {tw1.c_minus_re_0:.3e} + {tw1.c_minus_im_0:.3e}j')
print(f'c_minus b2 = {tw2.c_minus_re_0:.3e} + {tw2.c_minus_im_0:.3e}j')
print(f'f1001 b1 = {tw1.f1001.real[0]:.3e} + {tw1.f1001.imag[0]:.3e}j')
print(f'f1001 b2 = {tw2.f1001.real[0]:.3e} + {tw2.f1001.imag[0]:.3e}j')

env['kqsx3.r1'] = 5e-5
env['kqsx3.l1'] = 5e-5

tw1 = line_b1.twiss4d(coupling_edw_teng=True)
tw2 = line_b2.twiss4d(coupling_edw_teng=True)
print()
print('kqsx3.r1 = ', env['kqsx3.r1'], 'kqsx3.l1 = ', env['kqsx3.l1'])
print(f'c_minus b1 = {tw1.c_minus_re_0:.3e} + {tw1.c_minus_im_0:.3e}j')
print(f'c_minus b2 = {tw2.c_minus_re_0:.3e} + {tw2.c_minus_im_0:.3e}j')
print(f'f1001 b1 = {tw1.f1001.real[0]:.3e} + {tw1.f1001.imag[0]:.3e}j')
print(f'f1001 b2 = {tw2.f1001.real[0]:.3e} + {tw2.f1001.imag[0]:.3e}j')