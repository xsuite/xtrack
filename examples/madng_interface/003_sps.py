from cpymad.madx import Madx
import xtrack as xt
import numpy as np

madx = Madx()
# madx.input('option, -rbarc;')
madx.call('../../test_data/sps_thick/sps.seq')
madx.call('../../test_data/sps_thick/lhc_q20.str')
madx.beam()
madx.use('sps')
madx.twiss()
twmadx = xt.Table(madx.table.twiss)


import pymadng as pg
mng = pg.MAD()
mng.send('''
    MADX:load('../../test_data/sps_thick/sps.seq')
    MADX:load('../../test_data/sps_thick/lhc_q20.str')
    MADX.sps.beam = MAD.beam{particle='proton', pc=26};
''')
twng = mng.twiss(sequence='MADX.sps')[0].to_df()

ds = -twmadx['s', 'drift_10:0'] + twmadx['s', 'mbb.10150:1']
l_madx = madx.elements['mbb.10150'].l
print('madx:')
print(f'ds = {ds} m')
print(f'l = {l_madx} m')

i_mb = np.where(twng.name.values == 'mbb.10150'.upper())[0][0]

