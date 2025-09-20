import xtrack as xt
from pathlib import Path

import numpy as np
import xobjects as xo

tmp_path = Path('./')

env = xt.load('../../test_data/sps_thick/sps.seq')
env.vars.load('../../test_data/sps_thick/lhc_q20.str')
line = env.sps
line.set_particle_ref('p', energy0=26e9)

tw = line.twiss4d()
tw.to_csv(tmp_path / 'twiss_test.csv')

tw_test = xt.load(tmp_path / 'twiss_test.csv')

assert np.all(np.array(tw_test._col_names) == np.array(tw._col_names))
assert set(tw_test.keys()) - set(tw.keys()) == {'__class__', 'xtrack_version'}
assert set(tw.keys()) - set(tw_test.keys()) == {'_action'}

for kk in tw._data:
    if kk == '_action':
        continue
    if kk in ['particle_on_co', 'steps_r_matrix', 'R_matrix_ebe', 'radiation_method',
              'line_config', 'completed_init']:
        continue # To be checked separately
    if isinstance(tw[kk], np.ndarray) and isinstance(tw[kk][0], str):
        assert np.all(tw[kk] == tw_test[kk])
        continue
    elif isinstance(tw[kk], str):
        assert tw[kk] == tw_test[kk]
        continue
    xo.assert_allclose(tw[kk], tw_test[kk], rtol=1e-10, atol=1e-15)
