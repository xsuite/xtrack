import xtrack as xt
from pathlib import Path
import xobjects as xo

test_data_folder = Path('../../test_data')

line = xt.load(test_data_folder /
                        'hllhc15_thick/lhc_thick_with_knobs.json')

twng = line.madng_twiss()
det = line.get_amplitude_detuning_coefficients(num_turns=512)

xo.assert_allclose(twng.dqxdjx_nf_ng, det['det_xx'], rtol=7e-2)
xo.assert_allclose(twng.dqydjy_nf_ng, det['det_yy'], rtol=7e-2)
xo.assert_allclose(twng.dqxdjy_nf_ng, det['det_xy'], rtol=7e-2)
xo.assert_allclose(twng.dqydjx_nf_ng, det['det_yx'], rtol=7e-2)

tw = line.twiss4d()
xo.assert_allclose(tw.ddqx, twng.d2q1_nf_ng, rtol=1e-2)
xo.assert_allclose(tw.ddqy, twng.d2q2_nf_ng, rtol=1e-2)