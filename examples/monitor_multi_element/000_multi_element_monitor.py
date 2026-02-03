import xtrack as xt
import numpy as np

# TODO:
# - Add new monitor to pre-compiled kernels
# - Forbid backtrack for now
# - Forbid collective mode for now
# - Forbid GPU for now
# - Need to test with and without progress bar...

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

tt = line.get_table()
tt_obs = tt.rows.match(name='bpm.*')
tt_obs.name # is ['bpmw.4r7.b1', 'bpmwe.4r7.b1', 'bpmw.5r7.b1', ...,

p = xt.Particles(p0c=7e12, x=1e-6*np.arange(20),
                           delta=0
)
line.track(p, num_turns=10, multi_element_monitor_at=tt_obs.name)