import numpy as np
import xtrack as xt

line = xt.Line(
    elements = [xt.Drift(length=1) for _ in range(5)]
)

assert np.all(np.array([0,1,2,3,4]) == np.array(line.get_s_positions()))
assert np.all(np.array([0,1,2,3,4]) == np.array(line.get_s_positions(mode='upstream')))
assert np.all(np.array([1,2,3,4,5]) == np.array(line.get_s_positions(mode='downstream')))

assert line.get_s_positions(at_elements='e3') == 3.
assert np.isscalar(line.get_s_positions(at_elements='e3'))
assert len(line.get_s_positions(at_elements=['e3'])) == 1
assert np.all(np.array([4,2]) == np.array(line.get_s_positions(at_elements=['e4', 'e2'])))