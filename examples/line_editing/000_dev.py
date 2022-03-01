import numpy as np
import xtrack as xt

line = xt.Line(
    elements = [xt.Drift(length=1) for _ in range(5)]
)

assert np.all(np.array([0,1,2,3,4]) == np.array(line.get_s_elements()))
assert np.all(np.array([0,1,2,3,4]) == np.array(line.get_s_elements(mode='upstream')))
assert np.all(np.array([1,2,3,4,5]) == np.array(line.get_s_elements(mode='downstream')))

assert line.get_s_position(at_elements='e3') == 3.
assert np.isscalar(line.get_s_position(at_elements='e3'))
assert len(line.get_s_position(at_elements=['e3'])) == 1
assert np.all(np.array([4,2]) == np.array(line.get_s_position(at_elements=['e4', 'e2'])))

line.insert_element(element=xt.Cavity(), name="cav", at_s=3.3)
assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5
assert line.get_s_position('cav') == 3.3