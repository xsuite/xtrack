import numpy as np
import xtrack as xt

line0 = xt.Line(
    elements = [xt.Drift(length=1) for _ in range(5)]
)

line = line0.copy()
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
assert len(line.elements) == 7

line = line0.copy()
line.insert_element(element=xt.Drift(length=0.2), at_s=0.11, name='inserted_drift')
assert line.get_s_position('inserted_drift') == 0.11
assert len(line.elements) == 7
assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
            ['e0_part0', 'inserted_drift', 'e0_part1', 'e1', 'e2', 'e3', 'e4']))])
assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5

line = line0.copy()
line.insert_element(element=xt.Drift(length=0.2), at_s=0.95, name='inserted_drift')
assert len(line.elements) == 6
assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
            ['e0_part0', 'inserted_drift', 'e1_part1', 'e2', 'e3', 'e4']))])
assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5

line = line0.copy()
line.insert_element(element=xt.Drift(length=0.2), at_s=1.0, name='inserted_drift')
assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5
assert len(line.elements) == 6
assert np.all([nn==nnref for nn, nnref in list(zip(line.element_names,
            ['e0', 'inserted_drift', 'e1_part1', 'e2', 'e3', 'e4']))])
assert line.get_length() == line.get_s_elements(mode='downstream')[-1] == 5