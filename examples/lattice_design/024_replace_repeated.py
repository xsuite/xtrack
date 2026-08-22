import xtrack as xt
import numpy as np

env = xt.Environment()
l1 = env.new_line(length=9.0,
    components=[
        env.new('m', xt.Marker, at=3.0),
        env.place('m', at=6.0)])

tt1 = l1.get_table()

l2 = l1.copy(shallow=True)
l2.replace_all_repeated_elements()
tt2 = l2.get_table()

l3 = l1.copy(shallow=True)
l3.replace_all_repeated_elements(replace_generated_drifts=True)
tt3 = l3.get_table()

assert np.all(tt1.name == [
    '||drift_1::0', 'm::0', '||drift_1::1', 'm::1', '||drift_1::2',
    '_end_point'])
assert np.all(tt1.env_name == [
    '||drift_1', 'm', '||drift_1', 'm', '||drift_1',
    '_end_point'])

assert np.all(tt2.name == [
       '||drift_1::0', 'm.0', '||drift_1::1', 'm.1', '||drift_1::2',
       '_end_point'])
assert np.all(tt2.env_name == [
       '||drift_1', 'm.0', '||drift_1', 'm.1', '||drift_1',
       '_end_point'])

assert np.all(tt3.name == [
         'drift_1.0', 'm.2', 'drift_1.1', 'm.3', 'drift_1.2',
         '_end_point'])
assert np.all(tt3.env_name == [
         'drift_1.0', 'm.2', 'drift_1.1', 'm.3', 'drift_1.2',
         '_end_point'])