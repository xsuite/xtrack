import xtrack as xt
from pathlib import Path

line_path = (Path(xt.__file__).parent / '../test_data/fcc_ee/fccee_h_thin.json').absolute()

line = xt.Line.from_json(line_path)
line.build_tracker()

import time
start_time = time.time()
tw = line.twiss()
end_time = time.time()
print(f"Vanilla twiss: {end_time - start_time} seconds")

line.configure_radiation(model='mean')
line.build_tracker()
line.twiss()

start_time = time.time()
tw = line.twiss()
end_time = time.time()
print(f"With radiation twiss: {end_time - start_time} seconds")

line.twiss(eneloss_and_damping=True)
start_time = time.time()
tw = line.twiss(eneloss_and_damping=True)
end_time = time.time()
print(f"Energy loss and damping twiss: {end_time - start_time} seconds")
