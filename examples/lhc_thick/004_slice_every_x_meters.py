import matplotlib.pyplot as plt
from pathlib import Path
from time import time

import xtrack as xt
import numpy as np

all_num_cuts = np.logspace(np.log2(10), np.log2(50_000), base=2, num=15)
time_to_cut = []

line0 = xt.Line.from_json(
    Path(__file__).parent /
    '../../test_data/hllhc15_thick/lhc_thick_with_knobs.json'
)

for num_cuts in all_num_cuts:
    ss = np.linspace(0, line0.get_length(), num=int(num_cuts))
    line = line0.copy()
    start = time()
    line.cut_at_s(s=list(ss))
    end = time()
    time_to_cut.append(end - start)

fig, ax = plt.subplots()
ax.plot(all_num_cuts, time_to_cut)
ax.set_title("Slicing time vs number of cuts")
ax.set_ylabel('time [s]')
ax.set_xlabel('number of cuts [1]')
plt.show()
