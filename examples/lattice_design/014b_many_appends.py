import xtrack as xt
from tqdm import tqdm
import time

line = xt.load('../../test_data/hllhc15_thick/lhc_thick_with_knobs.json')

n_markers = 1000
t1 = time.time()
for nn in tqdm(range(n_markers)):
    line.append(f'm{nn}', xt.Marker())
t2 = time.time()
print(f'Time taken to append {n_markers} markers: {t2-t1:.2f} seconds')