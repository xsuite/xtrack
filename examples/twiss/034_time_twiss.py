import xtrack as xt
import time

env = xt.load('../../test_data/hllhc15_thick/hllhc15_collider_thick.json')

line = env.lhcb1

tw = line.twiss(start='ip5', end='ip6', betx=0.15, bety=0.15)

t1 = time.perf_counter()
for _ in range(100):
    tw = line.twiss(start='ip5', end='ip6', betx=0.15, bety=0.15)
t2 = time.perf_counter()
print(f"Time taken for twiss: {t2 - t1:.6f} seconds")

