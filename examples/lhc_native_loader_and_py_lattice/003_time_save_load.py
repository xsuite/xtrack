import xtrack as xt
import time as time

t1 = time.time()
env_mad = xt.load('../../test_data/lhc_2024/lhc.seq')
t2 = time.time()
print(f'Time to load LHC from madx sequence: {t2-t1:.3f} s')

env_mad.lhcb1.regenerate_from_composer()
env_mad.lhcb2.regenerate_from_composer()

# Trash all drifts
tt = env_mad.elements.get_table()
drift_names = tt.rows['drift_.*'].name
for ii, dn in enumerate(drift_names):
    print(f'Removing drift {ii+1}/{len(drift_names)}', end='\r', flush=True)
    # I bypass the xdeps checks, I know there are no expressions in drifts
    del env_mad._element_dict[dn]


t1 = time.time()
env_mad.to_json('lhc_composers.json')
t2 = time.time()

env_mad.lhcb1.end_compose()
env_mad.lhcb2.end_compose()
env_mad.lhcb1.composer = None
env_mad.lhcb2.composer = None
t1 = time.time()
env_mad.to_json('lhc_lines.json')
t2 = time.time()
print(f'Time to save LHC lines: {t2-t1:.3f} s')

t1 = time.time()
env_composer = xt.load('lhc_composers.json')
t2 = time.time()
print(f'Time to load LHC composers: {t2-t1:.3f} s')

t1 = time.time()
env_lines = xt.load('lhc_lines.json')
t2 = time.time()
print(f'Time to load LHC lines: {t2-t1:.3f} s')