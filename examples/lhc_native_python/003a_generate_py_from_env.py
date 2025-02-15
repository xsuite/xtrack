import xtrack as xt
import xdeps as xd

env = xt.load_madx_lattice('../../test_data/lhc_2024/lhc.seq', reverse_lines=['lhcb2'])
env.vars.load_madx('../../test_data/lhc_2024/injection_optics.madx')


nn = 'mb.b30l2.b1'
nn = 'mq.30l2.b1'

nn = 'mcd.b29l2.b1'

ee = env.get(nn)
ee_ref = env.ref[nn]

formatter = xd.refs.CompactFormatter(scope=None)

arr_name = 'knl'
arr_ref = getattr(ee_ref, arr_name)
out = []
for ii, vv_ref in enumerate(arr_ref._value):
    if arr_ref[ii]._expr is not None:
        out.append(f'{arr_name}[{ii}] = "{arr_ref[ii]._expr._formatted(formatter)}"')
    else:
        out.append(f'{arr_name}[{ii}] = {arr_ref[ii]._value}')