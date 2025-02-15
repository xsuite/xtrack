import xtrack as xt
import xdeps as xd

def _repr_arr_ref(arr_ref, formatter):
    out = []
    for ii, vv_ref in enumerate(arr_ref._value):
        if arr_ref[ii]._expr is not None:
            out.append(f'"{arr_ref[ii]._expr._formatted(formatter)}"')
        else:
            out.append(f'{arr_ref[ii]._value:g}')

    # Trim trailing zeros
    while out and out[-1] == '0':
        out.pop()

    return f'[{", ".join(out)}]'

####################

env = xt.load_madx_lattice('../../test_data/lhc_2024/lhc.seq', reverse_lines=['lhcb2'])
env.vars.load_madx('../../test_data/lhc_2024/injection_optics.madx')


nn = 'mb.b30l2.b1'
nn = 'mq.30l2.b1'

nn = 'mcd.b29l2.b1'

ee = env.get(nn)
ee_ref = env.ref[nn]

formatter = xd.refs.CompactFormatter(scope=None)

dd = ee.to_dict()

params = []
for kk in dd:
    if kk == '__class__':
        continue
    if kk == 'knl' or kk == 'ksl':
        arr_ref = getattr(ee_ref, kk)
        vv = _repr_arr_ref(arr_ref, formatter)
        if vv != '[]':
            params.append(f'{kk}={vv}')
    elif ee_ref[kk]._expr is not None:
        params.append(f'{kk}={getattr(ee_ref, kk)._expr._formatted(formatter)}')
    else:
        params.append(f'{kk}={getattr(ee_ref, kk)._value:g}')


arr_name = 'knl'
arr_ref = getattr(ee_ref, arr_name)
knl_str = _repr_arr_ref(arr_ref, formatter)

