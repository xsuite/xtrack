import xtrack as xt
import xdeps as xd

formatter = xd.refs.CompactFormatter(scope=None)
SKIP_PARAMS = ['order', 'model', 'edge_entry_model', 'edge_exit_model',
               'k0_from_h', 'h']

def _repr_arr_ref(arr_ref, formatter):
    out = []
    for ii, vv in enumerate(arr_ref._value):
        if arr_ref[ii]._expr is not None:
            out.append(f'"{arr_ref[ii]._expr._formatted(formatter)}"')
        else:
            out.append(f'{arr_ref[ii]._value:g}')

    # Trim trailing zeros
    while out and out[-1] == '0':
        out.pop()

    return f'[{", ".join(out)}]'

def _elem_to_tokens(env, nn, formatter):

    ee = env.get(nn)
    ee_ref = env.ref[nn]

    dd = ee.to_dict()

    params = []
    for kk in dd:
        if kk == '__class__':
            continue
        if kk in SKIP_PARAMS:
            continue
        if kk == 'knl' or kk == 'ksl':
            arr_ref = getattr(ee_ref, kk)
            vv = _repr_arr_ref(arr_ref, formatter)
            if vv != '[]':
                params.append(f'{kk}={vv}')
        elif getattr(ee_ref, kk)._expr is not None:
            params.append(f'{kk}="{getattr(ee_ref, kk)._expr._formatted(formatter)}"')
        else:
            params.append(f'{kk}={getattr(ee_ref, kk)._value:g}')

    out = {'name': nn, 'element_type': ee.__class__.__name__, 'params': params}
    return out

####################

env = xt.load_madx_lattice('../../test_data/lhc_2024/lhc.seq', reverse_lines=['lhcb2'])
env.vars.load_madx('../../test_data/lhc_2024/injection_optics.madx')


out_bend = _elem_to_tokens(env, 'mb.b30l2.b1', formatter)
out_quad = _elem_to_tokens(env, 'mq.30l2.b1', formatter)
out_dec_corr = _elem_to_tokens(env, 'mcd.b29l2.b1', formatter)

