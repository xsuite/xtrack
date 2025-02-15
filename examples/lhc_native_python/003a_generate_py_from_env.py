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

    # The fileds to consider are those in the dictionary, plus knl and ksl, plus 
    # anything that has an expression
    fields = list(ee.to_dict().keys())
    if hasattr(ee, 'knl'):
        fields += ['knl']
    if hasattr(ee, 'ksl'):
        fields += ['ksl']

    tt = env[nn].get_table()
    for kk in tt.name:
        if tt['expr', kk] is not None and tt['expr', kk] != 'None':
            fields.append(kk)

    params = []
    for kk in fields:
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

    out = {'name': nn, 'element_type': ee.__class__.__name__, 'params': params,
           'clone_parent': getattr(ee, 'clone_parent', None)}
    return out

####################

env = xt.load_madx_lattice('../../test_data/lhc_2024/lhc.seq', reverse_lines=['lhcb2'])

# Test a few
out_bend = _elem_to_tokens(env, 'mb.b30l2.b1', formatter)
out_quad = _elem_to_tokens(env, 'mq.30l2.b1', formatter)
out_dec_corr = _elem_to_tokens(env, 'mcd.b29l2.b1', formatter)

all_elems = []
for lname in env.lines.keys():
    ll = env.lines[lname]
    bb = ll.builder
    all_elems += [cc.name for cc in bb.components]

elem_tokens = {}
for nn in all_elems:
    elem_tokens[nn] = _elem_to_tokens(env, nn, formatter)

while True:
    added = False
    for nn in list(elem_tokens.keys()):
        if elem_tokens[nn]['clone_parent'] is not None:
            parent_name = elem_tokens[nn]['clone_parent']
            if parent_name not in elem_tokens:
                elem_tokens[parent_name] = _elem_to_tokens(env, parent_name, formatter)
                added = True
    if not added:
        break

# populate diff params
for nn in elem_tokens:
    diff_params = []
    if elem_tokens[nn]['clone_parent'] is not None:
        parent_name = elem_tokens[nn]['clone_parent']
        parent_params = elem_tokens[parent_name]['params']
        elem_params = set(elem_tokens[nn]['params'])
        for pp in elem_params:
            if pp not in parent_params:
                diff_params.append(pp)
    else:
        diff_params = elem_tokens[nn]['params']
    elem_tokens[nn]['diff_params'] = diff_params

# Sort based on hierarchy
sorted_elems = []
def _add_elem(nn):
    if elem_tokens[nn]['clone_parent'] is not None:
        _add_elem(elem_tokens[nn]['clone_parent'])
    if nn not in sorted_elems:
        sorted_elems.append(nn)
for nn in elem_tokens:
    _add_elem(nn)