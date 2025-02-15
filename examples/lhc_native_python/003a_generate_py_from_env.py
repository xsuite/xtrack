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

    # The fields to consider are those in the dictionary, plus knl and ksl, plus
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

# Some customizations
elem_tokens['multipole']['params'].append('knl=[0,0,0,0,0,0]')

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

# Build def instruction
for nn in elem_tokens:
    out_parts = []
    out_parts.append(f'env.new("{nn}"')
    if elem_tokens[nn]['clone_parent'] is not None:
        out_parts.append(f'"{elem_tokens[nn]["clone_parent"]}"')
    else:
        out_parts.append(f'"{elem_tokens[nn]["element_type"]}"')
    out_parts += elem_tokens[nn]['diff_params']
    elem_tokens[nn]['def_instruction'] = ', '.join(out_parts) + ')'

# Sort based on hierarchy
sorted_elems = []
def _add_elem(nn):
    if elem_tokens[nn]['clone_parent'] is not None:
        _add_elem(elem_tokens[nn]['clone_parent'])
    if nn not in sorted_elems:
        sorted_elems.append(nn)
for nn in elem_tokens:
    _add_elem(nn)

# Build elem def part
elem_def_lines = []
for nn in sorted_elems:
    elem_def_lines.append(elem_tokens[nn]['def_instruction'])
elem_def_part = '\n'.join(elem_def_lines)

# builders
builder_lines = []
for lname in env.lines.keys():
    builder_lines.append(f'# Builder for line {lname}')
    bb = env.lines[lname].builder
    builder_lines.append(f'{lname} = env.new_builder(name="{lname}")')
    for cc in bb.components:
        cc_tokens=[]
        for kk, vv in cc.__dict__.items():
            if vv is None or kk == 'name':
                continue
            cc_tokens.append(f'{kk}="{vv}"')
        builder_lines.append(f'{lname}.place("{cc.name}", ' + ', '.join(cc_tokens) + ')')
    builder_lines.append(f'{lname}.build()')
    builder_lines.append('')

builder_part = '\n'.join(builder_lines)

# Variables
ttvars = env.vars.get_table()
lattice_parameters = []
for nn in ttvars.name:
    if 'element_refs' in str(env.ref[nn]._find_dependant_targets()):
        lattice_parameters.append(nn)
# Add also parameters used in the builders
pars_builder = set()
for lname in env.lines.keys():
    bb = env.lines[lname].builder
    for cc in bb.components:
        if cc.at is None:
            continue
        env.ref['__temp__'] = env.new_expr(cc.at)
        if env.ref['__temp__']._expr is None:
            continue
        vv_list = list(env.ref['__temp__']._expr._get_dependencies())
        for vv in vv_list:
            pars_builder.add(vv._formatted(formatter))
lattice_parameters += list(pars_builder)

tt_lattice_pars_all = ttvars.rows[lattice_parameters]
mask_keep = (tt_lattice_pars_all['expr'] != None) | (tt_lattice_pars_all['value'] != 0)
tt_lattice_pars = tt_lattice_pars_all.rows[mask_keep]

lines_vars = []
for nn in tt_lattice_pars.name:
    if tt_lattice_pars['expr', nn] is not None:
        lines_vars.append(f'env["{nn}"] = "{tt_lattice_pars["expr", nn]}"')
    else:
        lines_vars.append(f'env["{nn}"] = {tt_lattice_pars["value", nn]}')
lines_vars.append('')
vars_part = '\n'.join(lines_vars)


preamble = '''import xtrack as xt
env = xt.get_environment()
env.vars.default_to_zero=True

'''

postamble = '''

env.vars.default_to_zero=False

'''

with open('test.py', 'w') as ff:
    ff.write(preamble + vars_part + elem_def_part + '\n\n' + builder_part + postamble)