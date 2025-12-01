import xtrack as xt
import xdeps as xd
import numpy as np

from xtrack.mad_parser.loader import CONSTANTS

formatter = xd.refs.CompactFormatter(scope=None)
SKIP_PARAMS = ['order', 'model', '_edge_entry_model', '_edge_exit_model',
               'h']


def _repr_arr_ref(arr_ref, formatter):
    out = []
    for ii, vv in enumerate(arr_ref._value):
        if arr_ref[ii]._expr is not None:
            out.append(f'"{arr_ref[ii]._expr._formatted(formatter)}"')
        else:
            out.append(f'{arr_ref[ii]._value:g}')

    # Trim trailing zeros
    while out and (out[-1] == '0' or out[-1] == '-0'):
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
        if kk in fields:
            continue
        if tt['expr', kk] is not None and tt['expr', kk] != 'None':
            fields.append(kk)

    if isinstance(ee, xt.RBend):
        assert 'length_straight' in fields
        fields = [ff for ff in fields if ff != 'length']

    params = []
    for kk in fields:
        if kk == '__class__':
            continue
        if kk in SKIP_PARAMS:
            continue
        if kk == 'extra':
            continue
        if kk == 'prototype':
            continue
        if kk == 'k0_from_h' and not ee.k0_from_h:
            continue
        if kk == 'knl' or kk == 'ksl':
            arr_ref = getattr(ee_ref, kk)
            vv = _repr_arr_ref(arr_ref, formatter)
            if vv != '[]':
                params.append(f'{kk}={vv}')
        elif getattr(ee_ref, kk)._expr is not None:
            params.append(f'{kk}="{getattr(ee_ref, kk)._expr._formatted(formatter)}"')
        else:
            vv = getattr(ee_ref, kk)._value
            if isinstance(vv, str):
                params.append(f'{kk}="{vv}"')
            else:
                params.append(f'{kk}={vv:.20g}')

    extra_params = None
    if hasattr(ee, 'extra') and ee.extra is not None:
        extra_params = {}
        for kextra, vextra in ee.extra.items():
            if ee_ref.extra[kextra]._expr is not None:
                expr = ee_ref.extra[kextra]._expr._formatted(formatter)
                extra_params[kextra] = f'"{kextra}": expr("{expr}")'
            elif isinstance(vextra, str):
                extra_params[kextra] = f'"{kextra}": "{vextra}"'
            else:
                extra_params[kextra] = f'"{kextra}": {vextra}'

    out = {'name': nn, 'element_type': ee.__class__.__name__, 'params': params,
           'prototype': getattr(ee, 'prototype', None),
           'extra': extra_params}
    return out

def write_py_lattice_file(env, output_fname):

    file_content = gen_py_lattice(env)

    with open(output_fname, 'w') as ff:
        ff.write(file_content)

def gen_py_lattice(env):

    ###################
    # Handle elements #
    ###################

    all_elems = []
    for lname in env.lines.keys():
        ll = env.lines[lname]
        bb = ll.builder.flatten()
        all_elems += [cc.name for cc in bb.components]

    elem_tokens = {}
    for nn in all_elems:
        elem_tokens[nn] = _elem_to_tokens(env, nn, formatter)

    while True:
        added = False
        for nn in list(elem_tokens.keys()):
            if elem_tokens[nn]['prototype'] is not None:
                parent_name = elem_tokens[nn]['prototype']
                if parent_name not in elem_tokens:
                    elem_tokens[parent_name] = _elem_to_tokens(env, parent_name, formatter)
                    added = True
        if not added:
            break

    # Some customizations
    if 'multipole' in elem_tokens:
        elem_tokens['multipole']['params'].append('knl=[0,0,0,0,0,0]')

    # populate diff params
    for nn in elem_tokens:
        diff_params = []
        if elem_tokens[nn]['prototype'] is not None:
            parent_name = elem_tokens[nn]['prototype']
            parent_params = elem_tokens[parent_name]['params']
            elem_params = set(elem_tokens[nn]['params'])
            for pp in elem_params:
                if pp not in parent_params:
                    diff_params.append(pp)
            if elem_tokens[nn]['extra'] is not None:
                diff_extra = []
                parent_extra = elem_tokens[parent_name]['extra']
                for kk in elem_tokens[nn]['extra']:
                    if (parent_extra is None) or (kk not in parent_extra) or \
                       (elem_tokens[nn]['extra'][kk] != parent_extra[kk]):
                        diff_extra.append(elem_tokens[nn]["extra"][kk])
        else:
            diff_params = elem_tokens[nn]['params']
            if elem_tokens[nn]['extra'] is not None:
                diff_extra = []
                for kk in elem_tokens[nn]['extra']:
                    diff_extra.append(elem_tokens[nn]["extra"][kk])

        if elem_tokens[nn]['extra'] is not None and len(diff_extra) > 0:
            diff_params.append('extra={' + ', '.join(diff_extra) + '}')

        elem_tokens[nn]['diff_params'] = diff_params

    # Build def instruction
    for nn in elem_tokens:
        out_parts = []
        out_parts.append(f'env.new("{nn}"')
        if elem_tokens[nn]['prototype'] is not None:
            out_parts.append(f'"{elem_tokens[nn]["prototype"]}"')
        else:
            out_parts.append(f'"{elem_tokens[nn]["element_type"]}"')
        if len(elem_tokens[nn]['diff_params']) > 0:
            out_parts += elem_tokens[nn]['diff_params']
        elem_tokens[nn]['def_instruction'] = ', '.join(out_parts) + ')'

    # Sort based on hierarchy
    sorted_elems = []
    def _add_elem(nn):
        if elem_tokens[nn]['prototype'] is not None:
            _add_elem(elem_tokens[nn]['prototype'])
        if nn not in sorted_elems:
            sorted_elems.append(nn)
    for nn in elem_tokens:
        _add_elem(nn)

    tt_edefs = xt.Table({
        'name': np.array(sorted_elems),
        'element_type': np.array([elem_tokens[nn]['element_type'] for nn in sorted_elems]),
        'prototype': np.array([elem_tokens[nn]['prototype'] for nn in sorted_elems]),
        'def_instruction': np.array([elem_tokens[nn]['def_instruction'] for nn in sorted_elems]),
    })

    tt_gen0 = tt_edefs.rows[tt_edefs['prototype'] == None]
    tt_gen0.gen_name = 'Xsuite types'

    generation_tree = [{'Xsuite types': tt_gen0}]
    while True:
        print(f'Generation {len(generation_tree)}')
        last_gen = generation_tree[-1]
        names_last_gen = []
        for nn in last_gen.keys():
            names_last_gen += list(last_gen[nn]['name'])
        this_gen = {}
        added = False
        for nn in names_last_gen:
            tt_gen = tt_edefs.rows[tt_edefs['prototype'] == nn]
            if len(tt_gen) > 0:
                tt_gen.gen_name = nn
                this_gen[nn] = tt_gen
                added = True
        if added:
            generation_tree.append(this_gen)
        else:
            break

    # Flatten generations
    generations = []
    for gg in generation_tree:
        for nn in gg.keys():
            generations.append(gg[nn])

    # Build elem def part (with generations)
    elem_def_lines = []
    for tt_gen in generations:
        elem_def_lines.append(f'\n# Elements of type: {tt_gen.gen_name}')
        for nn in tt_gen.name:
            elem_def_lines.append(elem_tokens[nn]['def_instruction'])


    # # Build elem def part (no generations)
    # elem_def_lines = []
    # for nn in sorted_elems:
    #     elem_def_lines.append(elem_tokens[nn]['def_instruction'])

    elem_def_part = '\n'.join(elem_def_lines)


    ###################
    # Handle builders #
    ###################

    builder_lines = []
    for lname in env.lines.keys():
        builder_lines.append(f'# Line: {lname}')
        bb = env.lines[lname].builder.flatten()
        mirror_token=''
        if bb.mirror:
            mirror_token=f', mirror={bb.mirror}'
        builder_lines.append(f'{lname} = env.new_line(name="{lname}", compose=True{mirror_token})')
        builder_lines.append(f'{lname} = env["{lname}"]')
        for cc in bb.components:
            cc_tokens=[]
            for kk, vv in cc.__dict__.items():
                if vv is None or kk == 'name':
                    continue
                if hasattr(vv, '_expr'): # has expression
                    # Get string representation of expression
                    env['__temp__'] = vv
                    vv = env.ref["__temp__"]._expr._formatted(formatter)
                cc_tokens.append(f'{kk}="{vv}"')
            builder_lines.append(f'{lname}.place("{cc.name}", ' + ', '.join(cc_tokens) + ')')
        builder_lines.append(f'{lname}.end_compose()')
        builder_lines.append('')

    builder_part = '\n'.join(builder_lines)

    ####################
    # Handle variables #
    ####################

    tt_lattice_pars_all = env.vars.get_table()

    mask_keep = (tt_lattice_pars_all['expr'] != None) | (tt_lattice_pars_all['value'] != 0)
    tt_lattice_pars = tt_lattice_pars_all.rows[mask_keep]

    lines_vars = []
    for nn in tt_lattice_pars.name:
        if nn == '__temp__':
            continue
        if tt_lattice_pars['expr', nn] is not None:
            lines_vars.append(f'env["{nn}"] = "{tt_lattice_pars["expr", nn]}"')
        else:
            lines_vars.append(f'env["{nn}"] = {tt_lattice_pars["value", nn]}')
    lines_vars.append('')
    vars_part = '\n'.join(lines_vars)

    #####################
    # Assemble the file #
    #####################

    preamble = '\n'.join([
    'import xtrack as xt',
    'env = xt.get_environment()',
    'env.vars.default_to_zero=True',
    'expr = env.new_expr # used in extra, where expressions cannot be defined as strings',
    ''
    ])

    postamble = '\n'.join([
    '',
    'env.vars.default_to_zero=False',
    ''
    ])

    file_content = '\n'.join([
        preamble,
    '#############',
    '# Variables #',
    '#############',
    '',
    vars_part,
    '############',
    '# Elements #',
    '############',
    elem_def_part,
    '',
    '##############',
    '# Beam lines #',
    '##############',
    '',
    builder_part,
    postamble])

    return file_content


