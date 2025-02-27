import xtrack as xt



fun_container = xt.line.Functions()
_functions = []
for ff in fun_container._mathfunctions.keys():
    _functions.append(ff)

def get_mad_str_expr(var_expr):
    str_expr = str(var_expr)
    mad_str_expr = str_expr.replace("vars['", '').replace("']", '')
    for ff in _functions:
        mad_str_expr = mad_str_expr.replace('f.' + ff, ff)
    return mad_str_expr

def extract_val_or_madexpr(var, dct_expr, dct_val):
    var_name = var._key
    if var_name in dct_expr or var_name in dct_val:
        return
    if var_name in _functions:
        return
    if var._expr is not None:
        dct_expr[var_name] = get_mad_str_expr(var._expr)
        for vv in var._expr._get_dependencies():
            extract_val_or_madexpr(vv, dct_expr, dct_val)
        for vv in var._find_dependant_targets():
            if str(vv).startswith('vars['):
                extract_val_or_madexpr(vv, dct_expr, dct_val)
    else:
        dct_val[var_name] = var._value


def gen_madx_optics_file_auto(collider, fname):

    dct_expr = {}
    dct_val = {}

    vtable = collider.vars.get_table()
    vsave = vtable.rows["(acb.*)|(kd.*)|(kq.*)|(ks.*)"]
    for nn in vsave.name:
        nn = str(nn) # to avoid numpy string
        vv = collider.vars[nn]
        extract_val_or_madexpr(vv, dct_expr, dct_val)

    out_lines = []

    for nn in sorted(dct_val.keys()):
        out_lines.append(nn + ' = ' + str(dct_val[nn]) + ';')

    out_lines.append('')

    for nn in sorted(dct_expr.keys()):
        out_lines.append(nn + ' := ' + dct_expr[nn] + ';')

    with open(fname, 'w') as fid:
        fid.write('\n'.join(out_lines))
