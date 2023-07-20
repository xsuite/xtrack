import re
import numpy as np

import xtrack as xt
import lhc_match as lm

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

def get_mad_str_expr(var_expr):
    str_expr = str(var_expr)
    mad_str_expr = str_expr.replace("vars['", '').replace("']", '')
    return mad_str_expr

def extract_val_or_madexpr(var, dct_expr, dct_val):
    var_name = var._key
    if var._expr is not None and var_name not in dct_expr:
        dct_expr[var_name] = get_mad_str_expr(var._expr)
        for vv in var._expr._get_dependencies():
            extract_val_or_madexpr(vv, dct_expr, dct_val)
        for vv in var._find_dependant_targets():
            if str(vv).startswith('vars['):
                extract_val_or_madexpr(vv, dct_expr, dct_val)
    else:
        dct_val[var_name] = var._value

dct_expr = {}
dct_val = {}

vv = collider.vars['acbxh1.l8']
extract_val_or_madexpr(vv, dct_expr, dct_val)








