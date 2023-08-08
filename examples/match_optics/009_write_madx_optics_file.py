import numpy as np

import xtrack as xt
import xdeps as xd

collider = xt.Multiline.from_json('hllhc.json')
collider.build_trackers()
collider.vars.load_madx_optics_file(
    "../../test_data/hllhc15_thick/opt_round_150_1500.madx")

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

dct_expr = {}
dct_val = {}

vtable = collider.vars.get_table()
vsave = vtable.rows[
    vtable.mask[vtable.mask['acb.*'] | vtable.mask['kd.*']
                | vtable.mask['kq.*'] | vtable.mask['ks.*']]]
for nn in vsave.name:
    vv = collider.vars[nn]
    extract_val_or_madexpr(vv, dct_expr, dct_val)

out_lines = []

for nn in sorted(dct_val.keys()):
    out_lines.append(nn + ' = ' + str(dct_val[nn]) + ';')

out_lines.append('')

for nn in sorted(dct_expr.keys()):
    out_lines.append(nn + ' := ' + dct_expr[nn] + ';')

with open('opt_round_150_1500_xs.madx', 'w') as fid:
    fid.write('\n'.join(out_lines))


# Try loading optics file in MAD-X
from cpymad.madx import Madx

mad=Madx()
mad.call('../../test_data/hllhc15_thick/lhc.seq')
mad.call('../../test_data/hllhc15_thick/hllhc_sequence.madx')
mad.input('beam, sequence=lhcb1, particle=proton, energy=7000;')
mad.use('lhcb1')
mad.input('beam, sequence=lhcb2, particle=proton, energy=7000, bv=-1;')
mad.use('lhcb2')
mad.call("opt_round_150_1500_xs.madx")

mad.input('twiss, sequence=lhcb1, table=twb1')
mad.input('twiss, sequence=lhcb2, table=twb2')
twmad_b1 = xd.Table(mad.table.twb1)
twmad_b2 = xd.Table(mad.table.twb2)

assert np.isclose(twmad_b1['betx', 'ip1:1'], 0.15, rtol=1e-10, atol=0)
assert np.isclose(twmad_b1['bety', 'ip1:1'], 0.15, rtol=1e-10, atol=0)
assert np.isclose(twmad_b2['betx', 'ip1:1'], 0.15, rtol=1e-10, atol=0)
assert np.isclose(twmad_b2['bety', 'ip1:1'], 0.15, rtol=1e-10, atol=0)

import xdeps as xd

twmad_b1.rows['ip.*'].cols['betx bety x y px py'].show()
twmad_b2.rows['ip.*'].cols['betx bety x y px py'].show()

# Test orbit knobs
mad.globals.on_x8 = 100
mad.globals.on_x2 = 110

mad.input('twiss, sequence=lhcb1, table=twb1')
mad.input('twiss, sequence=lhcb2, table=twb2')
twmad_b1 = xd.Table(mad.table.twb1)
twmad_b2 = xd.Table(mad.table.twb2)

assert np.isclose(twmad_b1['px', 'ip8:1'], 100e-6, rtol=0, atol=1e-10)
assert np.isclose(twmad_b2['px', 'ip8:1'], -100e-6, rtol=0, atol=1e-10)
assert np.isclose(twmad_b1['py', 'ip2:1'], 110e-6, rtol=0, atol=1e-10)
assert np.isclose(twmad_b2['py', 'ip2:1'], -110e-6, rtol=0, atol=1e-10)

opt = collider.lhcb1.match(
    vary=
        xt.VaryList(['kqtf.b1', 'kqtd.b1', 'ksf.b1', 'ksd.b1'], step=1e-8),
    targets = [
        xt.TargetSet(qx=62.315, qy=60.325, tol=1e-6),
        xt.TargetSet(dqx=10.0, dqy=12.0, tol=1e-5)]
)