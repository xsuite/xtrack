import numpy as np
import xtrack as xt
import xdeps as xd

def expr_to_mad_str(expr):

    expr_str = str(expr)

    fff = xt.line.Functions()
    for nn in fff._mathfunctions:
        expr_str = expr_str.replace(f'f.{nn}(', f'{nn}(')

    expr_str = expr_str.replace("'", "")
    expr_str = expr_str.replace('"', "")

    # transform vars[...] in (...)
    while "vars[" in expr_str:
        before, after = tuple(*[expr_str.split("vars[", 1)])
        # find the corresponding closing bracket
        count = 1
        for ii, cc in enumerate(after):
            if cc == "]":
                count -= 1
            elif cc == "[":
                count += 1
            if count == 0:
                break

        expr_str = before + "(" + after[:ii] + ")" + after[ii+1:]

        expr_str = expr_str.replace("**", "^")

    return expr_str

def mad_str_or_value(var):
    if _is_ref(var):
        out = expr_to_mad_str(var)
        out = out.strip('._expr')
        return out
    else:
        return var

def mad_assignment(lhs, rhs):
    if _is_ref(rhs):
        rhs = mad_str_or_value(rhs)
    if isinstance(rhs, str):
        return f"{lhs} := {rhs}"
    else:
        return f"{lhs} = {rhs}"

_ge = xt.elements._get_expr
_is_ref = xd.refs.is_ref

def _knl_ksl_to_mad(mult):
    knl_mad = []
    ksl_mad = []
    for kl, klmad in zip([mult.knl, mult.ksl], [knl_mad, ksl_mad]):
        for ii in range(len(kl._value)):
            item = mad_str_or_value(_ge(kl[ii]))
            if not isinstance(item, str):
                item = str(item)
            klmad.append(item)
    knl_token = 'knl:={' + ','.join(knl_mad) + '}'
    ksl_token = 'ksl:={' + ','.join(ksl_mad) + '}'
    return knl_token, ksl_token

def _get_eref(line, name):
    return line.element_refs[name]

def _handle_transforms(tokens, el):
    if el.shift_x._expr is not None or el.shift_x._value:
        tokens.append(mad_assignment('dx', _ge(el.shift_x)))
    if el.shift_y._expr is not None or el.shift_y._value:
        tokens.append(mad_assignment('dy', _ge(el.shift_y)))
    if el.rot_s_rad._expr is not None or el.rot_s_rad._value:
        tokens.append(mad_assignment('tilt', _ge(el.rot_s_rad)))
    if el.shift_s._expr is not None or el.shift_s._value:
        raise NotImplementedError("shift_s is not yet supported in mad writer")

def cavity_to_madx_str(name, line):
    cav = _get_eref(line, name)
    tokens = []
    tokens.append('rfcavity')
    tokens.append(mad_assignment('freq', _ge(cav.frequency) * 1e-6))
    tokens.append(mad_assignment('volt', _ge(cav.voltage) * 1e-6))
    tokens.append(mad_assignment('lag', _ge(cav.lag) / 360.))
    _handle_transforms(tokens, cav)

    return ', '.join(tokens)

def marker_to_madx_str(name, line):
    if name.endswith('_entry'):
         parent_name = name.replace('_entry', '')
         if (parent_name in line.element_dict):
             return None
    if name.endswith('_exit'):
        parent_name = name.replace('_exit', '')
        if (parent_name in line.element_dict):
            return None
    return 'marker'

def drift_to_madx_str(name, line):
    drift = _get_eref(line, name)
    tokens = []
    tokens.append('drift')
    tokens.append(mad_assignment('l', _ge(drift.length)))
    return ', '.join(tokens)

def multipole_to_madx_str(name, line):
    mult = _get_eref(line, name)

    if (len(mult.knl._value) == 1 and len(mult.ksl._value) == 1
        and mult.hxl._value == 0):
        # It is a dipole corrector
        tokens = []
        tokens.append('kicker')
        tokens.append(mad_assignment('hkick', -1 * _ge(mult.knl[0])))
        tokens.append(mad_assignment('vkick', _ge(mult.ksl[0])))
        tokens.append(mad_assignment('lrad', _ge(mult.length)))

        _handle_transforms(tokens, mult)

        return ', '.join(tokens)

    # correctors are not handled correctly!!!!
    # https://github.com/MethodicalAcceleratorDesign/MAD-X/issues/911
    # assert mult.hyl._value == 0

    tokens = []
    tokens.append('multipole')
    knl_token, ksl_token = _knl_ksl_to_mad(mult)
    tokens.append(knl_token)
    tokens.append(ksl_token)
    tokens.append(mad_assignment('lrad', _ge(mult.length)))
    tokens.append(mad_assignment('angle', _ge(mult.hxl)))

    _handle_transforms(tokens, mult)

    return ', '.join(tokens)

def rfmultipole_to_madx_str(name, line):
    rfmult = _get_eref(line, name)

    tokens = []
    tokens.append('rfmultipole')
    knl_mad = []
    ksl_mad = []
    for kl, klmad in zip([rfmult.knl, rfmult.ksl], [knl_mad, ksl_mad]):
        for ii in range(len(kl._value)):
            item = mad_str_or_value(_ge(kl[ii]))
            if not isinstance(item, str):
                item = str(item)
            klmad.append(item)
    pnl_mad = []
    psl_mad = []
    for pp, plmad in zip([rfmult.pn, rfmult.ps], [pnl_mad, psl_mad]):
        for ii in range(len(pp._value)):
            item = mad_str_or_value(_ge(pp[ii]) / 360)
            if not isinstance(item, str):
                item = str(item)
            plmad.append(item)

    tokens.append('knl:={' + ','.join(knl_mad) + '}')
    tokens.append('ksl:={' + ','.join(ksl_mad) + '}')
    tokens.append('pnl:={' + ','.join(pnl_mad) + '}')
    tokens.append('psl:={' + ','.join(psl_mad) + '}')
    tokens.append(mad_assignment('freq', _ge(rfmult.frequency) * 1e-6))
    tokens.append(mad_assignment('volt', _ge(rfmult.voltage) * 1e-6))
    tokens.append(mad_assignment('lag', _ge(rfmult.lag) / 360.))

    _handle_transforms(tokens, rfmult)

    return ', '.join(tokens)

def dipoleedge_to_madx_str(name, line):
    raise NotImplementedError("isolated dipole edges are not yet supported")

def bend_to_madx_str(name, line, bend_type='sbend'):

    assert bend_type in ['sbend', 'rbend']

    bend = _get_eref(line, name)

    tokens = []
    tokens.append(bend_type)
    tokens.append(mad_assignment('l', _ge(bend.length)))
    tokens.append(mad_assignment('angle', _ge(bend.h) * _ge(bend.length)))
    tokens.append(mad_assignment('k0', _ge(bend.k0)))
    tokens.append(mad_assignment('e1', _ge(bend.edge_entry_angle)))
    tokens.append(mad_assignment('e2', _ge(bend.edge_exit_angle)))
    tokens.append(mad_assignment('fint', _ge(bend.edge_entry_fint)))
    tokens.append(mad_assignment('fintx', _ge(bend.edge_exit_fint)))
    tokens.append(mad_assignment('hgap', _ge(bend.edge_entry_hgap)))
    tokens.append(mad_assignment('k1', _ge(bend.k1)))
    knl_token, ksl_token = _knl_ksl_to_mad(bend)
    tokens.append(knl_token)
    tokens.append(ksl_token)

    _handle_transforms(tokens, bend)

    return ', '.join(tokens)

def rbend_to_madx_str(name, line):
    return bend_to_madx_str(name, line, bend_type='rbend')

def sextupole_to_madx_str(name, line):
    sext = _get_eref(line, name)
    tokens = []
    tokens.append('sextupole')
    tokens.append(mad_assignment('l', _ge(sext.length)))
    tokens.append(mad_assignment('k2', _ge(sext.k2)))
    tokens.append(mad_assignment('k2s', _ge(sext.k2s)))

    _handle_transforms(tokens, sext)

    return ', '.join(tokens)

def octupole_to_madx_str(name, line):
    octup = _get_eref(line, name)
    tokens = []
    tokens.append('octupole')
    tokens.append(mad_assignment('l', _ge(octup.length)))
    tokens.append(mad_assignment('k3', _ge(octup.k3)))
    tokens.append(mad_assignment('k3s', _ge(octup.k3s)))

    _handle_transforms(tokens, octup)

    return ', '.join(tokens)

def quadrupole_to_madx_str(name, line):
    quad = _get_eref(line, name)
    tokens = []
    tokens.append('quadrupole')
    tokens.append(mad_assignment('l', _ge(quad.length)))
    tokens.append(mad_assignment('k1', _ge(quad.k1)))
    tokens.append(mad_assignment('k1s', _ge(quad.k1s)))
    knl_token, ksl_token = _knl_ksl_to_mad(quad)
    tokens.append(knl_token)
    tokens.append(ksl_token)

    _handle_transforms(tokens, quad)

    return ', '.join(tokens)

def solenoid_to_madx_str(name, line):
    sol = _get_eref(line, name)
    tokens = []
    tokens.append('solenoid')
    tokens.append(mad_assignment('l', _ge(sol.length)))
    tokens.append(mad_assignment('ks', _ge(sol.ks)))
    tokens.append(mad_assignment('ksi', _ge(sol.ksi)))

    _handle_transforms(tokens, sol)

    return ', '.join(tokens)

def srotation_to_madx_str(name, line):
    raise NotImplementedError("isolated rotations are not yet supported")
    return 'marker'
    # srot = _get_eref(line, name)
    # tokens = []
    # tokens.append('srotation')
    # tokens.append(mad_assignment('angle', _ge(srot.angle)*np.pi/180.))
    # return ', '.join(tokens)

xsuite_to_mad_conveters={
    xt.Cavity: cavity_to_madx_str,
    xt.Marker: marker_to_madx_str,
    xt.Drift: drift_to_madx_str,
    xt.Multipole: multipole_to_madx_str,
    xt.DipoleEdge: dipoleedge_to_madx_str,
    xt.Bend: bend_to_madx_str,
    xt.RBend: rbend_to_madx_str,
    xt.Sextupole: sextupole_to_madx_str,
    xt.Octupole: octupole_to_madx_str,
    xt.Quadrupole: quadrupole_to_madx_str,
    xt.Solenoid: solenoid_to_madx_str,
    xt.SRotation: srotation_to_madx_str,
    xt.RFMultipole: rfmultipole_to_madx_str,
}

def to_madx_sequence(line, name='seq', mode='sequence'):
    # build variables part
    vars_str = ""
    for vv in line.vars.keys():
        if vv == '__vary_default':
            continue
        vars_str += mad_assignment(vv, _ge(line.vars[vv])) + ";\n"

    if mode =='line':
        elements_str = ""
        for nn in line.element_names:
            el = line[nn]
            el_str = xsuite_to_mad_conveters[el.__class__](nn, line)
            elements_str += f"{nn}: {el_str};\n"
        line_str = f'{name}: line=(' + ', '.join(line.element_names) + ');'
        machine_str = elements_str + line_str
    elif mode == 'sequence':
        tt = line.get_table()
        line_length = tt['s', -1]
        seq_str = f'{name}: sequence, l={line_length};\n' #, refer=entry;\n'
        # s_dict = {nn:ss for nn, ss in zip(tt.name, tt.s)}

        s_dict = {}
        tt_name = tt.name
        tt_s = tt.s
        tt_isthick = tt.isthick
        for ii in range(len(tt.name)):
            nn = tt_name[ii]
            if not(tt_isthick[ii]):
                s_dict[nn] = tt_s[ii]
            else:
                s_dict[nn] = 0.5 * (tt_s[ii] + tt_s[ii+1])

        for nn in line.element_names:
            el = line.element_dict[nn]
            el_str = xsuite_to_mad_conveters[el.__class__](nn, line)
            if nn + '_tilt_entry' in line.element_dict:
                el_str += ", " + mad_assignment('tilt',
                            _ge(line.element_refs[nn + '_tilt_entry'].angle) / 180. * np.pi)

            if el_str is None:
                continue

            nn_mad = nn.replace(':', '__') # : not supported in madx names
            seq_str += f"{nn_mad}: {el_str}, at={s_dict[nn]};\n"
        seq_str += 'endsequence;'
        machine_str = seq_str

    mad_input = vars_str + '\n' + machine_str + '\n'
    return mad_input