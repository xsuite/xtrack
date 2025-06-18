import numpy as np
import xtrack as xt
import xdeps as xd

LUA_VARS_PER_CHUNK = 200

MADNG_ATTR_DICT = {
    "length": "l",
    "frequency": "freq",
    "voltage": "volt",
    "hxl": "angle",
    "edge_entry_angle": "e1",
    "edge_exit_angle": "e2",
    "edge_entry_fint": "fint",
    "edge_exit_fint": "fintx",
    "edge_entry_hgap": "hgap",
    "rot_s_rad": "tilt",
}

MADNG_ATTR_IGNORE_LIST = [
    "radiation_flag", "delta_taper", "edge_entry_model", "edge_exit_model",
    "edge_entry_angle_fdown", "edge_exit_angle_fdown", "shift_x", "shift_y", "model",
    "integrator"
]

CNDICT = {
    'cavity': 'rfcavity',
    'bend': 'sbend'
}

def expr_to_mad_str(expr):

    expr_str = str(expr)

    fff = xt.line.Functions()
    for nn in fff._mathfunctions:
        expr_str = expr_str.replace(f'f.{nn}(', f'{nn}(')
        expr_str = expr_str.replace(f'f[\'{nn}\'](', f'{nn}(')

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

def madng_assignment(lhs, rhs):
    if _is_ref(rhs):
        rhs = mad_str_or_value(rhs)
    if isinstance(rhs, str):
        return f"{lhs} =\\ {rhs}"
    elif isinstance(rhs, np.ndarray):
        rhs = f"{{ {np.array2string(rhs, separator=', ')[1:-1]} }}"
        return f"{lhs} = {rhs}"
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

def to_madng_sequence(line, name='seq', mode='sequence'):
    code_str = ""
    chunk_start = "do\t -- Begin chunk\n"
    chunk_end = "end\t -- End chunk\n"
    var_lines = []
    substituted_vars = []
    for vv in line.vars.keys():
        vv_rep = vv.replace('.', '_')
        if vv_rep != vv:
            substituted_vars.append(vv)

    # Create variables
    for vv in line.vars.keys():
        if vv == '__vary_default':
            continue
        rhs = _ge(line.vars[vv])
        vv_rep = vv.replace('.', '_')
        if _is_ref(rhs):
            rhs = mad_str_or_value(rhs)
        if isinstance(rhs, str):
            for vv_sub in substituted_vars:
                rhs = rhs.replace(vv_sub, vv_sub.replace('.', '_'))
        if isinstance(rhs, str):
            vars_str = f"{vv_rep} =\\ {rhs};\n"
        else:
            vars_str = f"{vv_rep} = {rhs};\n"
        var_lines.append(vars_str)

    def _expr_to_madng_vars(expr, substituted_var_list):
        if isinstance(expr, str):
            for vv in substituted_var_list:
                expr = expr.replace(vv, vv.replace('.', '_'))
        return expr

    # Chunking variables
    for ii in range(len(var_lines)):
        if ii % LUA_VARS_PER_CHUNK == 0:
            if ii > 0:
                code_str += chunk_end
            code_str += chunk_start
        code_str += var_lines[ii]
    code_str += chunk_end

    # Create sequence elements
    tt = line.get_table()

    s_dict = {}
    el_strs = []

    for ii, nn in enumerate(tt.name[:-1]): # ignore "_end_point"
        if not(tt.isthick[ii]):
            s_dict[nn] = tt.s[ii]
        else:
            s_dict[nn] = 0.5 * (tt.s[ii] + tt.s[ii+1])

        el = line.element_dict[nn]

        class_name = el.__class__.__name__.lower()
        # class dict:
        if class_name in CNDICT.keys():
            class_name = CNDICT[class_name]

        nn_mad = nn.replace(':', '__') # : not supported in MADX names
        #nn_mad = nn.replace('$', '_')  # replace $ with _ for MAD-NG compatibility
        #nn_mad = nn.replace('.', '_')  # not needed
        el_str = f"{class_name} '{nn_mad}' {{"

        if isinstance(el, xt.Multipole):
            mult = line.element_refs[nn]

            if (len(mult.knl._value) == 1 and len(mult.ksl._value) == 1
                and mult.hxl._value == 0):
                # It is a dipole corrector
                class_name = 'kicker'
                el_str = f"{class_name} '{nn_mad}' {{"
                tokens = []
                knl_ge = _ge(mult.knl[0])
                ksl_ge = _ge(mult.ksl[0])
                len_ge = _ge(mult.length)

                tokens.append(madng_assignment('hkick', -1 * knl_ge))
                tokens.append(madng_assignment('vkick', ksl_ge))
                tokens.append(madng_assignment('lrad', len_ge))

                _handle_transforms(tokens, mult)

                for i, t in enumerate(tokens):
                    for vv_sub in substituted_vars:
                        tokens[i] = tokens[i].replace(vv_sub, vv_sub.replace('.', '_'))

                el_str += ", ".join(tokens) + ", "

            # correctors are not handled correctly!!!!
            # https://github.com/MethodicalAcceleratorDesign/MAD-X/issues/911
            # assert mult.hyl._value == 0

            else:

                tokens = []
                el_str = f"{class_name} '{nn_mad}' {{"
                knl_token, ksl_token = _knl_ksl_to_mad(mult)
                tokens.append(knl_token)
                tokens.append(ksl_token)
                lrad_ge = _ge(mult.length)
                hxl_ge = _ge(mult.hxl)
                if isinstance(lrad_ge, str):
                    for vv_sub in substituted_vars:
                        lrad_ge = lrad_ge.replace(vv_sub, vv_sub.replace('.', '_'))
                tokens.append(madng_assignment('lrad', lrad_ge))
                if isinstance(hxl_ge, str):
                    for vv_sub in substituted_vars:
                        hxl_ge = hxl_ge.replace(vv_sub, vv_sub.replace('.', '_'))
                tokens.append(mad_assignment('angle', hxl_ge))

                _handle_transforms(tokens, mult)

                for i, t in enumerate(tokens):
                    for vv_sub in substituted_vars:
                        tokens[i] = tokens[i].replace(vv_sub, vv_sub.replace('.', '_'))

                el_str += ", ".join(tokens) + ", "

        else:

            for key in el._xofields.keys():
                if key in MADNG_ATTR_IGNORE_LIST or key.startswith('_'):
                    continue
                if key in MADNG_ATTR_DICT:
                    mad_key = MADNG_ATTR_DICT[key]
                else:
                    mad_key = key

                #elem = _get_eref(line, nn)

                value = _ge(getattr(el, key))
                # if isinstance(_ge(getattr(el, key)), np.ndarray):
                #     value = _ge(getattr(el, key))
                # else:
                #     value = _ge(getattr(elem, key))
                if value is None:
                    continue
                value = madng_assignment(mad_key, value)
                if isinstance(value, str):
                    for vv_sub in substituted_vars:
                        value = value.replace(vv_sub, vv_sub.replace('.', '_'))
                el_str += f"{value}, "

        # el_str = xsuite_to_mad_conveters[el.__class__](nn, line)
        # if nn + '_tilt_entry' in line.element_dict:
        #     el_str += ", " + mad_assignment('tilt',
        #                 _ge(line.element_refs[nn + '_tilt_entry'].angle) / 180. * np.pi)

        # Misalignments
        if hasattr(el, 'shift_x') and hasattr(el, 'shift_y'):
            el_str += f"misalign =\\ {{dx={mad_str_or_value(_ge(line.ref[nn].shift_x))}, dy={mad_str_or_value(_ge(line.ref[nn].shift_y))}}}"
        else:
            el_str += "misalign =\\ {}"
        el_strs.append(el_str)

    # Chunking sequence
    seq_end = "}\n"
    chunk_count = 0
    for ii in range(len(el_strs)):
        if ii % LUA_VARS_PER_CHUNK == 0:
            if ii > 0:
                code_str += seq_end + chunk_end
            code_str += chunk_start
            code_str += f"seq_chunk_{chunk_count} = bline 'seq_chunk_{chunk_count}' {{\n"
            chunk_count += 1
        code_str += el_strs[ii]
        if ii == len(el_strs) - 1 or (ii + 1) % LUA_VARS_PER_CHUNK == 0:
            code_str += "}\n"
        else:
            code_str += "},\n"

    code_str += seq_end + chunk_end
    # create seq out of chunks

    code_str += chunk_start + f"{name} = sequence '{name}' {{ refer=centre,"

    for i in range(chunk_count):
        if i == chunk_count - 1:
            code_str += f" seq_chunk_{i} }}\n"
        else:
            code_str += f" seq_chunk_{i},"

    for i in range(chunk_count):
        code_str += f"seq_chunk_{i} = nil\n"
    code_str += chunk_end
    return code_str

    #seq_str += f"{nn_mad}: {el_str}, at={s_dict[nn]};\n"

