import numpy as np
import xtrack as xt
import xdeps as xd
from enum import Enum

LUA_VARS_PER_CHUNK = 200

class MadType(Enum):
    MADX = 1
    MADNG = 2

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

def _replace_var_dots_with_underscores(expr, substituted_vars):
    if isinstance(expr, str):
        # replace variables with substituted variables
        for vv_sub in substituted_vars:
            expr = expr.replace(vv_sub, vv_sub.replace('.', '_'))

    return expr

def mad_str_or_value(var):
    if _is_ref(var):
        out = expr_to_mad_str(var)
        out = out.strip('._expr')
        return out
    else:
        return var

def mad_assignment(lhs, rhs, mad_type=MadType.MADX, substituted_vars=None):
    if mad_type == MadType.MADNG:
        lhs = lhs.replace('.', '_')  # replace '.' with '_' for MADNG compatibility
    if _is_ref(rhs):
        rhs = mad_str_or_value(rhs)
        rhs = _replace_var_dots_with_underscores(rhs, substituted_vars) if mad_type == MadType.MADNG else rhs
    if isinstance(rhs, str):
        equal_str = ':=' if mad_type == MadType.MADX else '=\\'
        return f"{lhs} {equal_str} {rhs}"
    if isinstance(rhs, np.ndarray):
        rhs = f"{{ {np.array2string(rhs, separator=', ')[1:-1]} }}"
    return f"{lhs} = {rhs}"

def _handle_tokens_madng(tokens, substituted_vars):
    """
    Handle MAD-NG specific tokens, replacing variables with substituted variables.

    Parameters:
    - tokens: List of tokens to process.
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - List of processed tokens."""
    for i in range(2, len(tokens)):
        tokens[i] = _replace_var_dots_with_underscores(tokens[i], substituted_vars)
    # Merge first three tokens together in order to not have invalid commas
    if len(tokens) > 2:
        tokens[0] = tokens[0] + ' ' + tokens[1] + ' { ' + tokens[2]
        del tokens[1:3]
    else:
        tokens[0] = tokens[0] + ' ' + tokens[1] + ' { '
    return tokens

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
    knl_token = 'knl := {' + ','.join(knl_mad) + '}'
    ksl_token = 'ksl := {' + ','.join(ksl_mad) + '}'
    return knl_token, ksl_token

def _get_eref(line, name):
    return line.element_refs[name]

def _handle_transforms(tokens, el, mad_type=MadType.MADX, substituted_vars=None):
    if el.shift_x._expr is not None or el.shift_x._value:
        tokens.append(mad_assignment('dx', _ge(el.shift_x), mad_type, substituted_vars=substituted_vars))
    if el.shift_y._expr is not None or el.shift_y._value:
        tokens.append(mad_assignment('dy', _ge(el.shift_y), mad_type, substituted_vars=substituted_vars))
    if el.rot_s_rad._expr is not None or el.rot_s_rad._value:
        tokens.append(mad_assignment('tilt', _ge(el.rot_s_rad), mad_type, substituted_vars=substituted_vars))
    if el.shift_s._expr is not None or el.shift_s._value:
        raise NotImplementedError("shift_s is not yet supported in mad writer")

def cavity_to_mad_str(name, line, mad_type=MadType.MADX, substituted_vars=None):
    """
    Convert a cavity element to a MADX/MAD-NG string representation.

    Parameters:
    - name: Name of the cavity element.
    - line: The line containing the element.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the cavity in MADX/MAD-NG format.
    """

    cav = _get_eref(line, name)
    tokens = []
    tokens.append('rfcavity')
    if mad_type == MadType.MADNG:
        tokens.append(f"'{name.replace(':', '__')}'")  # replace ':' with '__' for MADNG
    tokens.append(mad_assignment('freq', _ge(cav.frequency) * 1e-6, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('volt', _ge(cav.voltage) * 1e-6, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('lag', _ge(cav.lag) / 360., mad_type, substituted_vars=substituted_vars))
    _handle_transforms(tokens, cav, mad_type=mad_type, substituted_vars=substituted_vars)

    if mad_type == MadType.MADNG:
        tokens = _handle_tokens_madng(tokens, substituted_vars)

    return ', '.join(tokens)

def marker_to_mad_str(name, line, mad_type=MadType.MADX, substituted_vars=None):
    """Convert a marker element to a MADX/MAD-NG string representation.

    Parameters:
    - name: Name of the marker element.
    - line: The line containing the element.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the marker in MADX/MAD-NG format.
    """
    if name.endswith('_entry'):
         parent_name = name.replace('_entry', '')
         if (parent_name in line.element_dict):
             return None
    if name.endswith('_exit'):
        parent_name = name.replace('_exit', '')
        if (parent_name in line.element_dict):
            return None
    if mad_type == MadType.MADX:
        return 'marker'

    tokens = []
    tokens.append('marker')
    tokens.append(name.replace(':', '__'))

    return f"marker '{name.replace(':', '__')}' {{ "

def drift_to_mad_str(name, line, mad_type=MadType.MADX, substituted_vars=None):
    """
    Convert a drift element to a MADX/MAD-NG string representation.

    Parameters:
    - name: Name of the drift element.
    - line: The line containing the element.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the drift in MADX/MAD-NG format.
    """

    drift = _get_eref(line, name)
    tokens = []
    tokens.append('drift')
    if mad_type == MadType.MADNG:
        tokens.append(f"'{name.replace(':', '__')}'")  # replace ':' with '__' for MADNG
    tokens.append(mad_assignment('l', _ge(drift.length), mad_type, substituted_vars=substituted_vars))

    if mad_type == MadType.MADNG:
        tokens = _handle_tokens_madng(tokens, substituted_vars)
    return ', '.join(tokens)

def multipole_to_mad_str(name, line, mad_type=MadType.MADX, substituted_vars=None):
    """ Convert a multipole element to a MADX/MAD-NG string representation.

    Parameters:
    - name: Name of the multipole element.
    - line: The line containing the element.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the multipole in MADX/MAD-NG format.
    """
    mult = _get_eref(line, name)

    if (len(mult.knl._value) == 1 and len(mult.ksl._value) == 1
        and mult.hxl._value == 0):
        # It is a dipole corrector
        tokens = []
        tokens.append('kicker')
        if mad_type == MadType.MADNG:
            tokens.append(f"'{name.replace(':', '__')}'")  # replace ':' with '__' for MADNG
        tokens.append(mad_assignment('hkick', -1 * _ge(mult.knl[0]), mad_type, substituted_vars=substituted_vars))
        tokens.append(mad_assignment('vkick', _ge(mult.ksl[0]), mad_type, substituted_vars=substituted_vars))
        tokens.append(mad_assignment('lrad', _ge(mult.length), mad_type, substituted_vars=substituted_vars))

        _handle_transforms(tokens, mult, mad_type=mad_type, substituted_vars=substituted_vars)

        if mad_type == MadType.MADNG:
            tokens = _handle_tokens_madng(tokens, substituted_vars)

        return ', '.join(tokens)

    # correctors are not handled correctly!!!!
    # https://github.com/MethodicalAcceleratorDesign/MAD-X/issues/911
    # assert mult.hyl._value == 0

    tokens = []
    tokens.append('multipole')
    if mad_type == MadType.MADNG:
        tokens.append(f"'{name.replace(':', '__')}'")  # replace ':' with '__' for MADNG
    knl_token, ksl_token = _knl_ksl_to_mad(mult)
    tokens.append(knl_token)
    tokens.append(ksl_token)
    tokens.append(mad_assignment('lrad', _ge(mult.length), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('angle', _ge(mult.hxl), mad_type, substituted_vars=substituted_vars))

    _handle_transforms(tokens, mult, mad_type=mad_type, substituted_vars=substituted_vars)

    if mad_type == MadType.MADNG:
        tokens = _handle_tokens_madng(tokens, substituted_vars)

    return ', '.join(tokens)

def rfmultipole_to_mad_str(name, line, mad_type=MadType.MADX, substituted_vars=None):
    """
    Convert an RF multipole element to a MADX/MAD-NG string representation.

    Parameters:
    - name: Name of the rfmultipole element.
    - line: The line containing the element.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the multipole in MADX/MAD-NG format.
    """
    rfmult = _get_eref(line, name)

    tokens = []
    tokens.append('rfmultipole')
    if mad_type == MadType.MADNG:
        tokens.append(f"'{name.replace(':', '__')}'")  # replace ':' with '__' for MADNG

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
    tokens.append(mad_assignment('freq', _ge(rfmult.frequency) * 1e-6, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('volt', _ge(rfmult.voltage) * 1e-6, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('lag', _ge(rfmult.lag) / 360., mad_type, substituted_vars=substituted_vars))

    _handle_transforms(tokens, rfmult, mad_type=mad_type, substituted_vars=substituted_vars)

    if mad_type == MadType.MADNG:
        tokens = _handle_tokens_madng(tokens, substituted_vars)

    return ', '.join(tokens)

def dipoleedge_to_mad_str(name, line, mad_type=MadType.MADX, substituted_vars=None):
    raise NotImplementedError("isolated dipole edges are not yet supported")

def bend_to_mad_str(name, line, bend_type='sbend', mad_type=MadType.MADX, substituted_vars=None):
    """ Convert a bend element to a MADX/MAD-NG string representation.

    Parameters:
    - name: Name of the bend element.
    - line: The line containing the element.
    - bend_type: Type of bend ('sbend' or 'rbend').
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the bend in MADX/MAD-NG format.
    """

    assert bend_type in ['sbend', 'rbend']

    bend = _get_eref(line, name)

    tokens = []
    tokens.append(bend_type)
    if mad_type == MadType.MADNG:
        tokens.append(f"'{name.replace(':', '__')}'")  # replace ':' with '__' for MADNG
    tokens.append(mad_assignment('l', _ge(bend.length), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('angle', _ge(bend.h) * _ge(bend.length), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k0', _ge(bend.k0), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('e1', _ge(bend.edge_entry_angle), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('e2', _ge(bend.edge_exit_angle), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('fint', _ge(bend.edge_entry_fint), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('fintx', _ge(bend.edge_exit_fint), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('hgap', _ge(bend.edge_entry_hgap), mad_type, substituted_vars=substituted_vars))
    if mad_type == MadType.MADNG:
        edge_entry_active_val = "false" if _ge(bend.edge_entry_active) == 1 else "true"
        edge_exit_active_val = "false" if _ge(bend.edge_exit_active) == 1 else "true"
        tokens.append(mad_assignment('kill_ent_fringe', edge_entry_active_val, mad_type, substituted_vars=substituted_vars))
        tokens.append(mad_assignment('kill_exi_fringe', edge_exit_active_val, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k1', _ge(bend.k1), mad_type, substituted_vars=substituted_vars))
    knl_token, ksl_token = _knl_ksl_to_mad(bend)
    tokens.append(knl_token)
    tokens.append(ksl_token)

    _handle_transforms(tokens, bend, mad_type=mad_type, substituted_vars=substituted_vars)

    if mad_type == MadType.MADNG:
        tokens = _handle_tokens_madng(tokens, substituted_vars)

    return ', '.join(tokens)

def rbend_to_mad_str(name, line, mad_type=MadType.MADX, substituted_vars=None):
    return bend_to_mad_str(name, line, bend_type='rbend',
                            mad_type=mad_type, substituted_vars=substituted_vars)

def sextupole_to_mad_str(name, line, mad_type=MadType.MADX, substituted_vars=None):
    """ Convert a sextupole element to a MADX/MAD-NG string representation.

    Parameters:
    - name: Name of the sextupole element.
    - line: The line containing the element.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the sextupole in MADX/MAD-NG format.
    """

    sext = _get_eref(line, name)
    tokens = []
    tokens.append('sextupole')
    if mad_type == MadType.MADNG:
        tokens.append(f"'{name.replace(':', '__')}'")  # replace ':' with '__' for MADNG
    tokens.append(mad_assignment('l', _ge(sext.length), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k2', _ge(sext.k2), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k2s', _ge(sext.k2s), mad_type, substituted_vars=substituted_vars))

    _handle_transforms(tokens, sext, mad_type=mad_type, substituted_vars=substituted_vars)

    if mad_type == MadType.MADNG:
        tokens = _handle_tokens_madng(tokens, substituted_vars)

    return ', '.join(tokens)

def octupole_to_mad_str(name, line, mad_type=MadType.MADX, substituted_vars=None):
    """ Convert a octupole element to a MADX/MAD-NG string representation.

    Parameters:
    - name: Name of the octupole element.
    - line: The line containing the element.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the octupole in MADX/MAD-NG format.
    """

    octup = _get_eref(line, name)
    tokens = []
    tokens.append('octupole')
    if mad_type == MadType.MADNG:
        tokens.append(f"'{name.replace(':', '__')}'")  # replace ':' with '__' for MADNG
    tokens.append(mad_assignment('l', _ge(octup.length), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k3', _ge(octup.k3), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k3s', _ge(octup.k3s), mad_type, substituted_vars=substituted_vars))

    _handle_transforms(tokens, octup, mad_type=mad_type, substituted_vars=substituted_vars)

    if mad_type == MadType.MADNG:
        tokens = _handle_tokens_madng(tokens, substituted_vars)

    return ', '.join(tokens)

def quadrupole_to_mad_str(name, line, mad_type=MadType.MADX, substituted_vars=None):
    """ Convert a quadrupole element to a MADX string representation.

    Parameters:
    - name: Name of the quadrupole element.
    - line: The line containing the element.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the quadrupole in MADX/MAD-NG format.
    """

    quad = _get_eref(line, name)
    tokens = []
    tokens.append('quadrupole')
    if mad_type == MadType.MADNG:
        tokens.append(f"'{name.replace(':', '__')}'")  # replace ':' with '__' for MADNG
    tokens.append(mad_assignment('l', _ge(quad.length), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k1', _ge(quad.k1), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k1s', _ge(quad.k1s), mad_type, substituted_vars=substituted_vars))
    knl_token, ksl_token = _knl_ksl_to_mad(quad)
    tokens.append(knl_token)
    tokens.append(ksl_token)

    _handle_transforms(tokens, quad, mad_type=mad_type, substituted_vars=substituted_vars)

    if mad_type == MadType.MADNG:
        tokens = _handle_tokens_madng(tokens, substituted_vars)

    return ', '.join(tokens)

def solenoid_to_mad_str(name, line, mad_type=MadType.MADX, substituted_vars=None):
    """ Convert a solenoid element to a MADX string representation.

    Parameters:
    - name: Name of the solenoid element.
    - line: The line containing the element.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the solenoid in MADX/MAD-NG format.
    """

    sol = _get_eref(line, name)
    tokens = []
    tokens.append('solenoid')
    if mad_type == MadType.MADNG:
        tokens.append(f"'{name.replace(':', '__')}'")  # replace ':' with '__' for MADNG
    tokens.append(mad_assignment('l', _ge(sol.length), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('ks', _ge(sol.ks), mad_type, substituted_vars=substituted_vars))

    if getattr(_ge(sol), 'ksi', 0) != 0:
        raise ValueError('Thin solenoids are not implemented.')

    _handle_transforms(tokens, sol, mad_type=mad_type, substituted_vars=substituted_vars)

    if mad_type == MadType.MADNG:
        tokens = _handle_tokens_madng(tokens, substituted_vars)

    return ', '.join(tokens)

def srotation_to_mad_str(name, line, mad_type=MadType.MADX, substituted_vars=None):
    raise NotImplementedError("isolated rotations are not yet supported")
    return 'marker'
    # srot = _get_eref(line, name)
    # tokens = []
    # tokens.append('srotation')
    # tokens.append(mad_assignment('angle', _ge(srot.angle)*np.pi/180.))
    # return ', '.join(tokens)


xsuite_to_mad_converters = {
    xt.Cavity: cavity_to_mad_str,
    xt.Marker: marker_to_mad_str,
    xt.Drift: drift_to_mad_str,
    xt.Multipole: multipole_to_mad_str,
    xt.DipoleEdge: dipoleedge_to_mad_str,
    xt.Bend: bend_to_mad_str,
    xt.RBend: rbend_to_mad_str,
    xt.Sextupole: sextupole_to_mad_str,
    xt.Octupole: octupole_to_mad_str,
    xt.Quadrupole: quadrupole_to_mad_str,
    xt.Solenoid: solenoid_to_mad_str,
    xt.UniformSolenoid: solenoid_to_mad_str,
    xt.SRotation: srotation_to_mad_str,
    xt.RFMultipole: rfmultipole_to_mad_str,
}

def to_madx_sequence(line, name='seq', mode='sequence'):
    # build variables part
    vars_str = ""
    for vv in line.vars.keys():
        if vv == '__vary_default':
            continue
        vars_str += mad_assignment(vv, _ge(line.vars[vv]), mad_type=MadType.MADX) + ";\n"

    if mode =='line':
        elements_str = ""
        for nn in line.element_names:
            el = line[nn]
            el_str = xsuite_to_mad_converters[el.__class__](nn, line, mad_type=MadType.MADX)
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
            el_str = xsuite_to_mad_converters[el.__class__](nn, line, mad_type=MadType.MADX)
            if nn + '_tilt_entry' in line.element_dict:
                el_str += ", " + mad_assignment('tilt',
                            _ge(line.element_refs[nn + '_tilt_entry'].angle) / 180. * np.pi,
                            mad_type=MadType.MADX)

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
        vars_str = mad_assignment(vv.replace('.', '_'), _ge(line.vars[vv]),
                            mad_type=MadType.MADNG, substituted_vars=substituted_vars) + "\n"
        var_lines.append(vars_str)

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

        el_str = xsuite_to_mad_converters[el.__class__](nn, line, mad_type=MadType.MADNG, substituted_vars=substituted_vars)
        if el_str is None:
            continue

        # Misalignments
        if hasattr(el, 'shift_x') and hasattr(el, 'shift_y'):
            el_str += f", misalign =\\ {{dx={mad_str_or_value(_ge(line.ref[nn].shift_x))}, dy={mad_str_or_value(_ge(line.ref[nn].shift_y))}}}"
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

    code_str += chunk_start + f"{name} = sequence '{name}' {{ refer='centre',"

    for i in range(chunk_count):
        if i == chunk_count - 1:
            code_str += f" seq_chunk_{i} }}\n"
        else:
            code_str += f" seq_chunk_{i},"

    for i in range(chunk_count):
        code_str += f"seq_chunk_{i} = nil\n"
    code_str += chunk_end
    return code_str
