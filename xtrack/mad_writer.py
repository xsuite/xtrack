import numpy as np
import xtrack as xt
from enum import Enum

from xdeps.refs import is_ref
from xtrack.functions import Functions


LUA_VARS_PER_CHUNK = 200

class MadType(Enum):
    MADX = 1
    MADNG = 2

def expr_to_mad_str(expr):

    expr_str = str(expr)

    fff = Functions()
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
    if is_ref(var):
        out = expr_to_mad_str(var)
        out = out.strip('._expr')
        return out
    else:
        return var

def mad_assignment(lhs, rhs, mad_type=MadType.MADX, substituted_vars=None):
    if mad_type == MadType.MADNG:
        lhs = lhs.replace('.', '_')  # replace '.' with '_' for MADNG compatibility
    if is_ref(rhs):
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

def _knl_ksl_to_mad(mult):
    weight = 1
    if hasattr(_ge(mult), '_parent'):
        weight = _ge(mult.weight)
        mult = mult._parent

    knl_mad = []
    ksl_mad = []
    for kl, klmad in zip([mult.knl, mult.ksl], [knl_mad, ksl_mad]):
        for ii in range(len(kl._value)):
            item = mad_str_or_value(_ge(kl[ii]) * weight)
            if not isinstance(item, str):
                item = str(item)
            klmad.append(item)
    knl_token = 'knl := {' + ','.join(knl_mad) + '}'
    ksl_token = 'ksl := {' + ','.join(ksl_mad) + '}'
    return knl_token, ksl_token

def _get_eref(line, name):
    return line.ref.elements[name]

def _handle_transforms(tokens, el_ref, mad_type=MadType.MADX, substituted_vars=None):
    def _defined_and_nonzero(field_name):
        el_instance = el_ref._value
        if not hasattr(el_instance, field_name):
            return False
        field = getattr(el_ref, field_name)
        return field._expr is not None or field._value != 0

    if _defined_and_nonzero('shift_x'):
        tokens.append(mad_assignment('dx', _ge(el_ref.shift_x), mad_type, substituted_vars=substituted_vars))
    if _defined_and_nonzero('shift_y'):
        tokens.append(mad_assignment('dy', _ge(el_ref.shift_y), mad_type, substituted_vars=substituted_vars))
    if _defined_and_nonzero('rot_s_rad'):
        tokens.append(mad_assignment('tilt', _ge(el_ref.rot_s_rad), mad_type, substituted_vars=substituted_vars))
    if _defined_and_nonzero('shift_s'):
        raise NotImplementedError("shift_s is not yet supported in mad writer")

def cavity_to_mad_str(eref, mad_type=MadType.MADX, substituted_vars=None):
    """
    Convert a cavity element to a MADX/MAD-NG string representation.

    Parameters:
    - eref: The element reference.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the cavity in MADX/MAD-NG format.
    """

    weight = 1
    if hasattr(_ge(eref), '_parent'):
        weight = _ge(eref.weight)
        eref = eref._parent

    tokens = []
    tokens.append('rfcavity')
    tokens.append(mad_assignment('freq', _ge(eref.frequency) * 1e-6, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('harmon', _ge(eref.harmonic), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('volt', _ge(eref.voltage) * 1e-6, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('lag', _ge(eref.lag) / 360., mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('l', _ge(eref.length) * weight, mad_type, substituted_vars=substituted_vars))

    return tokens

def crabcavity_to_mad_str(eref, mad_type=MadType.MADX, substituted_vars=None):
    """
    Convert a cavity element to a MADX/MAD-NG string representation.

    Parameters:
    - eref: The element reference.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the cavity in MADX/MAD-NG format.
    """

    tokens = []
    tokens.append('crabcavity')
    tokens.append(mad_assignment('freq', _ge(eref.frequency) * 1e-6, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('volt', _ge(eref.crab_voltage) * 1e-6, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('lag', _ge(eref.lag) / 360., mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('l', _ge(eref.length), mad_type, substituted_vars=substituted_vars))

    return tokens

def marker_to_mad_str(name, line, mad_type=MadType.MADX, substituted_vars=None):
    """Convert a marker element to a MADX/MAD-NG string representation.

    Parameters:
    - eref: The element reference.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the marker in MADX/MAD-NG format.
    """
    # if name.endswith('_entry'):
    #      parent_name = name.replace('_entry', '')
    #      if (parent_name in line._element_dict):
    #          return None
    # if name.endswith('_exit'):
    #     parent_name = name.replace('_exit', '')
    #     if (parent_name in line._element_dict):
    #         return None
    if mad_type == MadType.MADX:
        return 'marker'

    tokens = []
    tokens.append('marker')
    tokens.append(name.replace(':', '__'))

    return f"marker '{name.replace(':', '__')}' {{ "

def drift_to_mad_str(eref, mad_type=MadType.MADX, substituted_vars=None):
    """
    Convert a drift element to a MADX/MAD-NG string representation.

    Parameters:
    - eref: The element reference.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the drift in MADX/MAD-NG format.
    """

    tokens = []
    tokens.append('drift')
    tokens.append(mad_assignment('l', _ge(eref.length), mad_type, substituted_vars=substituted_vars))

    return tokens

def drift_slice_to_mad_str(eref, mad_type=MadType.MADX, substituted_vars=None):
    """
    Convert a drift element to a MADX/MAD-NG string representation.

    Parameters:
    - eref: The element reference.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the drift in MADX/MAD-NG format.
    """

    tokens = []
    tokens.append('drift')
    tokens.append(mad_assignment('l', (float(eref._parent.length._value)
                                     * float(eref.weight._value)),
                                 mad_type, substituted_vars=substituted_vars))

    return tokens

def multipole_to_mad_str(eref, mad_type=MadType.MADX, substituted_vars=None):
    """ Convert a multipole element to a MADX/MAD-NG string representation.

    Parameters:
    - eref: The element reference.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the multipole in MADX/MAD-NG format.
    """

    weight = 1
    if hasattr(_ge(eref), '_parent'):
        weight = _ge(eref.weight)
        eref = eref._parent

    # Special case for kicker
    if (len(eref.knl._value) == 1 and len(eref.ksl._value) == 1
        and (not hasattr(_ge(eref), 'hxl') or eref.hxl._value == 0)):
        # It is a dipole corrector
        tokens = []
        tokens.append('kicker')
        tokens.append(mad_assignment('hkick', -1 * _ge(eref.knl[0]) * weight, mad_type, substituted_vars=substituted_vars))
        tokens.append(mad_assignment('vkick', _ge(eref.ksl[0]) * weight, mad_type, substituted_vars=substituted_vars))
        if not eref.isthick._value or eref.length._value == 0:
            tokens.append(mad_assignment('lrad', _ge(eref.length) * weight, mad_type, substituted_vars=substituted_vars))
        else:
            tokens.append(mad_assignment('l', _ge(eref.length) * weight, mad_type, substituted_vars=substituted_vars))

        return tokens

    # correctors are not handled correctly!!!!
    # https://github.com/MethodicalAcceleratorDesign/MAD-X/issues/911
    # assert mult.hyl._value == 0

    if not hasattr(_ge(eref), 'hxl') or (weight < 1e-14) or (not eref.isthick._value or eref.length._value == 0):

        tokens = []
        tokens.append('multipole')
        knl_token, ksl_token = _knl_ksl_to_mad(eref)
        tokens.append(knl_token)
        tokens.append(ksl_token)
        tokens.append(mad_assignment('lrad', _ge(eref.length) * weight, mad_type, substituted_vars=substituted_vars))
        if hasattr(_ge(eref), 'hxl') and eref.hxl._value != 0:
            tokens.append(mad_assignment('angle', _ge(eref.hxl) * weight, mad_type, substituted_vars=substituted_vars))
        else:
            tokens.append(mad_assignment('angle', 0, mad_type, substituted_vars=substituted_vars))

        return tokens

    else:
        assert eref.hxl._value == 0, "Thick multipoles with hxl not supported"
        tokens = []
        tokens.append('sbend')
        knl_token, ksl_token = _knl_ksl_to_mad(eref)
        tokens.append(knl_token)
        tokens.append(ksl_token)
        tokens.append(mad_assignment('l', _ge(eref.length) * weight, mad_type, substituted_vars=substituted_vars))

        return tokens

def rfmultipole_to_mad_str(eref, mad_type=MadType.MADX, substituted_vars=None):
    """
    Convert an RF multipole element to a MADX/MAD-NG string representation.

    Parameters:
    - eref: The element reference.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the multipole in MADX/MAD-NG format.
    """
    weight = 1
    if hasattr(_ge(eref), '_parent'):
        weight = _ge(eref.weight)
        eref = eref._parent

    tokens = []
    tokens.append('rfmultipole')

    knl_mad = []
    ksl_mad = []
    for kl, klmad in zip([eref.knl, eref.ksl], [knl_mad, ksl_mad]):
        for ii in range(len(kl._value)):
            item = mad_str_or_value(_ge(kl[ii]) * weight)
            if not isinstance(item, str):
                item = str(item)
            klmad.append(item)
    pnl_mad = []
    psl_mad = []
    for pp, plmad in zip([eref.pn, eref.ps], [pnl_mad, psl_mad]):
        for ii in range(len(pp._value)):
            item = mad_str_or_value(_ge(pp[ii]) * weight / 360) # TODO: not sure here
            if not isinstance(item, str):
                item = str(item)
            plmad.append(item)

    tokens.append('knl:={' + ','.join(knl_mad) + '}')
    tokens.append('ksl:={' + ','.join(ksl_mad) + '}')
    tokens.append('pnl:={' + ','.join(pnl_mad) + '}')
    tokens.append('psl:={' + ','.join(psl_mad) + '}')
    tokens.append(mad_assignment('freq', _ge(eref.frequency) * 1e-6, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('volt', _ge(eref.voltage) * 1e-6, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('lag', _ge(eref.lag) / 360., mad_type, substituted_vars=substituted_vars))

    return tokens

def dipoleedge_to_mad_str(eref, mad_type=MadType.MADX, substituted_vars=None):
    raise NotImplementedError("isolated dipole edges are not yet supported")

def bend_to_mad_str(eref, bend_type='sbend', mad_type=MadType.MADX, substituted_vars=None):
    """ Convert a bend element to a MADX/MAD-NG string representation.

    Parameters:
    - eref: The element reference.
    - bend_type: Type of bend ('sbend' or 'rbend').
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the bend in MADX/MAD-NG format.
    """

    weight = 1
    if hasattr(_ge(eref), '_parent'):
        weight = _ge(eref.weight)
        weight = 1e-10 if weight < 1e-10 else weight
        eref = eref._parent

    assert bend_type in ['sbend', 'rbend']

    tokens = []
    tokens.append(bend_type)
    if bend_type == 'sbend' or mad_type == MadType.MADNG: # in MAD-NG all bends use the arc length
        tokens.append(mad_assignment('l', _ge(eref.length) * weight, mad_type, substituted_vars=substituted_vars))
    elif bend_type == 'rbend':
        tokens.append(mad_assignment('l', _ge(eref.length_straight) * weight, mad_type, substituted_vars=substituted_vars))
    else:
        raise ValueError(f"bend_type {bend_type} not recognized")
    tokens.append(mad_assignment('angle', _ge(eref.h) * _ge(eref.length) * weight, mad_type, substituted_vars=substituted_vars))
    if not eref.k0_from_h._value: 
        tokens.append(mad_assignment('k0', _ge(eref.k0), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('e1', _ge(eref.edge_entry_angle), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('e2', _ge(eref.edge_exit_angle), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('fint', _ge(eref.edge_entry_fint), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('fintx', _ge(eref.edge_exit_fint), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('hgap', _ge(eref.edge_entry_hgap), mad_type, substituted_vars=substituted_vars))
    if mad_type == MadType.MADNG:
        edge_entry_active_val = "false" if _ge(eref.edge_entry_active) == 1 else "true"
        edge_exit_active_val = "false" if _ge(eref.edge_exit_active) == 1 else "true"
        tokens.append(mad_assignment('kill_ent_fringe', edge_entry_active_val, mad_type, substituted_vars=substituted_vars))
        tokens.append(mad_assignment('kill_exi_fringe', edge_exit_active_val, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k1', _ge(eref.k1), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k2', _ge(eref.k2), mad_type, substituted_vars=substituted_vars))
    knl_token, ksl_token = _knl_ksl_to_mad(eref)
    tokens.append(knl_token)
    tokens.append(ksl_token)

    return tokens

def rbend_to_mad_str(eref, mad_type=MadType.MADX, substituted_vars=None):
    return bend_to_mad_str(eref, bend_type='rbend',
                            mad_type=mad_type, substituted_vars=substituted_vars)

def sextupole_to_mad_str(eref, mad_type=MadType.MADX, substituted_vars=None):
    """ Convert a sextupole element to a MADX/MAD-NG string representation.

    Parameters:
    - eref: The element reference.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the sextupole in MADX/MAD-NG format.
    """

    weight = 1
    if hasattr(_ge(eref), '_parent'):
        weight = _ge(eref.weight)
        weight = 1e-10 if weight < 1e-10 else weight
        eref = eref._parent

    tokens = []
    tokens.append('sextupole')
    tokens.append(mad_assignment('l', _ge(eref.length) * weight, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k2', _ge(eref.k2), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k2s', _ge(eref.k2s), mad_type, substituted_vars=substituted_vars))

    return tokens

def octupole_to_mad_str(eref, mad_type=MadType.MADX, substituted_vars=None):
    """ Convert a octupole element to a MADX/MAD-NG string representation.

    Parameters:
    - eref: The element reference.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the octupole in MADX/MAD-NG format.
    """

    weight = 1
    if hasattr(_ge(eref), '_parent'):
        weight = _ge(eref.weight)
        weight = 1e-10 if weight < 1e-10 else weight
        eref = eref._parent

    tokens = []
    tokens.append('octupole')
    tokens.append(mad_assignment('l', _ge(eref.length) * weight, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k3', _ge(eref.k3), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k3s', _ge(eref.k3s), mad_type, substituted_vars=substituted_vars))

    return tokens

def quadrupole_to_mad_str(eref, mad_type=MadType.MADX, substituted_vars=None):
    """ Convert a quadrupole element to a MADX string representation.

    Parameters:
    - eref: The element reference.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the quadrupole in MADX/MAD-NG format.
    """

    weight = 1
    if hasattr(_ge(eref), '_parent'):
        weight = _ge(eref.weight)
        weight = 1e-10 if weight < 1e-10 else weight
        eref = eref._parent

    tokens = []
    tokens.append('quadrupole')
    tokens.append(mad_assignment('l', _ge(eref.length) * weight, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k1', _ge(eref.k1), mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('k1s', _ge(eref.k1s), mad_type, substituted_vars=substituted_vars))
    knl_token, ksl_token = _knl_ksl_to_mad(eref)
    tokens.append(knl_token)
    tokens.append(ksl_token)

    return tokens

def solenoid_to_mad_str(eref, mad_type=MadType.MADX, substituted_vars=None):
    """ Convert a solenoid element to a MADX string representation.

    Parameters:
    - eref: The element reference.
    - mad_type: Type of MAD (MADX or MADNG).
    - substituted_vars: List of substituted variables for MADNG.

    Returns:
    - A string representation of the solenoid in MADX/MAD-NG format.
    """

    weight = 1
    if hasattr(_ge(eref), '_parent'):
        weight = _ge(eref.weight)
        weight = 1e-10 if weight < 1e-10 else weight
        eref = eref._parent

    tokens = []
    tokens.append('solenoid')
    tokens.append(mad_assignment('l', _ge(eref.length) * weight, mad_type, substituted_vars=substituted_vars))
    tokens.append(mad_assignment('ks', _ge(eref.ks), mad_type, substituted_vars=substituted_vars))

    if getattr(_ge(eref), 'ksi', 0) != 0:
        raise ValueError('Thin solenoids are not implemented.')

    return tokens

def srotation_to_mad_str(eref, mad_type=MadType.MADX, substituted_vars=None):
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
    xt.CrabCavity: crabcavity_to_mad_str,
    xt.DriftSlice: drift_slice_to_mad_str,
}

element_types_converted_to_markers = {
    xt.LimitEllipse,
    xt.LimitPolygon,
    xt.LimitRacetrack,
    xt.LimitRect,
    xt.LimitRectEllipse,
}

def element_to_mad_str(
    name,
    line,
    mad_type=MadType.MADX,
    substituted_vars=None,
):
    """
    Generic converter for elements to MADX/MAD-NG.
    """

    el = line._element_dict[name]
    eref = _get_eref(line, name)

    while isinstance(el, xt.Replica):
        eref = line.ref[el.parent_name]
        el = line._element_dict[el.parent_name]

    parent_flag = hasattr(el, '_parent')

    if (el.__class__ == xt.Marker or el.__class__ in element_types_converted_to_markers
        or parent_flag and el._parent.__class__ == xt.Marker):
        return marker_to_mad_str(name, line, mad_type=mad_type, substituted_vars=substituted_vars)

    if el.__class__ not in xsuite_to_mad_converters:
        if isinstance(el, xt.beam_elements.slice_elements_drift._DriftSliceElementBase):
            tokens = drift_slice_to_mad_str(eref, mad_type=mad_type, substituted_vars=substituted_vars)
        elif parent_flag and el._parent.__class__ in xsuite_to_mad_converters:
            tokens = xsuite_to_mad_converters[el._parent.__class__](eref, mad_type=mad_type, substituted_vars=substituted_vars)
            if isinstance(el, xt.beam_elements.slice_elements_edge._ThinSliceEdgeBase):
                tokens.append(mad_assignment('kill_body', True, mad_type, substituted_vars=substituted_vars))
        else:
            raise NotImplementedError(f"Element of type {el.__class__} not supported yet in MAD writer")
    else:
        tokens = xsuite_to_mad_converters[el.__class__](eref, mad_type=mad_type, substituted_vars=substituted_vars)

    if el.__class__ not in [xt.Drift, xt.DriftSlice]:
        _handle_transforms(tokens, eref, mad_type=mad_type, substituted_vars=substituted_vars)

    if mad_type == MadType.MADNG:
        tokens = [tokens[0]] + [f"'{name.replace(':', '__')}'"] + tokens[1:]
        tokens = _handle_tokens_madng(tokens, substituted_vars)

    return ', '.join(tokens)

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
        seq_str = f'{name}: sequence, l={line_length};\n'

        s_dict = {}
        tt_name = tt.name
        tt_s = tt.s
        tt_isthick = tt.isthick
        for ii, nn in enumerate(tt.name):
            if nn.startswith("||drift_"):
                continue
            nn = tt_name[ii]
            if not(tt_isthick[ii]):
                s_dict[nn] = tt_s[ii]
            else:
                s_dict[nn] = 0.5 * (tt_s[ii] + tt_s[ii+1])

        for nn in line.element_names:
            if nn.startswith("||drift_"):
                continue
            el = line._element_dict[nn]
            el_str = element_to_mad_str(nn, line, mad_type=MadType.MADX)
            if nn + '_tilt_entry' in line._element_dict:
                el_str += ", " + mad_assignment('tilt',
                            _ge(line.element_refs[nn + '_tilt_entry'].angle) / 180. * np.pi,
                            mad_type=MadType.MADX)

            if el_str is None:
                continue

            nn_mad = nn.replace(':', '__')  # : not supported in madx names
            nn_mad = nn.replace('/', '__')  # / not supported in madx names
            seq_str += f"{nn_mad}: {el_str}, at={s_dict[nn]};\n"
        seq_str += 'endsequence;'
        machine_str = seq_str

    mad_input = vars_str + '\n' + machine_str + '\n'
    return mad_input

def to_madng_sequence(line, name='seq'):
    code_str = ""
    chunk_start = "(function()\t -- Begin chunk\n"
    chunk_end = "end)();\t -- End chunk\n"
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

    for ii, nn in enumerate(tt.env_name[:-1]): # ignore "_end_point"
        if not(tt.isthick[ii]):
            s_dict[nn] = tt.s[ii]
        else:
            s_dict[nn] = 0.5 * (tt.s[ii] + tt.s[ii+1])

        el = line._element_dict[nn]

        el_str = element_to_mad_str(nn, line, mad_type=MadType.MADNG, substituted_vars=substituted_vars)

        if el_str is None:
            continue

        # Misalignments
        if (hasattr(el, 'shift_x') and hasattr(el, 'shift_y')
            and el.__class__ not in element_types_converted_to_markers):
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
