import numpy as np
import xtrack as xt


def validate_max_multipole_order(max_multipole_order):
    max_supported_order = xt.SplineBoris._SB_MAX_MULTIPOLE_ORDER - 1
    if max_multipole_order < 0:
        raise ValueError('MAX_MULTIPOLE_ORDER must be non-negative')
    if max_multipole_order > max_supported_order:
        raise ValueError(
            f'MAX_MULTIPOLE_ORDER={max_multipole_order} is too high; '
            f'this SplineBoris supports at most {max_supported_order}')


def zero_negligible_central_derivative_values(
        bx, by, s_axis, derivative_order,
        zero_from_order, zero_half_length):
    if (
        derivative_order < zero_from_order
        or zero_half_length <= 0
    ):
        return bx, by

    bx = np.array(bx, copy=True)
    by = np.array(by, copy=True)
    mask_center = np.abs(s_axis) <= zero_half_length
    bx[mask_center] = 0.0
    by[mask_center] = 0.0
    return bx, by


def load_field_data_json(file):
    data = xt.json.load(file)
    fields_in = data['fields']
    if isinstance(fields_in, list):
        fields_in = {item['name']: item for item in fields_in}

    fields = {}
    for name, item in fields_in.items():
        fields[name] = {
            'name': item['name'],
            's_axis': np.array(item['s_axis'], dtype=float),
            'bs': np.array(item['bs'], dtype=float),
            'bx': {
                int(order): np.array(values, dtype=float)
                for order, values in item['bx'].items()
            },
            'by': {
                int(order): np.array(values, dtype=float)
                for order, values in item['by'].items()
            },
            'scale_b': float(item['scale_b']),
            'bs_integral_average': np.array(
                item['bs_integral_average'], dtype=float),
            'bx_integral_average': {
                int(order): np.array(values, dtype=float)
                for order, values in item['bx_integral_average'].items()
            },
            'by_integral_average': {
                int(order): np.array(values, dtype=float)
                for order, values in item['by_integral_average'].items()
            },
        }

    return data['metadata'], fields


def compute_bs_integrals(fields):
    return {
        name: np.trapezoid(
            field['scale_b'] * field['bs'], field['s_axis'])
        for name, field in fields.items()
    }


def format_bs_integrals_title(bs_integrals):
    entries = [
        f'{name}={value:.6g}' for name, value in bs_integrals.items()]
    entries.append(f'sum={sum(bs_integrals.values()):.6g}')
    return 'int Bs ds [T m]: ' + ', '.join(entries)
