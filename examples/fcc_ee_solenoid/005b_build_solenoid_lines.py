import numpy as np
import xtrack as xt
from pathlib import Path

from solenoid_helpers_005 import (
    load_field_data_json,
    validate_max_multipole_order,
)


HERE = Path(__file__).parent
FIELD_DATA_JSON = HERE / '005_solenoid_field_data.json'
LINES_JSON = HERE / '005_solenoid_lines.json'
SPLINE_STEPS_PER_POINT = 10
USE_PIECEWISE_LINEAR_SPLINES = False
FORCE_IDEAL_SOLENOID_TRANSVERSE_FIELD = False
FORCE_ZERO_FIELD_AT_SOLENOID_ENDS = False

metadata, fields = load_field_data_json(FIELD_DATA_JSON)
max_multipole_order = metadata['max_multipole_order']
validate_max_multipole_order(max_multipole_order)
if FORCE_IDEAL_SOLENOID_TRANSVERSE_FIELD and max_multipole_order < 1:
    raise ValueError(
        'FORCE_IDEAL_SOLENOID_TRANSVERSE_FIELD needs '
        'max_multipole_order >= 1')

lines = {}

for name, field in fields.items():
    s_axis = field['s_axis']
    n_intervals = len(s_axis) - 1

    bs_values = np.array(field['bs'], copy=True)
    bx_values = {
        order: np.array(values, copy=True)
        for order, values in field['bx'].items()
    }
    by_values = {
        order: np.array(values, copy=True)
        for order, values in field['by'].items()
    }
    bs_integral_average_values = np.array(
        field['bs_integral_average'], copy=True)
    bx_integral_average_values = {
        order: np.array(values, copy=True)
        for order, values in field['bx_integral_average'].items()
    }
    by_integral_average_values = {
        order: np.array(values, copy=True)
        for order, values in field['by_integral_average'].items()
    }

    if FORCE_ZERO_FIELD_AT_SOLENOID_ENDS:
        bs_values[0] = 0.0
        bs_values[-1] = 0.0
        for order in range(max_multipole_order + 1):
            bx_values[order][0] = 0.0
            bx_values[order][-1] = 0.0
            by_values[order][0] = 0.0
            by_values[order][-1] = 0.0

        bs_integral_average_values[0] = 0.5 * (bs_values[0] + bs_values[1])
        bs_integral_average_values[-1] = 0.5 * (bs_values[-2] + bs_values[-1])
        for order in range(max_multipole_order + 1):
            bx_integral_average_values[order][0] = 0.5 * (
                bx_values[order][0] + bx_values[order][1])
            bx_integral_average_values[order][-1] = 0.5 * (
                bx_values[order][-2] + bx_values[order][-1])
            by_integral_average_values[order][0] = 0.5 * (
                by_values[order][0] + by_values[order][1])
            by_integral_average_values[order][-1] = 0.5 * (
                by_values[order][-2] + by_values[order][-1])

    bs_s_derivative = np.gradient(bs_values, s_axis, edge_order=2)
    bx_s_derivatives = {
        order: np.gradient(values, s_axis, edge_order=2)
        for order, values in bx_values.items()
    }
    by_s_derivatives = {
        order: np.gradient(values, s_axis, edge_order=2)
        for order, values in by_values.items()
    }
    ideal_dbx_dx = -0.5 * bs_s_derivative
    ideal_dbx_dx_s_derivative = np.gradient(
        ideal_dbx_dx, s_axis, edge_order=2)

    elements = []
    element_names = []
    name_width = len(str(max(0, n_intervals - 1)))

    for ii in range(n_intervals):
        length = s_axis[ii + 1] - s_axis[ii]

        if USE_PIECEWISE_LINEAR_SPLINES:
            bs_derivative = (bs_values[ii + 1] - bs_values[ii]) / length
            bs_integral_average = 0.5 * (bs_values[ii] + bs_values[ii + 1])
        else:
            bs_derivative = None
            bs_integral_average = bs_integral_average_values[ii]

        bs = xt.Spline4(
            val_start=bs_values[ii],
            der_start=(bs_derivative if USE_PIECEWISE_LINEAR_SPLINES
                       else bs_s_derivative[ii]),
            val_end=bs_values[ii + 1],
            der_end=(bs_derivative if USE_PIECEWISE_LINEAR_SPLINES
                     else bs_s_derivative[ii + 1]),
            integral=bs_integral_average,
        )

        bx = []
        by = []
        for order in range(max_multipole_order + 1):
            if FORCE_IDEAL_SOLENOID_TRANSVERSE_FIELD:
                if order == 1:
                    if USE_PIECEWISE_LINEAR_SPLINES:
                        ideal_value = -0.5 * bs_derivative
                        bx_val_start = ideal_value
                        bx_val_end = ideal_value
                        bx_der_start = 0.0
                        bx_der_end = 0.0
                        bx_integral_average = ideal_value
                    else:
                        bx_val_start = ideal_dbx_dx[ii]
                        bx_val_end = ideal_dbx_dx[ii + 1]
                        bx_der_start = ideal_dbx_dx_s_derivative[ii]
                        bx_der_end = ideal_dbx_dx_s_derivative[ii + 1]
                        bx_integral_average = (
                            -0.5 * (bs_values[ii + 1] - bs_values[ii])
                            / length
                        )
                else:
                    bx_val_start = 0.0
                    bx_val_end = 0.0
                    bx_der_start = 0.0
                    bx_der_end = 0.0
                    bx_integral_average = 0.0

                by_val_start = 0.0
                by_val_end = 0.0
                by_der_start = 0.0
                by_der_end = 0.0
                by_integral_average = 0.0

            elif USE_PIECEWISE_LINEAR_SPLINES:
                bx_val_start = bx_values[order][ii]
                bx_val_end = bx_values[order][ii + 1]
                by_val_start = by_values[order][ii]
                by_val_end = by_values[order][ii + 1]
                bx_derivative = (
                    (bx_values[order][ii + 1] - bx_values[order][ii])
                    / length
                )
                by_derivative = (
                    (by_values[order][ii + 1] - by_values[order][ii])
                    / length
                )
                bx_der_start = bx_derivative
                bx_der_end = bx_derivative
                by_der_start = by_derivative
                by_der_end = by_derivative
                bx_integral_average = 0.5 * (
                    bx_values[order][ii] + bx_values[order][ii + 1])
                by_integral_average = 0.5 * (
                    by_values[order][ii] + by_values[order][ii + 1])
            else:
                bx_val_start = bx_values[order][ii]
                bx_val_end = bx_values[order][ii + 1]
                by_val_start = by_values[order][ii]
                by_val_end = by_values[order][ii + 1]
                bx_der_start = bx_s_derivatives[order][ii]
                bx_der_end = bx_s_derivatives[order][ii + 1]
                by_der_start = by_s_derivatives[order][ii]
                by_der_end = by_s_derivatives[order][ii + 1]
                bx_integral_average = bx_integral_average_values[order][ii]
                by_integral_average = by_integral_average_values[order][ii]

            bx.append(xt.Spline4(
                val_start=bx_val_start,
                der_start=bx_der_start,
                val_end=bx_val_end,
                der_end=bx_der_end,
                integral=bx_integral_average,
            ))
            by.append(xt.Spline4(
                val_start=by_val_start,
                der_start=by_der_start,
                val_end=by_val_end,
                der_end=by_der_end,
                integral=by_integral_average,
            ))

        elements.append(xt.SplineBoris(
            bs=bs,
            bx=tuple(bx),
            by=tuple(by),
            length=length,
            n_steps=SPLINE_STEPS_PER_POINT,
            scale_b=field['scale_b'],
        ))
        element_names.append(f'{name}_splineboris_{ii:0{name_width}d}')

    lines[name] = xt.Line(elements=elements, element_names=element_names)

output_data = {
    'lines': {
        name: line.to_dict()
        for name, line in lines.items()
    },
}
xt.json.dump(output_data, LINES_JSON, indent=1)

print(f'Loaded {FIELD_DATA_JSON}')
print(f'Wrote {LINES_JSON}')
print(f'USE_PIECEWISE_LINEAR_SPLINES = {USE_PIECEWISE_LINEAR_SPLINES}')
print(
    'FORCE_IDEAL_SOLENOID_TRANSVERSE_FIELD = '
    f'{FORCE_IDEAL_SOLENOID_TRANSVERSE_FIELD}')
print(f'FORCE_ZERO_FIELD_AT_SOLENOID_ENDS = {FORCE_ZERO_FIELD_AT_SOLENOID_ENDS}')
for name, line in lines.items():
    print(f'  {name}: {len(line.elements)} SplineBoris elements')
