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
USE_PIECEWISE_LINEAR_SPLINES = True

metadata, fields = load_field_data_json(FIELD_DATA_JSON)
max_multipole_order = metadata['max_multipole_order']
validate_max_multipole_order(max_multipole_order)

lines = {}

for name, field in fields.items():
    s_axis = field['s_axis']
    n_intervals = len(s_axis) - 1

    bs_s_derivative = np.gradient(field['bs'], s_axis, edge_order=2)
    bx_s_derivatives = {
        order: np.gradient(values, s_axis, edge_order=2)
        for order, values in field['bx'].items()
    }
    by_s_derivatives = {
        order: np.gradient(values, s_axis, edge_order=2)
        for order, values in field['by'].items()
    }

    elements = []
    element_names = []
    name_width = len(str(max(0, n_intervals - 1)))

    for ii in range(n_intervals):
        length = s_axis[ii + 1] - s_axis[ii]

        if USE_PIECEWISE_LINEAR_SPLINES:
            bs_derivative = (field['bs'][ii + 1] - field['bs'][ii]) / length
            bs_integral_average = 0.5 * (field['bs'][ii] + field['bs'][ii + 1])
        else:
            bs_derivative = None
            bs_integral_average = field['bs_integral_average'][ii]

        bs = xt.Spline4(
            val_start=field['bs'][ii],
            der_start=(bs_derivative if USE_PIECEWISE_LINEAR_SPLINES
                       else bs_s_derivative[ii]),
            val_end=field['bs'][ii + 1],
            der_end=(bs_derivative if USE_PIECEWISE_LINEAR_SPLINES
                     else bs_s_derivative[ii + 1]),
            integral=bs_integral_average,
        )

        bx = []
        by = []
        for order in range(max_multipole_order + 1):
            if USE_PIECEWISE_LINEAR_SPLINES:
                bx_derivative = (
                    (field['bx'][order][ii + 1] - field['bx'][order][ii])
                    / length
                )
                by_derivative = (
                    (field['by'][order][ii + 1] - field['by'][order][ii])
                    / length
                )
                bx_integral_average = 0.5 * (
                    field['bx'][order][ii] + field['bx'][order][ii + 1])
                by_integral_average = 0.5 * (
                    field['by'][order][ii] + field['by'][order][ii + 1])
            else:
                bx_derivative = None
                by_derivative = None
                bx_integral_average = field['bx_integral_average'][order][ii]
                by_integral_average = field['by_integral_average'][order][ii]

            bx.append(xt.Spline4(
                val_start=field['bx'][order][ii],
                der_start=(bx_derivative if USE_PIECEWISE_LINEAR_SPLINES
                           else bx_s_derivatives[order][ii]),
                val_end=field['bx'][order][ii + 1],
                der_end=(bx_derivative if USE_PIECEWISE_LINEAR_SPLINES
                         else bx_s_derivatives[order][ii + 1]),
                integral=bx_integral_average,
            ))
            by.append(xt.Spline4(
                val_start=field['by'][order][ii],
                der_start=(by_derivative if USE_PIECEWISE_LINEAR_SPLINES
                           else by_s_derivatives[order][ii]),
                val_end=field['by'][order][ii + 1],
                der_end=(by_derivative if USE_PIECEWISE_LINEAR_SPLINES
                         else by_s_derivatives[order][ii + 1]),
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
for name, line in lines.items():
    print(f'  {name}: {len(line.elements)} SplineBoris elements')
