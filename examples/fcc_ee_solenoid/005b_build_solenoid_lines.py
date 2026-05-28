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

        bs = xt.Spline4(
            val_start=field['bs'][ii],
            der_start=bs_s_derivative[ii],
            val_end=field['bs'][ii + 1],
            der_end=bs_s_derivative[ii + 1],
            integral=field['bs_integral_average'][ii],
        )

        bx = []
        by = []
        for order in range(max_multipole_order + 1):
            bx.append(xt.Spline4(
                val_start=field['bx'][order][ii],
                der_start=bx_s_derivatives[order][ii],
                val_end=field['bx'][order][ii + 1],
                der_end=bx_s_derivatives[order][ii + 1],
                integral=field['bx_integral_average'][order][ii],
            ))
            by.append(xt.Spline4(
                val_start=field['by'][order][ii],
                der_start=by_s_derivatives[order][ii],
                val_end=field['by'][order][ii + 1],
                der_end=by_s_derivatives[order][ii + 1],
                integral=field['by_integral_average'][order][ii],
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
for name, line in lines.items():
    print(f'  {name}: {len(line.elements)} SplineBoris elements')
