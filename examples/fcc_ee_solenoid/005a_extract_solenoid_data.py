import xtrack as xt
import matplotlib.pyplot as plt
import numpy as np

from tilted_solenoid import TiltedSolenoid
from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField

from solenoid_helpers_005 import (
    FIELD_DATA_JSON,
    MAX_MULTIPOLE_ORDER,
    SolenoidSpec,
    compute_bs_integrals,
    extract_required_field_data,
    extracted_field_to_dict,
    field_data_metadata,
    format_bs_integrals_title,
    plot_extracted_fields,
    validate_max_multipole_order,
)


THETA = -0.015
N_SLICES_MAIN_SOLENOID = 201
N_SLICES_COMP_SOLENOID = 201

validate_max_multipole_order(MAX_MULTIPOLE_ORDER)

main_solenoid = SolenoidSpec(
    name='main_solenoid',
    field_model=TiltedSolenoid(L=1.23 * 2, a=0.13, B0=2.0, theta=THETA),
    s_axis=np.linspace(-2.399, 2.399, N_SLICES_MAIN_SOLENOID),
    scale_b=1.0,
)

compensation_solenoid = SolenoidSpec(
    name='compensation_solenoid',
    field_model=SolenoidField(L=1.5, a=0.03, B0=1.0, z0=0.0),
    s_axis=np.linspace(-1.0, 1.0, N_SLICES_COMP_SOLENOID),
    scale_b=1.0,
)

_, _, bz_main = main_solenoid.field_model.get_field(
    np.zeros_like(main_solenoid.s_axis),
    np.zeros_like(main_solenoid.s_axis),
    main_solenoid.s_axis,
)
_, _, bz_comp = compensation_solenoid.field_model.get_field(
    np.zeros_like(compensation_solenoid.s_axis),
    np.zeros_like(compensation_solenoid.s_axis),
    compensation_solenoid.s_axis,
)
compensation_solenoid.scale_b = (
    -np.trapezoid(bz_main, main_solenoid.s_axis)
    / np.trapezoid(bz_comp, compensation_solenoid.s_axis)
    / 2.0
)

specs = {
    main_solenoid.name: main_solenoid,
    compensation_solenoid.name: compensation_solenoid,
}

extracted_fields = {
    name: extract_required_field_data(spec, MAX_MULTIPOLE_ORDER)
    for name, spec in specs.items()
}

output_data = {
    'metadata': field_data_metadata(MAX_MULTIPOLE_ORDER),
    'fields': {
        name: extracted_field_to_dict(extracted)
        for name, extracted in extracted_fields.items()
    },
}

bs_integrals = compute_bs_integrals(extracted_fields)
bs_integrals_title = format_bs_integrals_title(bs_integrals)

xt.json.dump(output_data, FIELD_DATA_JSON, indent=1)

print(f'Wrote {FIELD_DATA_JSON}')
print(bs_integrals_title)

fig_extracted_fields, axes_extracted_fields = plot_extracted_fields(
    extracted_fields,
    MAX_MULTIPOLE_ORDER,
    bs_integrals_title,
)

plt.show()
