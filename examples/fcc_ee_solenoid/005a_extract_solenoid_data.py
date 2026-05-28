import xtrack as xt
import matplotlib.pyplot as plt

from solenoid_helpers_005 import (
    FIELD_DATA_JSON,
    MAX_MULTIPOLE_ORDER,
    compute_bs_integrals,
    extract_required_field_data,
    extracted_field_to_dict,
    field_data_metadata,
    format_bs_integrals_title,
    make_solenoid_specs,
    plot_extracted_fields,
    validate_max_multipole_order,
)


validate_max_multipole_order(MAX_MULTIPOLE_ORDER)

specs = make_solenoid_specs()
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
