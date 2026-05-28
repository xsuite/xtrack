from solenoid_helpers_005 import (
    FIELD_DATA_JSON,
    MAX_MULTIPOLE_ORDER,
    compute_bs_integrals,
    extract_required_field_data,
    format_bs_integrals_title,
    make_solenoid_specs,
    save_extracted_fields_json,
    validate_max_multipole_order,
)


validate_max_multipole_order(MAX_MULTIPOLE_ORDER)

specs = make_solenoid_specs()
extracted_fields = [
    extract_required_field_data(spec, MAX_MULTIPOLE_ORDER)
    for spec in specs
]

bs_integrals = compute_bs_integrals(extracted_fields)
bs_integrals_title = format_bs_integrals_title(bs_integrals)

save_extracted_fields_json(
    extracted_fields,
    file=FIELD_DATA_JSON,
    max_multipole_order=MAX_MULTIPOLE_ORDER,
)

print(f'Wrote {FIELD_DATA_JSON}')
print(bs_integrals_title)
