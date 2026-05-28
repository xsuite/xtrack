from solenoid_helpers_005 import (
    FIELD_DATA_JSON,
    LINES_JSON,
    build_splineboris_line,
    load_extracted_fields_json,
    save_lines_json,
    validate_max_multipole_order,
)


metadata, extracted_fields = load_extracted_fields_json(FIELD_DATA_JSON)
max_multipole_order = metadata['max_multipole_order']
validate_max_multipole_order(max_multipole_order)

lines = {
    extracted.name: build_splineboris_line(extracted, max_multipole_order)
    for extracted in extracted_fields.values()
}

save_lines_json(lines, file=LINES_JSON)

print(f'Loaded {FIELD_DATA_JSON}')
print(f'Wrote {LINES_JSON}')
for name, line in lines.items():
    print(f'  {name}: {len(line.elements)} SplineBoris elements')
