import matplotlib.pyplot as plt

from solenoid_helpers_005 import (
    FIELD_DATA_JSON,
    LINES_JSON,
    X_FIELD_COMPARISON,
    Y_FIELD_COMPARISON,
    compute_bs_integrals,
    compute_derivative_comparison_data,
    format_bs_integrals_title,
    load_extracted_fields_json,
    load_lines_json,
    plot_field_comparison,
    plot_transverse_derivative_comparison,
    validate_max_multipole_order,
)


metadata, extracted_fields = load_extracted_fields_json(FIELD_DATA_JSON)
max_multipole_order = metadata['max_multipole_order']
validate_max_multipole_order(max_multipole_order)

lines = load_lines_json(LINES_JSON)

bs_integrals = compute_bs_integrals(extracted_fields)
bs_integrals_title = format_bs_integrals_title(bs_integrals)

plt.close('all')

fig_field_comparison, axes_field_comparison = plot_field_comparison(
    extracted_fields,
    lines,
    x_eval=X_FIELD_COMPARISON,
    y_eval=Y_FIELD_COMPARISON,
    bs_integrals_title=bs_integrals_title,
)

derivative_comparison_data = compute_derivative_comparison_data(
    extracted_fields,
    lines,
    max_derivative_order=max_multipole_order,
    x_eval=X_FIELD_COMPARISON,
    y_eval=Y_FIELD_COMPARISON,
)

derivative_comparison_figures = {}
derivative_comparison_axes = {}
for order in range(1, max_multipole_order + 1):
    fig, axes = plot_transverse_derivative_comparison(
        derivative_comparison_data,
        derivative_order=order,
        x_eval=X_FIELD_COMPARISON,
        y_eval=Y_FIELD_COMPARISON,
        bs_integrals_title=bs_integrals_title,
    )
    derivative_comparison_figures[order] = fig
    derivative_comparison_axes[order] = axes

print(f'Loaded {FIELD_DATA_JSON}')
print(f'Loaded {LINES_JSON}')
print(bs_integrals_title)

plt.show()
