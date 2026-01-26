import numpy as np
import pandas as pd

import xtrack as xt

from xtrack.beam_elements.spline_param_schema import (
    FIELD_FIT_INDEX_COLUMNS,
    SplineParameterSchema,
    build_parameter_table_from_df,
)


def _reference_param_names(multipole_order=3, poly_order=4):
    """Reference implementation mirroring the original C generator logic."""
    names = []
    for i in range(multipole_order):
        for k in range(poly_order + 1):
            names.append(f"ks_{i}_{k}")
            names.append(f"kn_{i}_{k}")
            if i == 0:
                names.append(f"bs_{k}")
    names.sort()
    return names


def _make_synthetic_df(multipole_order=2, poly_order=1):
    """Build a minimal FieldFitter-like DataFrame for testing."""
    param_names = SplineParameterSchema.get_param_names(
        multipole_order=multipole_order, poly_order=poly_order
    )

    rows = []
    for idx, pname in enumerate(param_names):
        if pname.startswith("ks_"):
            field_component = "Bx"
        elif pname.startswith("kn_"):
            field_component = "By"
        elif pname.startswith("bs_"):
            field_component = "Bs"
        else:
            field_component = "Bx"

        rows.append(
            {
                "field_component": field_component,
                "derivative_x": 0,
                "region_name": "Poly_0",
                "s_start": 0.0,
                "s_end": 1.0,
                "idx_start": 0,
                "idx_end": 10,
                "param_index": idx,
                "param_name": pname,
                "param_value": float(idx + 1),
                "to_fit": True,
            }
        )

    df = pd.DataFrame(rows)
    df.set_index(list(FIELD_FIT_INDEX_COLUMNS), inplace=True)
    return df


def test_schema_param_names_and_count():
    m = 3
    p = 4
    names = SplineParameterSchema.get_param_names(multipole_order=m, poly_order=p)
    ref_names = _reference_param_names(multipole_order=m, poly_order=p)

    assert names == ref_names
    assert len(names) == SplineParameterSchema.get_num_params(multipole_order=m, poly_order=p)
    assert len(names) == (2 * m + 1) * (p + 1)


def test_validate_param_array_shapes():
    m = 2
    p = 1
    n_params = SplineParameterSchema.get_num_params(multipole_order=m, poly_order=p)

    # 1D vector (single step)
    vec = np.zeros(n_params)
    assert SplineParameterSchema.validate_param_array(vec, multipole_order=m, poly_order=p)

    # 2D table (multiple steps)
    table = np.zeros((5, n_params))
    assert SplineParameterSchema.validate_param_array(table, multipole_order=m, poly_order=p)

    # Wrong length should raise
    bad_vec = np.zeros(n_params + 1)
    try:
        SplineParameterSchema.validate_param_array(bad_vec, multipole_order=m, poly_order=p)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid parameter length")


def test_build_parameter_table_from_df_ordering():
    m = 2
    p = 1
    n_steps = 4
    df = _make_synthetic_df(multipole_order=m, poly_order=p)

    par_table, s_start, s_end = build_parameter_table_from_df(
        df_fit_pars=df,
        n_steps=n_steps,
        multipole_order=m,
        poly_order=p,
    )

    assert par_table.shape[0] == n_steps
    expected_names = SplineParameterSchema.get_param_names(multipole_order=m, poly_order=p)
    expected_row = np.arange(1, len(expected_names) + 1, dtype=float)

    # All rows should be identical to the expected mapping
    for i in range(n_steps):
        np.testing.assert_allclose(par_table[i], expected_row)

    assert s_start == 0.0
    assert s_end == 1.0


def test_splineboris_from_fieldfit_df_and_csv(tmp_path):
    m = 2
    p = 1
    n_steps = 5
    df = _make_synthetic_df(multipole_order=m, poly_order=p)

    par_table, s_start, s_end = build_parameter_table_from_df(
        df_fit_pars=df,
        n_steps=n_steps,
        multipole_order=m,
        poly_order=p,
    )

    # From DataFrame - use direct constructor with build_parameter_table_from_df
    sb_df = xt.SplineBoris(
        par_table=par_table,
        s_start=s_start,
        s_end=s_end,
        multipole_order=m,
        n_steps=n_steps,
    )

    assert sb_df.multipole_order == m
    assert sb_df.n_steps == n_steps
    np.testing.assert_allclose(np.asarray(sb_df.par_table), par_table)

    # Save to CSV and load via direct constructor
    csv_path = tmp_path / "fit_pars.csv"
    df.to_csv(csv_path)
    
    df_csv = pd.read_csv(csv_path, index_col=list(FIELD_FIT_INDEX_COLUMNS))
    par_table_csv, s_start_csv, s_end_csv = build_parameter_table_from_df(
        df_fit_pars=df_csv,
        n_steps=n_steps,
        multipole_order=m,
        poly_order=p,
    )

    sb_csv = xt.SplineBoris(
        par_table=par_table_csv,
        s_start=s_start_csv,
        s_end=s_end_csv,
        multipole_order=m,
        n_steps=n_steps,
    )

    assert sb_csv.multipole_order == m
    assert sb_csv.n_steps == n_steps
    np.testing.assert_allclose(np.asarray(sb_csv.par_table), par_table)

    # Optional validation method should succeed
    assert sb_csv.validate_params(poly_order=p)

