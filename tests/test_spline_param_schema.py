import numpy as np
import pandas as pd

import xtrack as xt


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
    param_names = xt.SplineBoris.get_param_names(
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
    df.set_index(list(xt.SplineBoris.FIELD_FIT_INDEX_COLUMNS), inplace=True)
    return df


def test_schema_param_names_and_count():
    m = 3
    p = 4
    names = xt.SplineBoris.get_param_names(multipole_order=m, poly_order=p)
    ref_names = _reference_param_names(multipole_order=m, poly_order=p)

    assert names == ref_names
    assert len(names) == xt.SplineBoris.get_num_params(multipole_order=m, poly_order=p)
    assert len(names) == (2 * m + 1) * (p + 1)


def test_validate_param_array_shapes():
    m = 2
    p = 1
    n_params = xt.SplineBoris.get_num_params(multipole_order=m, poly_order=p)

    # 1D vector (single step)
    vec = np.zeros(n_params)
    assert xt.SplineBoris.validate_param_array(vec, multipole_order=m, poly_order=p)

    # 2D table (multiple steps)
    table = np.zeros((5, n_params))
    assert xt.SplineBoris.validate_param_array(table, multipole_order=m, poly_order=p)

    # Wrong length should raise
    bad_vec = np.zeros(n_params + 1)
    try:
        xt.SplineBoris.validate_param_array(bad_vec, multipole_order=m, poly_order=p)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid parameter length")


def test_splineboris_sequence_creates_elements():
    """Test SplineBorisSequence creates elements with correct parameters."""
    m = 2
    p = 4  # use default POLY_ORDER
    df = _make_synthetic_df(multipole_order=m, poly_order=p)

    # Create SplineBorisSequence
    seq = xt.SplineBorisSequence(
        df_fit_pars=df,
        multipole_order=m,
        steps_per_point=1,
        poly_order=p,
    )

    # Should have 1 element (single piece in synthetic df)
    assert seq.n_pieces == 1
    assert len(seq.elements) == 1

    # The element should have correct multipole order
    elem = seq.elements[0]
    assert elem.multipole_order == m

    # Length should match s_end - s_start from df
    assert np.isclose(seq.length, 1.0)

    # Get the Line and verify
    line = seq.to_line()
    assert len(line.element_names) == 1


def test_splineboris_sequence_step_count():
    """Test SplineBorisSequence n_steps based on idx_end - idx_start."""
    m = 2
    p = 4
    df = _make_synthetic_df(multipole_order=m, poly_order=p)
    # df has idx_start=0, idx_end=10, so 10 data points

    # With steps_per_point=1, should get n_steps=10
    seq = xt.SplineBorisSequence(
        df_fit_pars=df,
        multipole_order=m,
        steps_per_point=1,
        poly_order=p,
    )
    assert seq.elements[0].n_steps == 10

    # With steps_per_point=2, should get n_steps=20
    seq2 = xt.SplineBorisSequence(
        df_fit_pars=df,
        multipole_order=m,
        steps_per_point=2,
        poly_order=p,
    )
    assert seq2.elements[0].n_steps == 20

    # With steps_per_point=3, should get n_steps=30
    seq3 = xt.SplineBorisSequence(
        df_fit_pars=df,
        multipole_order=m,
        steps_per_point=3,
        poly_order=p,
    )
    assert seq3.elements[0].n_steps == 30


def test_splineboris_sequence_from_csv(tmp_path):
    """Test SplineBorisSequence.from_csv creates elements correctly."""
    m = 2
    p = 4
    df = _make_synthetic_df(multipole_order=m, poly_order=p)

    # Save to CSV
    csv_path = tmp_path / "fit_pars.csv"
    df.to_csv(csv_path)

    # Load via from_csv
    seq = xt.SplineBorisSequence.from_csv(
        csv_path=csv_path,
        multipole_order=m,
        steps_per_point=1,
        poly_order=p,
    )

    assert seq.n_pieces == 1
    assert seq.multipole_order == m
    assert np.isclose(seq.length, 1.0)

    # Element should validate
    elem = seq.elements[0]
    assert elem.validate_params(poly_order=p)


def test_splineboris_sequence_with_shifts():
    """Test SplineBorisSequence passes shift_x/shift_y to elements."""
    m = 2
    p = 4
    df = _make_synthetic_df(multipole_order=m, poly_order=p)

    seq = xt.SplineBorisSequence(
        df_fit_pars=df,
        multipole_order=m,
        steps_per_point=1,
        shift_x=0.001,
        shift_y=0.002,
        poly_order=p,
    )

    elem = seq.elements[0]
    assert np.isclose(elem.shift_x, 0.001)
    assert np.isclose(elem.shift_y, 0.002)


def _make_overlapping_df():
    """Build a DataFrame with overlapping s-ranges for different components.

    Creates:
    - bs_0: covers [0, 10] with 100 points (density=10 pts/unit)
    - kn_0_0: covers [0, 5] with 50 points (density=10 pts/unit)
    - kn_0_1: covers [5, 10] with 50 points (density=10 pts/unit)

    This should result in 2 atomic regions:
    - [0, 5]: bs_0=1.0, kn_0_0=2.0
    - [5, 10]: bs_0=1.0, kn_0_1=3.0
    """
    rows = [
        # bs_0 covers full range [0, 10]
        {
            "field_component": "Bs",
            "derivative_x": 0,
            "region_name": "Poly_full",
            "s_start": 0.0,
            "s_end": 10.0,
            "idx_start": 0,
            "idx_end": 100,
            "param_index": 0,
            "param_name": "bs_0",
            "param_value": 1.0,
            "to_fit": True,
        },
        # kn_0_0 covers [0, 5]
        {
            "field_component": "By",
            "derivative_x": 0,
            "region_name": "Poly_0",
            "s_start": 0.0,
            "s_end": 5.0,
            "idx_start": 0,
            "idx_end": 50,
            "param_index": 0,
            "param_name": "kn_0_0",
            "param_value": 2.0,
            "to_fit": True,
        },
        # kn_0_0 covers [5, 10] (different region, different value)
        {
            "field_component": "By",
            "derivative_x": 0,
            "region_name": "Poly_1",
            "s_start": 5.0,
            "s_end": 10.0,
            "idx_start": 50,
            "idx_end": 100,
            "param_index": 0,
            "param_name": "kn_0_0",
            "param_value": 3.0,
            "to_fit": True,
        },
    ]

    df = pd.DataFrame(rows)
    df.set_index(list(xt.SplineBoris.FIELD_FIT_INDEX_COLUMNS), inplace=True)
    return df


def test_splineboris_sequence_overlapping_regions():
    """Test SplineBorisSequence correctly merges overlapping s-ranges."""
    df = _make_overlapping_df()

    seq = xt.SplineBorisSequence(
        df_fit_pars=df,
        multipole_order=1,
        steps_per_point=1,
        poly_order=0,  # only _0 coefficients
    )

    # Should create 2 atomic regions
    assert seq.n_pieces == 2

    # First region [0, 5]
    elem0 = seq.elements[0]
    assert np.isclose(elem0.s_start, 0.0)
    assert np.isclose(elem0.s_end, 5.0)
    # Should have bs_0=1.0 and kn_0_0=2.0

    # Second region [5, 10]
    elem1 = seq.elements[1]
    assert np.isclose(elem1.s_start, 5.0)
    assert np.isclose(elem1.s_end, 10.0)
    # Should have bs_0=1.0 and kn_0_0=3.0

    # Total length should be 10
    assert np.isclose(seq.length, 10.0)

    # n_steps should be based on density (10 pts/unit * 5 units = 50 per region)
    assert elem0.n_steps == 50
    assert elem1.n_steps == 50

