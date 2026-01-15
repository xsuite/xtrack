from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Index columns used by FieldFitter and saved CSV files
FIELD_FIT_INDEX_COLUMNS: Tuple[str, ...] = (
    "field_component",
    "derivative_x",
    "region_name",
    "s_start",
    "s_end",
    "idx_start",
    "idx_end",
    "param_index",
)


@dataclass(frozen=True)
class SplineParameterSchema:
    """
    Canonical definition of the SplineBoris parameter ordering.

    The ordering implemented here is the single source of truth and must stay
    consistent with the expectations of the generated C code
    (see ``_generate_bpmeth_to_C.py``). Parameters are named:

    - ``bs_k``      : longitudinal field polynomial coefficients (only for i=1)
    - ``kn_i_k``    : normal multipole of order ``i`` and polynomial power ``k``
    - ``ks_i_k``    : skew multipole of order ``i`` and polynomial power ``k``

    The final ordering is alphabetical, i.e. the sorted list of all names.
    For a given ``multipole_order`` (number of multipole orders) and
    ``poly_order`` this leads to:

    - ``bs_0 .. bs_poly_order``
    - ``kn_0_0 .. kn_0_poly_order, kn_1_0 .., ..., kn_{multipole_order-1}_poly_order``
    - ``ks_0_0 .. ks_0_poly_order, ks_1_0 .., ..., ks_{multipole_order-1}_poly_order``
    """

    # Default polynomial order used across the current implementation (k = 0..4)
    POLY_ORDER: int = 4

    @classmethod
    def get_param_names(
        cls,
        multipole_order: int,
        poly_order: Optional[int] = None,
    ) -> List[str]:
        """
        Return the ordered list of parameter names for the given configuration.

        The logic matches the original ``param_names_list`` implementation in
        ``_generate_bpmeth_to_C.py`` exactly: build ks/kn/bs names and then
        sort them alphabetically.
        """
        if multipole_order is None:
            raise ValueError("multipole_order must be provided")
        if multipole_order <= 0:
            raise ValueError("multipole_order must be a positive integer")

        if poly_order is None:
            poly_order = cls.POLY_ORDER
        if poly_order < 0:
            raise ValueError("poly_order must be a non-negative integer")

        param_names: List[str] = []
        for i in range(multipole_order):
            for k in range(poly_order + 1):
                # Note: order and naming must remain consistent with bpmeth/C code
                ks_name = f"ks_{i}_{k}"
                kn_name = f"kn_{i}_{k}"
                param_names.append(ks_name)
                param_names.append(kn_name)

                # Longitudinal field only for i == 0
                if i == 0:
                    bs_name = f"bs_{k}"
                    param_names.append(bs_name)

        param_names.sort()
        return param_names

    @classmethod
    def get_num_params(
        cls,
        multipole_order: int,
        poly_order: Optional[int] = None,
    ) -> int:
        """
        Return the total number of parameters for the given configuration.

        For the current schema this is:

        .. math::
            N = (2 * \\text{multipole\\_order} + 1) * (\\text{poly\\_order} + 1)
        """
        if multipole_order is None:
            raise ValueError("multipole_order must be provided")
        if multipole_order <= 0:
            raise ValueError("multipole_order must be a positive integer")

        if poly_order is None:
            poly_order = cls.POLY_ORDER
        if poly_order < 0:
            raise ValueError("poly_order must be a non-negative integer")

        return (2 * multipole_order + 1) * (poly_order + 1)

    @classmethod
    def validate_param_array(
        cls,
        params: Sequence[Sequence[float]] | np.ndarray,
        multipole_order: int,
        poly_order: Optional[int] = None,
    ) -> bool:
        """
        Validate that the parameter array matches the expected schema.

        Parameters can be provided either as:

        - 1D array-like of shape ``(n_params,)``
        - 2D array-like of shape ``(n_steps, n_params)``
        """
        if poly_order is None:
            poly_order = cls.POLY_ORDER

        expected_n_params = cls.get_num_params(
            multipole_order=multipole_order, poly_order=poly_order
        )
        arr = np.asarray(params, dtype=float)

        if arr.ndim == 1:
            if arr.shape[0] != expected_n_params:
                raise ValueError(
                    f"Invalid parameter vector length {arr.shape[0]} "
                    f"(expected {expected_n_params} for "
                    f"multipole_order={multipole_order}, poly_order={poly_order})"
                )
        elif arr.ndim == 2:
            if arr.shape[1] != expected_n_params:
                raise ValueError(
                    f"Invalid parameter table shape {arr.shape} "
                    f"(expected (*, {expected_n_params}) for "
                    f"multipole_order={multipole_order}, poly_order={poly_order})"
                )
        else:
            raise ValueError(
                f"params must be 1D or 2D array-like, got array with ndim={arr.ndim}"
            )

        return True


def _reset_fieldfit_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure we have a flat DataFrame with the expected columns.

    Accepts both a MultiIndex (as produced by ``FieldFitter``) and a flat
    DataFrame (e.g. already reset).
    """
    if all(col in df.columns for col in FIELD_FIT_INDEX_COLUMNS):
        return df.copy()
    # Assume it's a MultiIndex with the expected names
    return df.reset_index()


def build_parameter_table_from_df(
    df_fit_pars: pd.DataFrame,
    n_steps: int,
    multipole_order: Optional[int] = None,
    poly_order: Optional[int] = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Build an ordered parameter table from a FieldFitter-style DataFrame.

    This is the shared implementation used by:

    - ``construct_undulator._contruct_par_table()``
    - ``SplineBoris.from_fieldfit_df`` / ``from_fieldfit_csv``

    The ordering is defined by :class:`SplineParameterSchema`.

    Parameters
    ----------
    df_fit_pars :
        DataFrame produced by ``FieldFitter`` with at least the columns from
        :data:`FIELD_FIT_INDEX_COLUMNS` plus ``param_name`` and ``param_value``.
    n_steps :
        Number of longitudinal steps in the constructed table.
    multipole_order :
        Maximum multipole order. If ``None``, inferred from the parameter
        names present in the DataFrame.
    poly_order :
        Polynomial order. If ``None``, defaults to
        :attr:`SplineParameterSchema.POLY_ORDER`.

    Returns
    -------
    par_table :
        ``(n_steps, n_params)`` array with parameters ordered according to
        :class:`SplineParameterSchema`.
    s_start, s_end :
        Longitudinal range covered by the table.
    """
    if df_fit_pars is None or len(df_fit_pars) == 0:
        raise ValueError("df_fit_pars must be a non-empty DataFrame")

    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")

    df_reset = _reset_fieldfit_index(df_fit_pars)

    if "param_name" not in df_reset.columns or "param_value" not in df_reset.columns:
        raise ValueError(
            "df_fit_pars must contain 'param_name' and 'param_value' columns"
        )

    if poly_order is None:
        poly_order = SplineParameterSchema.POLY_ORDER

    # Infer multipole_order from the parameter names if not provided
    if multipole_order is None:
        # Look for kn_i_k / ks_i_k and take the maximum i (0-based index)
        max_i = -1
        for name in df_reset["param_name"].unique():
            if isinstance(name, str) and (name.startswith("kn_") or name.startswith("ks_")):
                parts = name.split("_")
                if len(parts) >= 3:
                    try:
                        i_val = int(parts[1])
                    except ValueError:
                        continue
                    if i_val > max_i:
                        max_i = i_val
        if max_i < 0:
            raise ValueError(
                "Could not infer multipole_order from df_fit_pars; "
                "please provide multipole_order explicitly."
            )
        # ``multipole_order`` is the number of multipole orders, while the
        # index in the name runs from 0 to multipole_order-1.
        multipole_order = max_i + 1

    # Determine longitudinal range from s_start/s_end columns
    s_starts = np.sort(df_reset["s_start"].to_numpy(dtype=np.float64))
    s_ends = np.sort(df_reset["s_end"].to_numpy(dtype=np.float64))
    s_boundaries = np.sort(np.unique(np.concatenate((s_starts, s_ends))))
    s_start = float(s_boundaries[0])
    s_end = float(s_boundaries[-1])

    # Canonical parameter ordering for this configuration
    expected_params = SplineParameterSchema.get_param_names(
        multipole_order=multipole_order, poly_order=poly_order
    )

    # Longitudinal positions at which we want parameters
    s_vals = np.linspace(s_start, s_end, n_steps)

    par_table: List[List[float]] = []

    for s_val_i in s_vals:
        # Select all rows that cover this s position
        mask = (df_reset["s_start"] <= s_val_i) & (df_reset["s_end"] >= s_val_i)
        rows = df_reset[mask]

        param_dict: dict[str, float] = {}

        if not rows.empty:
            rows_sorted = rows.copy()
            # Prefer the most local regions when multiple overlaps exist
            rows_sorted["region_size"] = (
                rows_sorted["s_end"] - rows_sorted["s_start"]
            )
            rows_sorted = rows_sorted.sort_values(
                ["field_component", "region_size", "param_name"]
            )

            # First occurrence of each param_name (smallest region, deterministic)
            for _, row in rows_sorted.iterrows():
                pname = row["param_name"]
                if pname not in param_dict:
                    param_dict[pname] = float(row["param_value"])

        # Build parameter list in canonical order, filling missing values with 0
        param_values: List[float] = []
        for pname in expected_params:
            param_values.append(param_dict.get(pname, 0.0))

        par_table.append(param_values)

    par_arr = np.asarray(par_table, dtype=np.float64)
    # Final validation for safety
    SplineParameterSchema.validate_param_array(
        par_arr, multipole_order=multipole_order, poly_order=poly_order
    )

    return par_arr, s_start, s_end

