"""
Module for constructing and matching undulator wigglers.

This module provides functions to:
- Load and process fit_parameters.csv
- Construct piecewise wiggler from SplineBoris elements
- Match wiggler with correction elements
- Create both standard and offset wiggler lines
"""

import xtrack as xt
import numpy as np
import pandas as pd
from pathlib import Path

from xtrack.beam_elements.spline_param_schema import (
    SplineParameterSchema,
    build_parameter_table_from_df,
)


def _contruct_par_table(n_steps, s_start, s_end, df_flat, multipole_order):
    """
    Construct parameter table for SplineBoris.
    
    Parameters are ordered as expected by the C code:
    - For multipole_order=3: bs_*, kn_0_*, kn_1_*, kn_2_*, ks_0_*, ks_1_*, ks_2_*
    - Within each group: ordered by polynomial order (0, 1, 2, 3, 4)
    - kn_* (normal multipole) and ks_* (skew multipole)
    """
    # Build the canonical parameter table using the shared helper to ensure
    # consistency with FieldFitter, SplineBoris, and the C code.
    par_table, s_start_inferred, s_end_inferred = build_parameter_table_from_df(
        df_flat,
        n_steps=n_steps,
        multipole_order=multipole_order,
    )

    # For backwards compatibility, keep returning a list of per-step dictionaries.
    # These are constructed from the ordered parameter table.
    expected_params = SplineParameterSchema.get_param_names(multipole_order)
    par_dicts = [
        {name: float(value) for name, value in zip(expected_params, row)}
        for row in par_table
    ]

    # Preserve the original return signature.
    return par_dicts, par_table
