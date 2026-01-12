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


def _contruct_par_table(n_steps, s_start, s_end, df_flat, multipole_order):
    """
    Construct parameter table for SplineBoris.
    
    Parameters are ordered as expected by the C code:
    - For multipole_order=3: bs_*, b_1_*, b_2_*, b_3_*, a_1_*, a_2_*, a_3_*
    - Within each group: ordered by polynomial order (0, 1, 2, 3, 4)
    - b_* maps to kn_* (normal multipole) and a_* maps to ks_* (skew multipole)
    """
    par_dicts = []
    par_table = []
    s_vals = np.linspace(float(s_start), float(s_end), n_steps)
    
    # Expected parameter order for the given multipole_order
    # C code expects: bs_*, kn_1_*, ..., kn_N_*, ks_1_*, ..., ks_N_*
    # Which maps to: bs_*, b_1_*, ..., b_N_*, a_1_*, ..., a_N_*
    expected_params = []
    # First: bs_ parameters (no multipole index)
    for poly_idx in range(5):
        expected_params.append(f"bs_{poly_idx}")
    # Second: b_ parameters (all multipoles) - these map to kn_*
    for multipole_idx in range(1, multipole_order + 1):
        for poly_idx in range(5):  # 0 to 4
            expected_params.append(f"b_{multipole_idx}_{poly_idx}")
    # Third: a_ parameters (all multipoles) - these map to ks_*
    for multipole_idx in range(1, multipole_order + 1):
        for poly_idx in range(5):  # 0 to 4
            expected_params.append(f"a_{multipole_idx}_{poly_idx}")
    
    for s_val_i in s_vals:
        # Filter rows that contain this s position
        # Only use derivative_x=0 (the field itself, not derivatives)
        mask = ((df_flat['s_start'] <= s_val_i) & 
                (df_flat['s_end'] >= s_val_i) & 
                (df_flat['derivative_x'] == 0))
        rows = df_flat[mask]
        
        # Create a dictionary from available parameters
        # Group by field_component to ensure we get all parameters from the same region
        # when possible, but allow different regions for different field components
        param_dict = {}
        if not rows.empty:
            # Sort by field_component and region to ensure consistent selection
            # when there are multiple matches, prefer the most specific region (smallest s range)
            rows_sorted = rows.copy()
            rows_sorted['region_size'] = rows_sorted['s_end'] - rows_sorted['s_start']
            rows_sorted = rows_sorted.sort_values(['field_component', 'region_size', 'param_name'])
            
            # Take the first occurrence of each param_name (should be from the smallest region)
            for _, row in rows_sorted.iterrows():
                param_name = row['param_name']
                if param_name not in param_dict:
                    param_dict[param_name] = row['param_value']
        
        # Build parameter list in expected order, filling missing values with 0
        param_values = []
        for param_name in expected_params:
            if param_name in param_dict:
                param_values.append(float(param_dict[param_name]))
            else:
                param_values.append(0.0)  # Fill missing parameters with 0
        
        par_dicts.append(param_dict)
        par_table.append(param_values)
    
    return par_dicts, par_table
