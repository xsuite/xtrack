"""
Module for constructing undulator wigglers using SplineBorisSequence.

This module provides functions to load field fit parameters and construct
undulator lines using the SplineBorisSequence class.
"""

import xtrack as xt
import pandas as pd
from pathlib import Path


def load_undulator_sequence(
    csv_path,
    multipole_order,
    steps_per_point=1,
    shift_x=0.0,
    shift_y=0.0,
):
    """
    Load a SplineBorisSequence from a field fit parameters CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing fit parameters.
    multipole_order : int
        Number of multipole orders to use.
    steps_per_point : int, optional
        Multiplier for integration steps per data point. Default is 1.
    shift_x : float, optional
        Transverse shift in x [m]. Default is 0.0.
    shift_y : float, optional
        Transverse shift in y [m]. Default is 0.0.

    Returns
    -------
    xt.SplineBorisSequence
        A SplineBorisSequence instance ready to use.
    """
    return xt.SplineBorisSequence.from_csv(
        csv_path=csv_path,
        multipole_order=multipole_order,
        steps_per_point=steps_per_point,
        shift_x=shift_x,
        shift_y=shift_y,
    )


def load_undulator_line(
    csv_path,
    multipole_order,
    steps_per_point=1,
    shift_x=0.0,
    shift_y=0.0,
):
    """
    Load an undulator line from a field fit parameters CSV file.

    This is a convenience function that creates a SplineBorisSequence
    and returns the Line directly.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file containing fit parameters.
    multipole_order : int
        Number of multipole orders to use.
    steps_per_point : int, optional
        Multiplier for integration steps per data point. Default is 1.
    shift_x : float, optional
        Transverse shift in x [m]. Default is 0.0.
    shift_y : float, optional
        Transverse shift in y [m]. Default is 0.0.

    Returns
    -------
    xt.Line
        A Line containing SplineBoris elements for the undulator.
    """
    seq = load_undulator_sequence(
        csv_path=csv_path,
        multipole_order=multipole_order,
        steps_per_point=steps_per_point,
        shift_x=shift_x,
        shift_y=shift_y,
    )
    return seq.to_line()
