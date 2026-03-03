from pathlib import Path

import pandas as pd

from xtrack._temp.field_fitter import FieldFitter


'''
Basic usage of FieldFitter.

This script fits a field map and saves the fit parameters to a file.

It plots the fit results for each derivative order.
It also plots the integrated field along the longitudinal direction.

The raw data only has three transverse x positions, which means the highest order polynomial that we can fit is 2.
This also means that we can only incorporate up to the second derivative of the field into the fit (sextupole components).
'''

dz = 0.001  # Step size in the z (longitudinal) direction for numerical differentiation

# Convert the field map to a DataFrame
file_path = Path(__file__).resolve().parent.parent.parent / "test_data" / "sls" / "undulator_field_map.txt"
df_raw_data = pd.read_csv(
    file_path, sep=r"\s+", header=None,
    names=["X", "Y", "Z", "Bx", "By", "Bs"],
).set_index(["X", "Y", "Z"])

deg = 2

fitter = FieldFitter(
    raw_data=df_raw_data,
    xy_point=(0.0, 0.0),
    distance_unit=dz,
    min_region_size=10,
    deg=deg,
)

fitter.fit()

for der in range(0, deg + 1):
    fitter.plot_fields(der=der)

fitter.plot_integrated_fields()