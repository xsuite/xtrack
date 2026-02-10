from pathlib import Path

# Simple local import so this file can be run directly (e.g. via IDE "Run" button)
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

here = Path(__file__).resolve().parent

# Standard 6-column format (X Y Z Bx By Bs)
# Use example knot-map file in the local `field_maps` folder
file_path = Path(__file__).parent / "example_data" / "knot_map_test.txt"

# Where to write the fitted spline coefficients
output_dir = Path(__file__).parent / "example_data"
output_dir.mkdir(parents=True, exist_ok=True)

deg = 2

if __name__ == "__main__":
    # Build and run the fitter (file path is parsed inline by FieldFitter)
    fitter = FieldFitter(
        raw_data=file_path,
        xy_point=(0.0, 0.0),
        dx=dz,
        dy=dz,
        ds=dz,
        min_region_size=10,
        deg=deg,
    )

    fitter.fit()
    
    for der in range(0, deg + 1):
        fitter.plot_fields(der=der)

    fitter.plot_integrated_fields()