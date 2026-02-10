from pathlib import Path

# Simple local import so this file can be run directly (e.g. via IDE "Run" button)
from field_fitter import FieldFitter


########################################################################################################################
# Fit field map in `field_maps` using FieldFitter
########################################################################################################################

dz = 0.001  # Step size in the z (longitudinal) direction for numerical differentiation

here = Path(__file__).resolve().parent

# Standard 6-column format (X Y Z Bx By Bs)
# Use example knot-map file in the local `field_maps` folder
file_path = here / "field_maps" / "knot_map_test.txt"

# Where to write the fitted spline coefficients
output_dir = here / "field_maps"
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
    for der in range(1, deg + 1):
        fitter.plot_fields(der=der)

    fitter.plot_integrated_fields()