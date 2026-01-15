from pathlib import Path

# Simple local import so this file can be run directly (e.g. via IDE "Run" button)
from field_fitter import FieldFitter
from fieldmap_parsers import StandardFieldMapParser


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
fit_par_path = output_dir / "field_fit_pars.csv"

if __name__ == "__main__":
    # Parse the field map into the standardized DataFrame
    parser = StandardFieldMapParser()
    df_raw_data = parser.parse(file_path)

    # Build and run the fitter
    fitter = FieldFitter(
        df_raw_data=df_raw_data,
        xy_point=(0.0, 0.0),
        dx=dz,
        dy=dz,
        ds=dz,
        min_region_size=10,
        deg=2,
    )

    fitter.set()
    fitter.save_fit_pars(fit_par_path)

    print(f"Field map fitted successfully.\nParameters saved to: {fit_par_path}")