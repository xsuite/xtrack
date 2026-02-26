from pathlib import Path

import numpy as np

from xtrack._temp.boris_and_solenoid_map.solenoid_field import SolenoidField
from xtrack._temp.field_fitter import FieldFitter


def main():
    interval = 30
    dx = 0.001
    dy = 0.001
    multipole_order = 2
    n_steps = 5000

    out_dir = Path(__file__).resolve().parent
    fieldmap_path = out_dir / "solenoid_field.dat"
    fit_pars_path = out_dir / "solenoid_fit_pars.csv"

    sf = SolenoidField(L=4, a=0.3, B0=1.5, z0=20)

    x_axis = np.linspace(
        -multipole_order * dx / 2, multipole_order * dx / 2, multipole_order + 1
    )
    y_axis = np.linspace(
        -multipole_order * dy / 2, multipole_order * dy / 2, multipole_order + 1
    )
    z_axis = np.linspace(0, interval, n_steps + 1)

    x_grid, y_grid, z_grid = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")
    bx, by, bz = sf.get_field(x_grid.ravel(), y_grid.ravel(), z_grid.ravel())
    data = np.column_stack(
        [x_grid.ravel(), y_grid.ravel(), z_grid.ravel(), bx.ravel(), by.ravel(), bz.ravel()]
    )
    np.savetxt(fieldmap_path, data)

    fitter = FieldFitter(
        raw_data=fieldmap_path,
        xy_point=(0, 0),
        distance_unit=1,
        min_region_size=10,
        deg=multipole_order - 1,
    )
    fitter.field_tol = 1e-4
    fitter.fit()
    fitter.save_fit_pars(fit_pars_path)

    print(f"Wrote {fieldmap_path}")
    print(f"Wrote {fit_pars_path}")


if __name__ == "__main__":
    main()
