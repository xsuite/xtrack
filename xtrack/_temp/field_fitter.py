from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import scipy as sc
import math

from scipy.signal import find_peaks

import xtrack as xt



class FieldFitter:
    '''
    Fit on-axis field data and transverse derivatives using piecewise polynomials.

    The fitting pipeline extracts on-axis data, identifies longitudinal regions,
    and fits per-region polynomials to produce spline parameters stored in
    ``df_fit_pars``.

    Parameters
    ----------
    raw_data :
        Either a file path (``str`` or ``pathlib.Path``) to a whitespace-
        separated file with six columns ``X Y Z Bx By Bs``, or a
        ``pd.DataFrame`` with MultiIndex ``('X', 'Y', 'Z')`` and columns
        ``('Bx', 'By', 'Bs')``.
    xy_point :
        On-axis transverse point ``(X, Y)`` in meters, used to select the
        longitudinal series for fitting. Because the input coordinates are
        multiplied by ``ds`` at import time, ``xy_point`` must be given in
        post-scaling (meter) coordinates.
    dx, dy :
        Transverse grid spacing in meters, used for derivative tolerance
        checks.
    ds :
        Coordinate scale factor applied to the X, Y, and Z index levels of
        the input data to convert them to meters.  For example, set
        ``ds=0.001`` when the input coordinates are in millimetres.
    min_region_size :
        Minimum number of points per longitudinal fitting region.
    deg :
        Maximum transverse derivative order to compute and fit.

    DataFrames
    ----------
    df_raw_data :
        Raw data DataFrame with MultiIndex ``('X', 'Y', 'Z')`` and columns
        ``('Bx', 'By', 'Bs')``.
    df_on_axis_raw :
        On-axis raw data DataFrame with MultiIndex ``('X', 'Y', 'Z')`` and columns
        ``('Bx', 'By', 'Bs')``. Filled with the on-axis data from the raw data and
        the on-axis transverse derivatives from fit_transverse_polynomials.
    df_on_axis_fit :
        On-axis fit data DataFrame with MultiIndex ``('X', 'Y', 'Z')`` and columns
        ``('Bx', 'By', 'Bs')``. Filled with the on-axis fit data from the raw data and
        the on-axis transverse derivatives from fit_transverse_polynomials.
    df_fit_pars :
        Fit parameters DataFrame with MultiIndex ``('field_component', 'derivative_x', 'region_name', 's_start', 's_end', 'idx_start', 'idx_end', 'param_index')``.
        Filled with the fit parameters for each polynomial piece for each field and derivative.
    '''

    def __init__(
            self,
            raw_data,
            xy_point=(0, 0),
            dx=0.001,
            dy=0.001,
            ds=0.001,
            min_region_size=10,
            deg=2,
    ):

        # Parameters
        self.xy_point = xy_point
        self.dx, self.dy, self.ds = dx, dy, ds
        self.poly_order = 4  # fixed at 4 for now (5 coefficients)
        self.min_region_size = min_region_size
        self.s_full = None
        self.length = None
        self.deg = deg
        self.field_tol = 1e-3

        # DataFrames
        self.df_raw_data = None
        self.df_on_axis_raw  = None
        self.df_on_axis_fit = None
        self.df_fit_pars = None
        self._set_raw_data(raw_data)

    # PUBLIC
    # Method that calls all the other methods to arrive at a fit.
    def fit(self):
        if self.df_raw_data is None:
            raise RuntimeError("Raw data must be provided before calling fit().")
        self._set_df_on_axis()
        self._find_regions()
        self._fit_slices()



    @staticmethod
    def _poly(s0, s1, coeffs):
        """
        Build a 4th-order spline polynomial over [s0, s1] from boundary data.

        This is a convenience wrapper around ``xt.SplineBoris.spline_poly``.
        See that method for full documentation.
        """
        return xt.SplineBoris.spline_poly(s0, s1, coeffs)



    # PRIVATE
    # This method stores raw data and extracts on-axis data.
    def _set_raw_data(self, raw_data):
        """
        Set the raw data DataFrame, scale coordinates to meters, and compute ``s_full``.

        After loading the DataFrame, the X, Y, and Z index levels are
        multiplied by ``self.ds`` so that all downstream code operates in
        metres.

        Parameters
        ----------
        raw_data :
            Either a file path (``str`` or ``pathlib.Path``) to a whitespace-
            separated file with six columns ``X Y Z Bx By Bs``, or a
            ``pd.DataFrame`` with MultiIndex ``('X', 'Y', 'Z')`` and columns
            ``('Bx', 'By', 'Bs')``.
        """

        if isinstance(raw_data, (str, pathlib.Path)):
            df_raw_data = pd.read_csv(
                raw_data, sep=r"\s+", header=None,
                names=["X", "Y", "Z", "Bx", "By", "Bs"],
            )
            df_raw_data.set_index(["X", "Y", "Z"], inplace=True)
        elif isinstance(raw_data, pd.DataFrame):
            df_raw_data = raw_data
        else:
            raise TypeError(
                f"raw_data must be a file path (str/Path) or a pd.DataFrame, "
                f"got {type(raw_data).__name__}"
            )

        self.df_raw_data = df_raw_data

        # Convert coordinates to meters (e.g. ds=1e-3 for mm input)
        idx = self.df_raw_data.index
        self.df_raw_data.index = pd.MultiIndex.from_arrays(
            [idx.get_level_values(lvl).astype(float) * self.ds for lvl in idx.names],
            names=idx.names,
        )

        self.s_full = np.sort(self.df_raw_data.index.get_level_values("Z").unique()).astype(float)

        # Check if Bs is much smaller than Bx and By
        # Sets an additional index der = 0.
        der = 0
        df_on = self.df_raw_data.xs(self.xy_point, level=("X", "Y")).sort_index().copy(deep=True)
        # convert columns to MultiIndex (field, derivative)
        df_on.columns = pd.MultiIndex.from_tuples([(col, der) for col in df_on.columns])
        self.df_on_axis_raw = df_on

    def save_fit_pars(self, file_path):
        """
        Save the fit parameters DataFrame to a CSV file.

        The DataFrame has a MultiIndex with levels ``('field_component', 'derivative_x', 'region_name', 's_start', 's_end', 'idx_start', 'idx_end', 'param_index')``.
        """
        self.df_fit_pars.to_csv(file_path, index=True)

    # PRIVATE
    # This method extracts on-axis data from the raw DataFrame and fits it to polynomials.
    # It computes the derivatives of said polynomials and stores them in the self.df_on_axis_raw DataFrame.
    # The data is not "raw" in the technical sense, but is used to fit a function of s to.
    def _set_df_on_axis(self):
        """
        Extract on-axis data and compute transverse derivatives.

        The on-axis data is extracted from the raw DataFrame and stored in the self.df_on_axis_raw DataFrame.
        The transverse derivatives are computed and stored in the self.df_on_axis_raw DataFrame.
        """

        # Re-extract on-axis data from raw data so that fit() is idempotent
        df_on = self.df_raw_data.xs(self.xy_point, level=("X", "Y")).sort_index().copy(deep=True)
        df_on.columns = pd.MultiIndex.from_tuples([(col, 0) for col in df_on.columns])
        self.df_on_axis_raw = df_on
        # compute transverse derivatives for der 1..deg and add as columns (skip Bs derivatives)
        self._fit_transverse_polynomials()

        # create a zeros-only DataFrame with the same index/columns as the on-axis raw data
        self.df_on_axis_fit = self.df_on_axis_raw.copy(deep=True)
        # set all values to 0.0 while preserving index and column structure
        self.df_on_axis_fit.loc[:, :] = 0.0

    # PRIVATE
    # This method loops over all fields and derivatives.
    # It first checks if a field/derivative needs fitting based on its maximum value compared to the maximum of the main field.
    # It finds peaks and valleys in the data within the peak_window, with specified width and prominence.
    # It uses these extrema to define regions for polynomial fitting.
    # Then, it cuts regions if they span too wide a range.
    # Finally, it stores the regions in the df_fit_pars DataFrame.
    def _find_regions(self):
        """
        Identify regions for polynomial fitting.

        This method first checks if a field/derivative needs fitting based on its maximum value compared to the maximum of the main field.
        It finds peaks and valleys in the data within the peak_window, with specified width and prominence.
        It uses these extrema to define regions for polynomial fitting.
        Then, it cuts regions if they span too wide a range.
        Finally, it stores the regions in the df_fit_pars DataFrame.
        """

        fields = ["Bx", "By", "Bs"]

        abs_max = 0
        for field in fields:
            series = self.df_on_axis_raw[(field, 0)].values
            field_max = np.max(np.abs(series))
            if field_max > abs_max:
                abs_max = field_max

        for field in fields:
            # Bs only has der = 0; other fields range 0..deg
            ders = [0] if field == "Bs" else range(0, self.deg + 1)

            for der in ders:
                series = self.df_on_axis_raw[(field, der)].values

                # FIELD TOLERANCE AREA: check if this field/derivative needs fitting
                field_der_max = np.max(np.abs(series))
                relative_max = 1 / math.factorial(der) * field_der_max * (self.dx ** der)
                if relative_max < self.field_tol * abs_max:
                    # set to single region with zero parameters and skip expensive processing
                    field_extrema = np.array([0, len(series) - 1], dtype=int)
                    to_fit = False
                else:
                    # SPLIT REGIONS AREA
                    # choose prominence: more permissive for Bs
                    std_series = np.std(series)
                    prominence = 0.5 * std_series if field == "Bs" else std_series

                    field_peaks = find_peaks(series, width=15, prominence=prominence)[0]
                    field_valleys = find_peaks(-series, width=15, prominence=prominence)[0]
                    field_extrema = np.sort(np.concatenate((field_peaks, field_valleys)))

                    # include endpoints
                    field_extrema = np.insert(field_extrema, 0, 0)
                    field_extrema = np.append(field_extrema, len(series) - 1)

                    # split long regions while ensuring each part has at least `min_region_size` points
                    this_min_region_size = self.min_region_size if der == 0 else self.min_region_size // 2
                    new_extrema = [int(field_extrema[0])]
                    for left, right in zip(field_extrema[:-1], field_extrema[1:]):
                        length = int(right - left)
                        if length < 2 * this_min_region_size:
                            new_extrema.append(int(right))
                            continue
                        n_parts = int(np.floor(length / this_min_region_size))
                        if n_parts <= 1:
                            new_extrema.append(int(right))
                            continue
                        splits = np.round(np.linspace(left, right, n_parts + 1)).astype(int)
                        for s_split in splits[1:]:
                            if s_split > new_extrema[-1]:
                                new_extrema.append(int(s_split))

                    field_extrema = np.unique(np.asarray(new_extrema, dtype=int))
                    to_fit = True

                # number of pieces is number of extrema - 1 (ensure at least 1)
                n_pieces = max(1, len(field_extrema) - 1)
                print(f"{field} der={der} -> n_pieces={n_pieces}")
                self._set_df_fit_pars(der, n_pieces, field, field_extrema, to_fit)


        self.df_fit_pars.set_index(['field_component', 'derivative_x', 'region_name', 's_start', 's_end', 'idx_start', 'idx_end', 'param_index'],
                                       inplace=True)

        # ensure MultiIndex is lexsorted so partial-key .loc lookups (e.g. .loc[(field, der)]) are fast and avoid PerformanceWarning
        if not self.df_fit_pars.empty:
            self.df_fit_pars.sort_index(inplace=True)

        #with pd.option_context('display.max_columns', None, 'display.max_rows', None, 'display.width', None):
            #print(self.df_fit_pars)

    # PRIVATE
    # This method initializes and appends rows to the df_fit_pars DataFrame.
    # Each row corresponds to a polynomial piece for a specific field and derivative.
    # It stores metadata about the piece, including parameter names and initial values.
    # This method is called by _find_regions to populate the DataFrame.
    # In case the set consists of only one piece, the parameters are initialized to 0.
    def _set_df_fit_pars(self, der_order, n_pieces, field, idx_extrema, to_fit=True):
        """
        Initialize and append rows to the df_fit_pars DataFrame.

        Each row corresponds to a polynomial piece for a specific field and derivative.
        It stores metadata about the piece, including parameter names and initial values.
        This method is called by _find_regions to populate the DataFrame.
        In case the set consists of only one piece, the parameters are initialized to 0.
        """

        rows = []
        # Zero-pad index so alphabetical sort matches numerical sort
        index_width = len(str(n_pieces - 1)) if n_pieces > 1 else 1
        for i in range(n_pieces):
            pars = xt.SplineBoris.ParamFormat.fieldfitter_param_names(
                field, der_order, self.poly_order
            )

            idx_start = idx_extrema[i]
            idx_end = idx_extrema[i+1]
            s_start = self.s_full[idx_start]
            s_end = self.s_full[idx_end]

            for idx, name in enumerate(pars):
                rows.append({
                    "field_component": field,
                    "derivative_x": der_order,
                    "region_name": f"Poly_{i:0{index_width}d}",
                    "s_start": s_start,
                    "s_end": s_end,
                    "idx_start": idx_start,
                    "idx_end": idx_end,
                    "param_index": idx,
                    "param_name": name,
                    "param_value": 0 if not to_fit else None,
                    "to_fit": to_fit,
                })

        results = pd.DataFrame(rows)
        self.df_fit_pars = pd.concat([self.df_fit_pars, results])


    
    def _boundary_from_poly(self, sL, poly):
        """
        Compute the boundary conditions from a previously fitted polynomial.

        Because the fitting is done from left to right, this method is used to compute the boundary conditions from the previous polynomial.
        Accepts the polynomial and the position sL where to evaluate it (we fit from left to right, so always the leftmost point).
        """

        dp = poly.deriv()
        return np.array([poly(sL), dp(sL)], dtype=float)

    def _boundary_from_finite_differences(self, b_region, s_region, get_right_point=True):
        """
        Compute the boundary conditions from finite differences in the specified region.

        Because the fitting is done from left to right, this method is used to compute the boundary conditions from the data on the rightmost point of the region.
        Accepts the field values in the region and the longitudinal spacing.
        """

        # Compute actual spacing from s_region (handle non-uniform spacing by using local spacing)
        if get_right_point:
            # Use spacing near the right boundary
            if len(s_region) >= 3:
                h = s_region[-1] - s_region[-2]  # Local spacing at right boundary
                if h == 0 and len(s_region) >= 4:
                    h = s_region[-2] - s_region[-3]  # Fallback to previous spacing
            else:
                h = self.ds  # Fallback if not enough points
            dbR = (3 * b_region[-1] - 4 * b_region[-2] + b_region[-3]) / (2 * h)
            return np.array([b_region[-1], dbR], dtype=float)
        else:
            # Use spacing near the left boundary
            if len(s_region) >= 3:
                h = s_region[1] - s_region[0]  # Local spacing at left boundary
                if h == 0 and len(s_region) >= 4:
                    h = s_region[2] - s_region[1]  # Fallback to next spacing
            else:
                h = self.ds  # Fallback if not enough points
            dbL = (-3 * b_region[0] + 4 * b_region[1] - b_region[2]) / (2 * h)
            return np.array([b_region[0], dbL], dtype=float)



    def _fit_single_poly(self, field, der_order, sub_df_this, sub_df_prev=None):
        """
        Fit a single polynomial piece to the specified region of data.

        This method fits a single polynomial piece to the specified region of data.
        It accepts the field, derivative order, the current region DataFrame, and the previous region DataFrame.
        It computes the boundary conditions from the previous polynomial and the data on the rightmost point of the region.
        It then fits a polynomial (see _poly) to the data in the region and stores the coefficients in the df_fit_pars DataFrame.
        """

        idx_left = int(sub_df_this.index.get_level_values('idx_start')[0])
        idx_right = int(sub_df_this.index.get_level_values('idx_end')[0])
        s_left = float(sub_df_this.index.get_level_values('s_start')[0])
        s_right = float(sub_df_this.index.get_level_values('s_end')[0])

        s_region = self.s_full[idx_left:idx_right + 1]
        b_region = self.df_on_axis_raw[(field, der_order)].values[idx_left:idx_right + 1]
        integral = sc.integrate.trapezoid(b_region, s_region)

        if sub_df_prev is not None:
            # Extract coefficients, filtering out None values and sorting by param_index
            # Only use parameters that have been fitted (param_value is not None)
            sub_df_prev_fitted = sub_df_prev[sub_df_prev['param_value'].notna()].sort_values('param_index')
            if len(sub_df_prev_fitted) == 0:
                # No fitted parameters yet, use finite differences instead
                left_bounds = self._boundary_from_finite_differences(b_region, s_region, get_right_point=False)
            else:
                coeff_prev = sub_df_prev_fitted['param_value'].values
                poly = np.polynomial.Polynomial(coeff_prev)
                left_bounds = self._boundary_from_poly(s_left, poly)
        else:
            left_bounds = self._boundary_from_finite_differences(b_region, s_region, get_right_point=False)

        right_bounds = self._boundary_from_finite_differences(b_region, s_region, get_right_point=True)
        coeffs = (left_bounds[0], left_bounds[1], right_bounds[0], right_bounds[1], integral)

        poly = self._poly(s_left, s_right, coeffs)

        # Convert to standard form and ensure we have the expected number of coefficients
        poly_std = poly.convert()
        coef = poly_std.coef
        # Pad with zeros if necessary (poly.convert() may drop trailing zeros)
        expected_len = self.poly_order + 1
        if len(coef) < expected_len:
            coef = np.pad(coef, (0, expected_len - len(coef)), mode='constant')

        # Assign coefficients into df_fit_pars
        for i in range(self.poly_order + 1):
            param_value = coef[i]  # get coefficient of s^i
            self.df_fit_pars.at[(field, der_order, sub_df_this['region_name'].iloc[0], s_left, s_right, idx_left, idx_right, i), 'param_value'] = param_value
        
        # Use .loc instead of .values to avoid read-only array error
        # Get the index slice for the region
        idx_slice = self.df_on_axis_fit.index[idx_left:idx_right + 1]
        self.df_on_axis_fit.loc[idx_slice, (field, der_order)] = poly(s_region)

    # PRIVATE
    # This method loops over all fields and derivatives and fits polynomials to each region.
    def _fit_slices(self):
        """
        Fit polynomials to each region for all fields and derivatives.

        This method loops over all fields and derivatives and fits polynomials to each region.
        It skips the derivatives of Bs and the regions that do not need fitting.
        It then fits a polynomial (see _fit_single_poly) to each region and stores the coefficients in the df_fit_pars DataFrame.
        """

        for field in ["Bx", "By", "Bs"]:
            for der in range(0, self.deg + 1):
                if field == "Bs" and der > 0:
                    continue

                print(f"Fitting field {field} derivative {der}")
                sub_df = self.df_fit_pars.loc[(field, der)]

                if not sub_df['to_fit'].any():
                    continue

                sub_df.reset_index(level='region_name', inplace=True)
                n_regions = sub_df['region_name'].nunique()
                index_width = len(str(n_regions - 1)) if n_regions > 1 else 1

                for i in range(n_regions):
                    sub_df_this = sub_df[sub_df['region_name'] == f"Poly_{i:0{index_width}d}"]
                    if i == 0:
                        sub_df_prev = None
                    else:
                        sub_df_prev = sub_df[sub_df['region_name'] == f"Poly_{i - 1:0{index_width}d}"]

                    self._fit_single_poly(field, der, sub_df_this, sub_df_prev)



    # TODO: Also allow y derivatives and use Maxwell's Equations to get the right sign.
    def _fit_transverse_polynomials(self):
        """
        Fit transverse polynomials and compute all derivatives at ``self.xy_point``.

        This method fits a polynomial of degree ``self.deg`` to the transverse
        field variation at each longitudinal position, using every X value
        present in the input data at the Y coordinate of ``self.xy_point``.
        It then evaluates all derivatives from order 1 to ``self.deg`` at
        the X coordinate of ``self.xy_point`` and stores them in
        ``df_on_axis_raw``.
        """
        x_point, y_point = self.xy_point

        idx = self.df_raw_data.index
        ys = idx.get_level_values("Y")
        xs = idx.get_level_values("X")
        mask = ys == y_point
        points = sorted(set(xs[mask]))

        subsets = {px: self.df_raw_data.xs((px, y_point), level=["X", "Y"]).sort_index() for px in points}

        for field in ["Bx", "By"]:
            x = points
            n = len(subsets[points[0]][field])
            derivs = {der: np.zeros(n) for der in range(1, self.deg + 1)}

            for i in range(n):
                B_i = [subsets[px][field].to_numpy()[i] for px in points]
                coeffs = np.polyfit(x, B_i, self.deg)
                for der in range(1, self.deg + 1):
                    d_coeffs = np.polyder(coeffs, m=der)
                    derivs[der][i] = np.polyval(d_coeffs, x_point)

            for der in range(1, self.deg + 1):
                self.df_on_axis_raw[(field, der)] = derivs[der]



    def plot_integrated_fields(self):
        """
        Plot the integrated fields for the raw and fit data.

        This method plots the integrated fields for the raw and fit data.
        It accepts the derivative order.
        It computes the derivatives of the polynomials and stores them in the df_on_axis_raw DataFrame.
        """
        import matplotlib.pyplot as plt

        if self.df_on_axis_raw is None or self.df_on_axis_fit is None:
            raise RuntimeError("`df_on_axis_raw` and `df_on_axis_fit` must be set before plotting.")

        s = self.s_full

        Bx_raw = self.df_on_axis_raw[('Bx', 0)].to_numpy()
        By_raw = self.df_on_axis_raw[('By', 0)].to_numpy()
        try:
            Bs_raw = self.df_on_axis_raw[('Bs', 0)].to_numpy()
        except KeyError:
            Bs_raw = np.zeros_like(Bx_raw)

        Bx_fit = self.df_on_axis_fit[('Bx', 0)].to_numpy()
        By_fit = self.df_on_axis_fit[('By', 0)].to_numpy()
        try:
            Bs_fit = self.df_on_axis_fit[('Bs', 0)].to_numpy()
        except KeyError:
            Bs_fit = np.zeros_like(Bx_fit)

        fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)

        Bx_int_raw = sc.integrate.cumulative_trapezoid(Bx_raw, x=s, initial=0)
        By_int_raw = sc.integrate.cumulative_trapezoid(By_raw, x=s, initial=0)
        Bs_int_raw = sc.integrate.cumulative_trapezoid(Bs_raw, x=s, initial=0)

        Bx_int_fit = sc.integrate.cumulative_trapezoid(Bx_fit, x=s, initial=0)
        By_int_fit = sc.integrate.cumulative_trapezoid(By_fit, x=s, initial=0)
        Bs_int_fit = sc.integrate.cumulative_trapezoid(Bs_fit, x=s, initial=0)

        ax1.plot(s, Bx_int_raw, label='Raw Data')
        ax1.plot(s, Bx_int_fit, label='Fit', linestyle='--')
        ax2.plot(s, By_int_raw, label='Raw Data')
        ax2.plot(s, By_int_fit, label='Fit', linestyle='--')
        ax3.plot(s, Bs_int_raw, label='Raw Data')
        ax3.plot(s, Bs_int_fit, label='Fit', linestyle='--')

        # Vertical border lines removed

        ax1.set_title(f"Integrated Magnetic Field at (X, Y) = {self.xy_point}")
        ax1.set_ylabel(r"Integrated Horizontal Field, $\int B_x \, ds$ [T·m]")
        ax2.set_ylabel(r"Integrated Vertical Field, $\int B_y \, ds$ [T·m]")
        ax3.set_ylabel(r"Integrated Longitudinal Field, $\int B_s \, ds$ [T·m]")
        ax3.set_xlabel(r"Longitudinal Position, $s$ [m]")

        ax1.legend(loc="lower right")
        ax2.legend(loc="lower right")
        ax3.legend(loc="upper right")

        ax1.grid()
        ax2.grid()
        ax3.grid()

        plt.show()

    def plot_fields(self, der=0):
        """
        Plot the data against the fit.

        This method plots the data against the fit.
        It accepts the derivative order.
        It computes the derivatives of the polynomials and stores them in the df_on_axis_raw DataFrame.
        """
        import matplotlib.pyplot as plt

        if self.df_on_axis_raw is None or self.df_on_axis_fit is None:
            raise RuntimeError("`df_on_axis_raw` and `df_on_axis_fit` must be set before plotting.")

        s = self.s_full
        fig1, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 4), constrained_layout=True)

        def get_series(df, field, der):
            try:
                return df[(field, der)].to_numpy()
            except KeyError:
                # fallback to zeros if Bs not present or derivative missing
                ref = df.iloc[:, 0].to_numpy()
                return np.zeros_like(ref)

        ax1.plot(s, get_series(self.df_on_axis_raw, "Bx", der), label='Raw Data')
        ax1.plot(s, get_series(self.df_on_axis_fit, "Bx", der), label='Fit', linestyle='--')
        ax2.plot(s, get_series(self.df_on_axis_raw, "By", der), label='Raw Data')
        ax2.plot(s, get_series(self.df_on_axis_fit, "By", der), label='Fit', linestyle='--')
        ax3.plot(s, get_series(self.df_on_axis_raw, "Bs", der), label='Raw Data')
        ax3.plot(s, get_series(self.df_on_axis_fit, "Bs", der), label='Fit', linestyle='--')

        # compute border indices per field/derivative (fall back to existing attribute if absent)
        def _borders_for_field(field_ax):
            if getattr(self, "df_fit_pars", None) is None:
                return getattr(self, "borders_idx", []) or []
            try:
                lvl_field = np.asarray(self.df_fit_pars.index.get_level_values('field_component'))
                lvl_der = np.asarray(self.df_fit_pars.index.get_level_values('derivative_x')).astype(int)
                mask = (lvl_field == field_ax) & (lvl_der == int(der))
                if not np.any(mask):
                    return []
                s_start_vals = np.asarray(self.df_fit_pars.index.get_level_values('s_start'))[mask].astype(float)
                s_end_vals = np.asarray(self.df_fit_pars.index.get_level_values('s_end'))[mask].astype(float)
                s_borders = np.unique(np.concatenate((s_start_vals, s_end_vals)))
                s_arr = np.asarray(s)
                return sorted({int(np.argmin(np.abs(s_arr - float(sb)))) for sb in s_borders})
            except Exception:
                return getattr(self, "borders_idx", []) or []

        for field_ax in ["Bx", "By", "Bs"]:
            ax = {"Bx": ax1, "By": ax2, "Bs": ax3}[field_ax]
            borders_idx_field = _borders_for_field(field_ax)
            for idx in borders_idx_field or []:
                if 0 <= idx < len(s):
                    ax.axvline(x=s[idx], color='k', linestyle='--', linewidth=1, alpha=0.3)

        if der == 2:
            x_label = r"$\frac{d^2 B_x}{d x^2}$"
            y_label = r"$\frac{d^2 B_y}{d x^2}$"
            s_label = r"$\frac{d^2 B_s}{d x^2}$"
        elif der == 1:
            x_label = r"$\frac{d B_x}{d x}$"
            y_label = r"$\frac{d B_y}{d x}$"
            s_label = r"$\frac{d B_s}{d x}$"
        else:
            x_label = r"$B_x$"
            y_label = r"$B_y$"
            s_label = r"$B_s$"

        ax1.set_title(f"Magnetic Field at (X, Y) = {self.xy_point}")
        ax1.set_ylabel(f"Horizontal Field, {x_label} [T]")
        ax2.set_ylabel(f"Vertical Field, {y_label} [T]")
        ax3.set_ylabel(f"Longitudinal Field, {s_label} [T]")
        ax3.set_xlabel(r"Longitudinal Position, $s$ [m]")

        ax1.legend([f"{x_label} Data", f"{x_label} Fit"], loc="lower right")
        ax2.legend([f"{y_label} Data", f"{y_label} Fit"], loc="lower right")
        ax3.legend([f"{s_label} Data", f"{s_label} Fit"], loc="upper right")

        ax1.grid()
        ax2.grid()
        ax3.grid()

        plt.show()