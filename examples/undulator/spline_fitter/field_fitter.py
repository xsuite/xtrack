from __future__ import annotations

import numpy as np
import pandas as pd
import scipy as sc
import sympy as sp
import xtrack as xt
import bisect
import math
from functools import partial

import bpmeth as bp
import time
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

import cProfile

from sympy.utilities.codegen import codegen

# Import the parsers from the local `fieldmap_parsers` module.
# This works when running these example scripts directly from their directory
# (e.g. via an IDE "Run" button), because Python puts this directory on sys.path.
from fieldmap_parsers import (
    FieldMapParser,
    StandardFieldMapParser,
    SolenoidFieldMapParser,
    get_parser_for_file,
)

class FieldFitter:

    def __init__(
            self,
            file_path,
            parser=None,  # New: parser instance or format name
            xy_point=(0, 0),
            dx=0.001,
            dy=0.001,
            ds=0.001,
            min_region_size=10,
            deg=2,
    ):

        # Parameters
        self.file_path = file_path
        self.xy_point = xy_point
        self.dx, self.dy, self.ds = dx, dy, ds
        self.poly_order = 4  # fixed at 4 for now (5 coefficients)
        self.min_region_size = min_region_size
        self.s_full = None
        self.length = None
        self.deg = deg
        self.field_tol = 1e-3

        # Parser setup
        if parser is None:
            self.parser = self._get_default_parser()
        elif isinstance(parser, FieldMapParser):
            self.parser = parser
        else:
            raise TypeError(f"parser must be a FieldMapParser instance or None, got {type(parser)}")

        # DataFrames
        self.df_raw_data = None
        self.df_on_axis_raw  = None
        self.df_on_axis_fit = None
        self.df_fit_pars = None

    # PUBLIC
    # Setter method that calls all the other methods to arrive at a fit.
    def set(self):
        self._parse_to_dataframe()
        self._set_df_on_axis()
        self._find_regions()
        self._fit_slices()



    ####################################################################################################################
    # EVALUATION FUNCTIONS
    ####################################################################################################################

    # PRIVATE
    # Polynomials, which coefficients are determined by the boundary conditions and integral over the interval.
    # c1 = f(s0)
    # c2 = f'(s0)
    # c3 = f(s1)
    # c4 = f'(s1)
    # c5 = integral from s0 to s1 of f(s) ds
    # TODO: Consider making this dynamic in self.poly_order.
    @staticmethod
    def _poly(s0, s1, coeffs):
        c1, c2, c3, c4, c5 = coeffs
        L = s1 - s0
        t = np.polynomial.Polynomial([-s0 / L, 1 / L])

        # basis functions on [0,1]
        b1_coeffs = [1, 0, -18, 32, -15]
        b2_coeffs = [0, 1, -4.5, 6, -2.5]
        b3_coeffs = [0, 0, -12, 28, -15]
        b4_coeffs = [0, 0, 1.5, -4, 2.5]
        b5_coeffs = [0, 0, 30, -60, 30]
        b1_poly = np.polynomial.Polynomial(b1_coeffs)
        b2_poly = np.polynomial.Polynomial(b2_coeffs)
        b3_poly = np.polynomial.Polynomial(b3_coeffs)
        b4_poly = np.polynomial.Polynomial(b4_coeffs)
        b5_poly = np.polynomial.Polynomial(b5_coeffs)

        # combine with correct scaling for derivatives/integral
        poly_t = c1 * b1_poly + L * c2 * b2_poly + c3 * b3_poly + L * c4 * b4_poly + (c5 / L) * b5_poly
        poly_s = poly_t(t)
        return poly_s



    ####################################################################################################################
    # IDENTIFYING REGIONS AND SETTING BORDERS IN DATA CLASSES
    ####################################################################################################################
    # PRIVATE
    # This method reads the data from the file and stores it in a pandas DataFrame.
    def _parse_to_dataframe(self):
        # Use parser to parse the file
        self.df_raw_data = self.parser.parse(self.file_path, dx=self.dx, dy=self.dy)
        self.s_full = np.sort(self.df_raw_data.index.get_level_values("Z").unique()).astype(float) * self.ds

        # Check if Bs is much smaller than Bx and By
        # Sets an additional index der = 0.
        der = 0
        df_on = self.df_raw_data.xs(self.xy_point, level=("X", "Y")).sort_index().copy(deep=True)
        # convert columns to MultiIndex (field, derivative)
        df_on.columns = pd.MultiIndex.from_tuples([(col, der) for col in df_on.columns])
        self.df_on_axis_raw = df_on

    # PRIVATE
    # Auto-detect format or use standard parser.
    def _get_default_parser(self):
        """Auto-detect format or use standard parser."""
        # Try to auto-detect format
        try:
            return get_parser_for_file(self.file_path)
        except (ValueError, FileNotFoundError):
            # Fall back to standard parser if auto-detection fails or file not found
            # The file will be checked again during actual parsing
            return StandardFieldMapParser()

    def save_fit_pars(self, file_path):
        self.df_fit_pars.to_csv(file_path, index=True)

    def get_parameter_table(self, n_steps, multipole_order=None):
        """
        Convert fit parameters DataFrame to parameter table format for SplineBoris.
        
        Parameters are ordered as expected by the C code:
        - For multipole_order: bs_*, kn_1_*, ..., kn_N_*, ks_1_*, ..., ks_N_*
        - Within each group: ordered by polynomial order (0, 1, 2, 3, 4)
        
        Parameters
        ----------
        n_steps : int
            Number of steps in the parameter table
        multipole_order : int, optional
            Multipole order. If None, inferred from self.deg (multipole_order = deg)
        
        Returns
        -------
        par_table : list
            List of parameter vectors, each containing all parameters for one step
        s_start : float
            Starting s position
        s_end : float
            Ending s position
        """
        if self.df_fit_pars is None:
            raise RuntimeError("Fit parameters not available. Call set() first.")
        
        # Infer multipole_order from deg if not provided
        if multipole_order is None:
            multipole_order = self.deg
        
        # Get s boundaries from fit parameters
        df_reset = self.df_fit_pars.reset_index()
        s_starts = np.sort(df_reset['s_start'].to_numpy(dtype=np.float64))
        s_ends = np.sort(df_reset['s_end'].to_numpy(dtype=np.float64))
        s_boundaries = np.sort(np.unique(np.concatenate((s_starts, s_ends))))
        s_start = float(s_boundaries[0])
        s_end = float(s_boundaries[-1])
        
        # Build expected parameter order
        expected_params = []
        # First: bs_ parameters (no multipole index)
        for poly_idx in range(5):
            expected_params.append(f"bs_{poly_idx}")
        # Second: kn_ parameters (all multipoles)
        for multipole_idx in range(1, multipole_order + 1):
            for poly_idx in range(5):  # 0 to 4
                expected_params.append(f"kn_{multipole_idx}_{poly_idx}")
        # Third: ks_ parameters (all multipoles)
        for multipole_idx in range(1, multipole_order + 1):
            for poly_idx in range(5):  # 0 to 4
                expected_params.append(f"ks_{multipole_idx}_{poly_idx}")
        
        # Create s values
        s_vals = np.linspace(s_start, s_end, n_steps)
        
        # Build parameter table
        par_table = []
        for s_val_i in s_vals:
            # Filter rows that contain this s position
            # Only use derivative_x=0 (the field itself, not derivatives)
            mask = ((df_reset['s_start'] <= s_val_i) & 
                    (df_reset['s_end'] >= s_val_i) & 
                    (df_reset['derivative_x'] == 0))
            rows = df_reset[mask]
            
            # Create a dictionary from available parameters
            param_dict = {}
            if not rows.empty:
                # Sort by field_component and region to ensure consistent selection
                rows_sorted = rows.copy()
                rows_sorted['region_size'] = rows_sorted['s_end'] - rows_sorted['s_start']
                rows_sorted = rows_sorted.sort_values(['field_component', 'region_size', 'param_name'])
                
                # Take the first occurrence of each param_name
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
            
            par_table.append(param_values)
        
        return par_table, s_start, s_end

    # PRIVATE
    # This method extracts on-axis data from the raw DataFrame and fits it to polynomials.
    # It computes the derivatives of said polynomials and stores them in the self.df_on_axis_raw DataFrame.
    # The data is not "raw" in the technical sense, but is used to fit a function of s to.
    def _set_df_on_axis(self):
        # 0th derivative columns
        self.df_on_axis_raw.columns = pd.MultiIndex.from_tuples([
                    (col[0] if isinstance(col, tuple) else col, 0) for col in self.df_on_axis_raw.columns
                ])
        # compute transverse derivatives for der > 0 and add as columns (skip Bs derivatives)
        for der in range(1, self.deg + 1):
            derivs = self._fit_transverse_polynomials(der=der)
            self.df_on_axis_raw[('Bx', der)] = derivs['Bx']
            self.df_on_axis_raw[('By', der)] = derivs['By']
            # intentionally do not compute/store Bs_{der}, as we are not using them.

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
                        for sp in splits[1:]:
                            if sp > new_extrema[-1]:
                                new_extrema.append(int(sp))

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
        rows = []
        for i in range(n_pieces):
            if field == "Bx":
                pars = [f"ks_{der_order+1}_{k}" for k in range(self.poly_order + 1)]
            elif field == "By":
                pars = [f"kn_{der_order+1}_{k}" for k in range(self.poly_order + 1)]
            else:  # Bs
                pars = [f"bs_{k}" for k in range(self.poly_order + 1)]

            idx_start = idx_extrema[i]
            idx_end = idx_extrema[i+1]
            s_start = self.s_full[idx_start]
            s_end = self.s_full[idx_end]

            for idx, name in enumerate(pars):
                rows.append({
                    "field_component": field,
                    "derivative_x": der_order,
                    "region_name": f"Poly_{i}",
                    "s_start": s_start,
                    "s_end": s_end,
                    "idx_start": idx_start,
                    "idx_end": idx_end,
                    "param_index": idx,
                    "param_name": name,
                    "param_symbol": sp.Symbol(name),
                    "param_value": 0 if not to_fit else None,
                    "to_fit": to_fit,
                })

        results = pd.DataFrame(rows)
        self.df_fit_pars = pd.concat([self.df_fit_pars, results])



    ####################################################################################################################
    # PIECEWISE POLYNOMIAL FITTING
    ####################################################################################################################

    # PRIVATE
    # This method computes the boundary conditions from a previously fitted polynomial.
    # Accepts the polynomial and the position sL where to evaluate it (we fit from left to right, so always the leftmost point).
    def _boundary_from_poly(self, sL, poly):
        dp = poly.deriv()
        return np.array([poly(sL), dp(sL)], dtype=float)

    # PRIVATE
    # This method computes the boundary conditions from finite differences in the specified region.
    def _boundary_from_finite_differences(self, b_region, s_region, get_right_point=True):
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

    # PRIVATE
    # This method fits a single polynomial piece to the specified region of data.
    def _fit_single_poly(self, field, der_order, sub_df_this, sub_df_prev=None):
        idx_left = int(sub_df_this.index.get_level_values('idx_start')[0])
        idx_right = int(sub_df_this.index.get_level_values('idx_end')[0])
        s_left = float(sub_df_this.index.get_level_values('s_start')[0])
        s_right = float(sub_df_this.index.get_level_values('s_end')[0])

        s_region = self.s_full[idx_left:idx_right + 1]
        b_region = self.df_on_axis_raw[(field, der_order)].values[idx_left:idx_right + 1]
        integral = sc.integrate.trapezoid(b_region, s_region)

        if sub_df_prev is not None:
            coeff_prev = sub_df_prev['param_value'].iloc[:].values
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
            self.df_on_axis_fit[(field, der_order)].values[idx_left:idx_right + 1] = poly(s_region)

    # PRIVATE
    # This method loops over all fields and derivatives and fits polynomials to each region.
    def _fit_slices(self):

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

                for i in range(n_regions):
                    sub_df_this = sub_df[sub_df['region_name'] == f"Poly_{i}"]
                    if i == 0:
                        sub_df_prev = None
                    else:
                        sub_df_prev = sub_df[sub_df['region_name'] == f"Poly_{i - 1}"]

                    self._fit_single_poly(field, der, sub_df_this, sub_df_prev)



    ####################################################################################################################
    # TRANSVERSE GRADIENTS
    ####################################################################################################################

    # PRIVATE
    # This method extracts the data at (x,y) = (-1,0), (0,0), (1,0) and fits parabolas to these points.
    # This is done because bpmeth needs the derivatives w.r.t. x at each point.
    # The first derivatives are zero, but can be extracted nevertheless.
    def _fit_transverse_polynomials(self, der=0):
        idx = self.df_raw_data.index
        ys = idx.get_level_values("Y")
        xs = idx.get_level_values("X")
        mask = ys == 0
        points = sorted(set(xs[mask]))

        subsets = {px: self.df_raw_data.xs((px, 0), level=["X", "Y"]).sort_index() for px in points}
        derivs = {"Bx": None, "By": None}

        for field in ["Bx", "By"]:
            x = [p * self.dx for p in points]
            n = len(subsets[points[0]][field])
            derivs[field] = np.zeros(n)

            for i in range(n):
                y = [subsets[px][field].to_numpy()[i] for px in points]
                coeffs = np.polyfit(x, y, self.deg)
                # Compute the der-th derivative at x=0
                d_coeffs = np.polyder(coeffs, m=der)
                # Evaluate at x=0
                derivs[field][i] = np.polyval(d_coeffs, 0)
            # Optionally store the result for later use
            col_name = (f"{field}", der)
            # Ensure df_on_axis_raw exists and assign the derivative column
            self.df_on_axis_raw[col_name] = derivs[field]

        return derivs



    ####################################################################################################################
    # PLOTTING
    ####################################################################################################################

    # PUBLIC
    # Plot the integrated fields.
    def plot_integrated_fields(self):
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

    # PUBLIC
    # Plot the data against the fit.
    # der: derivative order to plot (0 = field, 1 = first derivative, etc.)
    def plot_fields(self, der=0):
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
            y_label = r"$\frac{d^2 B_y}{d y^2}$"
            s_label = r"$\frac{d^2 B_s}{d x^2}$"
        elif der == 1:
            x_label = r"$\frac{d B_x}{d x}$"
            y_label = r"$\frac{d B_y}{d y}$"
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