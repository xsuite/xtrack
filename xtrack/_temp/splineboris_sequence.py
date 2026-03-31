import numpy as np
import pandas as pd
import xtrack as xt

from xtrack.beam_elements.elements import SplineBoris, Spline4


class SplineBorisSequence:
    '''
    Create a sequence of ``SplineBoris`` elements from FieldFitter output.

    Different field components (Bs, Bx, By, derivatives) may have different
    s-ranges in the FieldFitter output.  This class finds all unique
    s-boundaries and creates one ``SplineBoris`` element per region, using
    every parameter valid for that range.

    Parameters
    ----------
    df_fit_pars : pd.DataFrame
        DataFrame from FieldFitter containing fit parameters.  Must have
        columns: ``field_component``, ``derivative_x``, ``region_name``,
        ``s_start``, ``s_end``, ``idx_start``, ``idx_end``, ``param_index``,
        ``param_name``, ``param_value``.
    multipole_order : int
        Number of multipole orders to use.
    steps_per_point : int, optional
        Multiplier for integration steps per data point.  Default is 1.
    shift_x : float, optional
        Transverse shift in x [m].  Default is 0.0.
    shift_y : float, optional
        Transverse shift in y [m].  Default is 0.0.
    radiation_flag : int, optional
        Radiation flag for the SplineBoris elements.  Default is 0.
    '''

    # Index columns used by FieldFitter and saved CSV files.
    FIELD_FIT_INDEX_COLUMNS = (
        "field_component",
        "derivative_x",
        "region_name",
        "s_start",
        "s_end",
        "idx_start",
        "idx_end",
        "param_index",
    )
    _HERMITE_PARAM_COUNT = len(SplineBoris._HERMITE_SUFFIXES)
    _HERMITE_PARAM_INDEXES = tuple(range(_HERMITE_PARAM_COUNT))

    def __init__(
        self,
        df_fit_pars,
        multipole_order,
        steps_per_point=1,
        shift_x=0.0,
        shift_y=0.0,
        radiation_flag=0,
    ):
        if df_fit_pars is None:
            raise ValueError("df_fit_pars must be a non-empty DataFrame")
        if multipole_order is None or multipole_order <= 0:
            raise ValueError("multipole_order must be a positive integer")
        if not isinstance(steps_per_point, int) or steps_per_point <= 0:
            raise ValueError("steps_per_point must be a positive integer")

        self.multipole_order = int(multipole_order)
        self.steps_per_point = int(steps_per_point)
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.radiation_flag = radiation_flag

        df_reset = self._normalize_df_fit_pars(df_fit_pars)
        if len(df_reset) == 0:
            raise ValueError("df_fit_pars must be a non-empty DataFrame")

        elements, element_names = self._build_elements(df_reset)

        self.elements = tuple(elements)
        self.element_names = tuple(element_names)
        self.length = sum(float(e.length) for e in self.elements)
        self.n_pieces = len(self.elements)

    @staticmethod
    def _split_hermite_to_canonical(
        piece_s_start, piece_s_end, hermite_vals, canon_s_start, canon_s_end
    ):
        """Re-express Hermite boundary data for a canonical sub-interval.

        The original Hermite params describe the field on
        ``[piece_s_start, piece_s_end]``.  This method evaluates the
        implied polynomial (in piece-local coordinates) at the endpoints
        of ``[canon_s_start, canon_s_end]`` to produce new Hermite boundary
        conditions valid exactly for that sub-interval.
        """
        poly = SplineBoris.hermite_to_polynomial(piece_s_start, piece_s_end, hermite_vals)
        dpoly = poly.deriv()
        ipoly = poly.integ()

        # Convert canonical endpoints to piece-local coordinates
        l0 = float(canon_s_start - piece_s_start)
        l1 = float(canon_s_end - piece_s_start)
        ds = l1 - l0

        return Spline4(
            val_start=float(poly(l0)),
            der_start=float(dpoly(l0)),
            val_end=float(poly(l1)),
            der_end=float(dpoly(l1)),
            integral=float((ipoly(l1) - ipoly(l0)) / ds),
        )

    @classmethod
    def _normalize_df_fit_pars(cls, df_fit_pars):
        if not isinstance(df_fit_pars, pd.DataFrame):
            raise TypeError("df_fit_pars must be a pandas DataFrame")

        df_raw = df_fit_pars
        if all(col in df_raw.columns for col in cls.FIELD_FIT_INDEX_COLUMNS):
            df_reset = df_raw.copy()
        else:
            df_reset = df_raw.reset_index()

        required_cols = list(cls.FIELD_FIT_INDEX_COLUMNS) + ["param_value"]
        missing = [c for c in required_cols if c not in df_reset.columns]
        if missing:
            raise ValueError(f"df_fit_pars is missing required columns: {missing}")
        return df_reset

    def _build_elements(self, df_reset):
        """Build SplineBoris elements, one per s-region.

        Returns ``(elements_list, names_list)``.
        """
        multipole_order = self.multipole_order

        # Collect unique (s, idx) boundary pairs, deduplicated and sorted by s.
        seen = set()
        boundary_pairs = []
        for _, row in df_reset.drop_duplicates(subset=["s_start", "s_end"]).iterrows():
            for s_col, idx_col in (("s_start", "idx_start"), ("s_end", "idx_end")):
                pair = (float(row[s_col]), int(row[idx_col]))
                if pair not in seen:
                    seen.add(pair)
                    boundary_pairs.append(pair)
        boundary_pairs.sort(key=lambda p: p[0])

        n_regions = len(boundary_pairs) - 1
        if n_regions <= 0:
            return [], []

        name_width = len(str(n_regions))
        elements = []
        names = []

        zero_spline = Spline4(
            val_start=0.0, der_start=0.0, val_end=0.0, der_end=0.0, integral=0.0
        )

        for i in range(n_regions):
            region_start, idx_start = boundary_pairs[i]
            region_end, idx_end = boundary_pairs[i + 1]

            if region_end <= region_start:
                continue

            mask = (df_reset["s_start"] <= region_start) & (df_reset["s_end"] >= region_end)
            valid_params = df_reset[mask]
            if valid_params.empty:
                continue

            bs_spline = zero_spline
            by_dict = {}
            bx_dict = {}

            groups = valid_params.groupby(
                ["field_component", "derivative_x", "s_start", "s_end"]
            )
            for (fc, deriv, piece_s_start, piece_s_end), grp in groups:
                deriv = int(deriv)
                # Accept both legacy (Bx, By) and canonical (Bskew, Bnorm) names.
                if fc in ("By", "Bx", "Bnorm", "Bskew") and deriv >= multipole_order:
                    continue

                grp_sorted = grp.sort_values("param_index")
                param_index_vals = grp_sorted["param_index"].to_numpy(dtype=int)
                param_vals = grp_sorted["param_value"].to_numpy(dtype=float)
                group_key = (fc, int(deriv), float(piece_s_start), float(piece_s_end))

                if (
                    len(param_vals) != self._HERMITE_PARAM_COUNT
                    or tuple(param_index_vals) != self._HERMITE_PARAM_INDEXES
                ):
                    raise ValueError(
                        "Malformed Hermite group "
                        f"{group_key}: expected exactly {self._HERMITE_PARAM_COUNT} "
                        "rows with param_index values [0, 1, 2, 3, 4]"
                    )
                if not np.isfinite(param_vals).all():
                    raise ValueError(
                        "Malformed Hermite group "
                        f"{group_key}: all Hermite values must be finite"
                    )
                hermite_vals = param_vals
                # Re-express Hermite params for the canonical sub-interval;
                # if piece == canonical interval this is a no-op.
                spline = self._split_hermite_to_canonical(
                    float(piece_s_start),
                    float(piece_s_end),
                    hermite_vals,
                    region_start,
                    region_end,
                )

                if fc == "Bs":
                    bs_spline = spline
                elif fc in ("By", "Bnorm"):
                    by_dict[deriv] = spline
                elif fc in ("Bx", "Bskew"):
                    bx_dict[deriv] = spline

            by_tuple = tuple(by_dict.get(order, zero_spline) for order in range(multipole_order))
            bx_tuple = tuple(bx_dict.get(order, zero_spline) for order in range(multipole_order))

            n_steps = max(1, (idx_end - idx_start) * self.steps_per_point)

            elem = SplineBoris(
                bs=bs_spline,
                by=by_tuple,
                bx=bx_tuple,
                s_start=region_start,
                length=region_end - region_start,
                n_steps=n_steps,
                shift_x=self.shift_x,
                shift_y=self.shift_y,
                radiation_flag=self.radiation_flag,
            )

            idx_name = len(elements)
            elements.append(elem)
            names.append(f"splineboris_{idx_name:0{name_width}d}")

        return elements, names

    def to_line(self, env=None):
        """Return a Line containing all ``SplineBoris`` elements."""
        if env is not None:
            for name, elem in zip(self.element_names, self.elements):
                env.elements[name] = elem
            return xt.Line(env=env, element_names=self.element_names)
        return xt.Line(elements=self.elements)

    @classmethod
    def from_csv(
        cls,
        csv_path,
        multipole_order,
        steps_per_point=1,
        shift_x=0.0,
        shift_y=0.0,
        radiation_flag=0,
    ):
        """Build a ``SplineBorisSequence`` from a CSV file produced by `FieldFitter.save_fit_pars`."""
        df = pd.read_csv(csv_path, index_col=list(cls.FIELD_FIT_INDEX_COLUMNS))
        return cls(
            df_fit_pars=df,
            multipole_order=multipole_order,
            steps_per_point=steps_per_point,
            shift_x=shift_x,
            shift_y=shift_y,
            radiation_flag=radiation_flag,
        )

    def evaluate_field(self, x, y, s):
        """Evaluate the magnetic field by delegating to the appropriate element.

        Parameters
        ----------
        x : float or array-like
            Horizontal position [m].
        y : float or array-like
            Vertical position [m].
        s : float
            Longitudinal position [m].

        Returns
        -------
        Bx, By, Bs : float or array
            Magnetic field components [T].
        """
        for elem in self.elements:
            if elem.s_start <= s <= elem.s_end:
                return elem.evaluate_field(x, y, s)
        s_min = min(float(e.s_start) for e in self.elements)
        s_max = max(float(e.s_end) for e in self.elements)
        raise ValueError(f"s={s} is outside the sequence range [{s_min}, {s_max}]")
