# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

from typing import Literal
from warnings import warn

import numpy as np

from ..general import DEPRECATION_INFO_PREP_1_0
from .. import linear_normal_form as lnf
from ..table import Table
from .twissplot import TwissPlot
from .beam_covariance import _build_sigma_table
from .twiss_init import TwissInit, _W_phys2norm
from .strengths import (
    _add_strengths_to_twiss_res,
    _reverse_strengths,
    NORMAL_STRENGTHS_FROM_ATTR,
    OTHER_FIELDS_FROM_ATTR,
    OTHER_FIELDS_FROM_TABLE,
    SKEW_STRENGTHS_FROM_ATTR,
)
from .radiation import _compute_radiation_integrals

import xtrack as xt  # To avoid circular imports


CYCLICAL_QUANTITIES = ['mux', 'muy', 'dzeta', 's']

DEFAULT_COL_ORDER = [
    'name', 'element_type', 's', 'betx', 'bety', 'alfx', 'alfy', 'dx', 'dy'
    'dpx', 'dpy', 'x', 'y', 'px', 'py', 'delta', 'zeta']


class TwissTable(Table):
    """
    Table returned by :meth:`xtrack.Line.twiss`.

    ``TwissTable`` stores element-by-element optics and closed-orbit data
    produced by Twiss calculations. Typical columns include longitudinal
    position, beta functions, alpha functions, dispersion, phase advances,
    coordinates, momenta, and element strengths when requested.
    """

    # Messages to be shown when accessing deprecated fields
    _DEPRECATED_FIELDS = {
        'slip_factor_dz_ddelta': ('`slip_factor_dz_ddelta` is deprecated, '
                                  'use `slip_factor_dzeta_ddelta` instead.'
                                  + DEPRECATION_INFO_PREP_1_0),
        'T_rev0': ('`T_rev0` is deprecated, use `t_rev0` instead.'
                   + DEPRECATION_INFO_PREP_1_0),
        'T_rev': ('`T_rev` is deprecated, use `t_rev` instead.'
                  + DEPRECATION_INFO_PREP_1_0),
        'kin_xprime': ('`kin_xprime` is deprecated, use `kin_xp` instead.'
                       + DEPRECATION_INFO_PREP_1_0),
        'kin_yprime': ('`kin_yprime` is deprecated, use `kin_yp` instead.'
                       + DEPRECATION_INFO_PREP_1_0),
        'eneloss_turn': ('`eneloss_turn` is deprecated, use `energy_loss` instead.'
                         + DEPRECATION_INFO_PREP_1_0),
        'steps_r_matrix': ('`steps_r_matrix` is deprecated, use `steps_R_matrix` instead.'
                           + DEPRECATION_INFO_PREP_1_0),
        'circumference': ('`circumference` is deprecated, use `line_length` instead.'
                          + DEPRECATION_INFO_PREP_1_0),
        'angle_rad': ('`angle_rad` is deprecated, use `angle` instead.'
                      + DEPRECATION_INFO_PREP_1_0),
    }

    def __init__(self, data, *args, **kwargs):
        """
        Create a Twiss table.

        ``TwissTable`` stores element-by-element optics, closed orbit,
        transfer information, and global quantities produced by Twiss
        calculations.

        Parameters
        ----------
        data : mapping
            Mapping containing Twiss-table columns and scalar attributes.
        *args
            Positional arguments passed to :class:`xtrack.Table`.
        periodic : bool, optional
            Whether the stored Twiss solution is periodic. If not provided,
            the value is taken from ``data["periodic"]`` when available,
            otherwise it defaults to ``False``.
        **kwargs
            Keyword arguments passed to :class:`xtrack.Table`.

        Examples
        --------
        Build a compact Twiss-like table:

        >>> import numpy as np
        >>> import xtrack as xt
        >>> tab = xt.TwissTable({
        ...     "name": np.array(["mqf.1", "d1.1", "mb1.1", "_end_point"],
        ...                      dtype=object),
        ...     "element_type": np.array(["Quadrupole", "Drift", "Bend", ""],
        ...                              dtype=object),
        ...     "s": np.array([0.0, 0.3, 1.3, 4.3]),
        ...     "betx": np.array([1.28, 1.28, 2.27, 1.28]),
        ...     "bety": np.array([4.79, 4.79, 5.21, 4.79]),
        ...     "dx": np.array([2.28, 2.28, 2.24, 2.28]),
        ... })
        >>> tab
        TwissTable: 4 rows, 6 cols
        name       element_type             s          betx          bety            dx
        mqf.1      Quadrupole               0          1.28          4.79          2.28
        d1.1       Drift                  0.3          1.28          4.79          2.28
        mb1.1      Bend                   1.3          2.27          5.21          2.24
        _end_point                        4.3          1.28          4.79          2.28

        Select optics columns, including expressions:

        >>> tab.cols["betx bety dx/sqrt(betx)"]
        TwissTable: 4 rows, 4 cols
        name                betx          bety dx/sqrt(betx)
        mqf.1               1.28          4.79       2.01525
        d1.1                1.28          4.79       2.01525
        mb1.1               2.27          5.21       1.48674
        _end_point          1.28          4.79       2.01525

        Select elements by type:

        >>> tab.rows.match(element_type="Quadrupole|Bend")
        TwissTable: 2 rows, 6 cols
        name  element_type             s          betx          bety            dx
        mqf.1 Quadrupole               0          1.28          4.79          2.28
        mb1.1 Bend                   1.3          2.27          5.21          2.24
        """
        kwargs['sep_count'] = kwargs.get('sep_count', '::::')
        periodic = kwargs.pop('periodic', data.get('periodic', False))
        super().__init__(data, *args, **kwargs)
        self['periodic'] = periodic

    _error_on_row_not_found = True

    def _select_rows(self, rows):
        out = super()._select_rows(rows)
        out._data.pop('periodic', None)
        return out

    def to_pandas(self, index=None, columns=None):
        """
        Convert the Twiss table to a pandas DataFrame.

        Parameters
        ----------
        index : str, optional
            Column to use as the DataFrame index.
        columns : sequence of str, optional
            Columns to include in the DataFrame. If not provided, all table
            columns are included.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the selected Twiss table columns.
        """
        if columns is None:
            columns = self._col_names

        data = self._data.copy()
        if 'W_matrix' in data.keys():
            data['W_matrix'] = [
                self.W_matrix[ii] for ii in range(len(self.W_matrix))]

        import pandas as pd
        df = pd.DataFrame(data, columns=self._col_names)
        if index is not None:
            df.set_index(index, inplace=True)
        return df

    def _extra_metadata(self):
        extra = super()._extra_metadata()
        extra = dict(extra) if extra else {}
        extra['__class__'] = 'TwissTable'
        extra['xtrack_version'] = xt.__version__
        return extra

    @classmethod
    def _strip_extra_metadata(cls, payload):
        payload.pop('__class__', None)
        payload.pop('xtrack_version', None)
        super()._strip_extra_metadata(payload)

    def to_hdf5(self, file, *, include=None, exclude=None,
                missing='error', include_meta=True, group='twiss_table'):
        """
        Write the Twiss table to an HDF5 file.

        Parameters
        ----------
        file : str or pathlib.Path or h5py.File
            Output HDF5 file path or open HDF5 file object.
        include : sequence of str, optional
            Columns or metadata fields to include. If not provided, all supported
            fields are included unless excluded.
        exclude : sequence of str, optional
            Columns or metadata fields to exclude.
        missing : {"error", "ignore"}, optional
            Policy for names in ``include`` or ``exclude`` that are not present in
            the table.
        include_meta : bool, optional
            Whether to include table metadata.
        group : str, optional
            HDF5 group in which the table is stored. Defaults to
            ``"twiss_table"``.
        """
        super().to_hdf5(
            file,
            include=include,
            exclude=exclude,
            missing=missing,
            include_meta=include_meta,
            group=group,
        )

    @classmethod
    def from_hdf5(cls, file, *, group='twiss_table'):
        """
        Load a Twiss table from an HDF5 file.

        Parameters
        ----------
        file : str or pathlib.Path or h5py.File
            Input HDF5 file path or open HDF5 file object.
        group : str, optional
            HDF5 group from which the table is loaded. Defaults to
            ``"twiss_table"``.

        Returns
        -------
        xtrack.TwissTable
            Twiss table loaded from the HDF5 file.
        """
        return super().from_hdf5(
            file,
            group=group,
        )

    def to_tfs(self, file, *, include=None, exclude=None,
               missing='error', include_meta=True,
               default_column_width=None, float_precision=8,
               numeric_column_width=16, column_formats=None,
               column_widths=None):
        """
        Write the Twiss table to a TFS file.

        Parameters
        ----------
        file : str or pathlib.Path or file-like object
            Output TFS file path or writable file-like object.
        include : sequence of str, optional
            Columns or metadata fields to include. If not provided, all supported
            fields are included unless excluded.
        exclude : sequence of str, optional
            Columns or metadata fields to exclude.
        missing : {"error", "ignore"}, optional
            Policy for names in ``include`` or ``exclude`` that are not present in
            the table.
        include_meta : bool, optional
            Whether to include table metadata as TFS headers.
        default_column_width : int, optional
            Default width used for TFS columns.
        float_precision : int, optional
            Number of significant digits used for floating-point values.
        numeric_column_width : int, optional
            Width used for numeric TFS columns.
        column_formats : dict, optional
            Per-column TFS format strings.
        column_widths : dict, optional
            Per-column TFS column widths.
        """

        if exclude is None:
            exclude = []

        if 'completed_init' in self.keys():
            exclude.append('completed_init')

        super().to_tfs(
            file,
            include=include,
            exclude=exclude,
            missing=missing,
            include_meta=include_meta,
            default_column_width=default_column_width,
            float_precision=float_precision,
            numeric_column_width=numeric_column_width,
            column_formats=column_formats,
            column_widths=column_widths,
        )

    @classmethod
    def from_tfs(cls, file):
        """
        Load a Twiss table from a TFS file.

        Parameters
        ----------
        file : str or pathlib.Path or file-like object
            Input TFS file path or readable file-like object.

        Returns
        -------
        xtrack.TwissTable
            Twiss table loaded from the TFS file.
        """
        return super().from_tfs(file)

    def get_twiss_init(self, at_element):
        """
        Build Twiss initial conditions from this table at an element.

        The returned object contains the closed-orbit particle, W matrix, phase
        advances, and available chromatic quantities extracted from the selected
        row of the table. It can be passed as the ``init`` argument to
        :meth:`xtrack.Line.twiss` to start a Twiss calculation from the same
        optics conditions.

        Parameters
        ----------
        at_element : str or int
            Element name or row index at which the initial conditions are
            extracted. The table must contain values at element entry.

        Returns
        -------
        xtrack.TwissInit
            Initial conditions for a Twiss calculation at the selected element.
        """

        assert self.values_at == 'entry', 'Not yet implemented for exit'

        if isinstance(at_element, str):
            at_element = np.where(self.name == at_element)[0][0]
        part = self.particle_on_co.copy()
        part.x[:] = self.x[at_element]
        part.px[:] = self.px[at_element]
        part.y[:] = self.y[at_element]
        part.py[:] = self.py[at_element]
        part.zeta[:] = self.zeta[at_element]
        part.ptau[:] = self.ptau[at_element]
        part.s[:] = self.s[at_element]
        part.ax[:] = part.px[:] - self.kin_px[at_element]
        part.ay[:] = part.py[:] - self.kin_py[at_element]
        part.at_element[:] = -1

        W = self.W_matrix[at_element]

        if 'ax_chrom' in self.keys():
            ax_chrom = self.ax_chrom[at_element]
            bx_chrom = self.bx_chrom[at_element]
            ay_chrom = self.ay_chrom[at_element]
            by_chrom = self.by_chrom[at_element]
            ddx = self.ddx[at_element]
            ddpx = self.ddpx[at_element]
            ddy = self.ddy[at_element]
            ddpy = self.ddpy[at_element]
        else:
            ax_chrom = None
            bx_chrom = None
            ay_chrom = None
            by_chrom = None
            ddx = None
            ddpx = None
            ddy = None
            ddpy = None

        if 'mux' in self.keys():
            mux = self.mux[at_element]
            muy = self.muy[at_element]
            muzeta = self.muzeta[at_element]
        else:
            mux = 0
            muy = 0
            muzeta = 0

        if 'dzeta' in self.keys():
            dzeta = self.dzeta[at_element]
        else:
            dzeta = 0

        if hasattr(self, 'spin_x'):
            part.spin_x[:] = self.spin_x[at_element]
            part.spin_y[:] = self.spin_y[at_element]
            part.spin_z[:] = self.spin_z[at_element]

        return TwissInit(particle_on_co=part, W_matrix=W,
                        element_name=str(self.name[at_element]),
                        mux=mux, muy=muy, muzeta=muzeta, dzeta=dzeta,
                        ax_chrom=ax_chrom, bx_chrom=bx_chrom,
                        ay_chrom=ay_chrom, by_chrom=by_chrom,
                        ddx=ddx, ddpx=ddpx, ddy=ddy, ddpy=ddpy,
                        reference_frame=self.reference_frame)

    def get_betatron_sigmas(self, nemitt_x, nemitt_y):
        """
        Compute transverse beam covariance from normalized emittances.

        .. warning::

            This method is deprecated and will be removed in a future version.
            Use :meth:`get_beam_covariance` instead, with ``nemitt_x`` and
            ``nemitt_y``.

        Parameters
        ----------
        nemitt_x : float
            Horizontal normalized emittance.
        nemitt_y : float
            Vertical normalized emittance.

        Returns
        -------
        xtrack.Table
            Table containing the beam covariance matrix elements along the line.
        """
        warn(
            '`TwissTable.get_betatron_sigmas()` is deprecated and will be '
            'removed in future versions. Use '
            '`TwissTable.get_beam_covariance()` instead.',
            FutureWarning,
            stacklevel=2,
        )
        return self.get_beam_covariance(
            nemitt_x=nemitt_x, nemitt_y=nemitt_y)

    def get_beam_covariance(self,
            nemitt_x=None, nemitt_y=None, nemitt_zeta=None,
            gemitt_x=None, gemitt_y=None, gemitt_zeta=None):
        """
        Compute the beam covariance matrix along the line.

        The covariance matrix is built from the W matrices stored in the Twiss
        table and the provided transverse and longitudinal emittances. Normalized
        emittances are converted to geometric emittances using the reference
        particle beta and gamma.

        Parameters
        ----------
        nemitt_x : float, optional
            Horizontal normalized emittance.
        nemitt_y : float, optional
            Vertical normalized emittance.
        nemitt_zeta : float, optional
            Longitudinal normalized emittance.
        gemitt_x : float, optional
            Horizontal geometric emittance.
        gemitt_y : float, optional
            Vertical geometric emittance.
        gemitt_zeta : float, optional
            Longitudinal geometric emittance.

        Returns
        -------
        xtrack.Table
            Table containing the beam covariance matrix elements along the line.
        """

        # See MAD8 physics manual (Eq. 8.59)

        beta0 = self.particle_on_co.beta0
        gamma0 = self.particle_on_co.gamma0

        if nemitt_x is not None:
            assert gemitt_x is None, 'Cannot provide both nemitt_x and gemitt_x'
            gemitt_x = nemitt_x / (beta0 * gamma0)

        if nemitt_y is not None:
            assert gemitt_y is None, 'Cannot provide both nemitt_y and gemitt_y'
            gemitt_y = nemitt_y / (beta0 * gamma0)

        if nemitt_zeta is not None:
            assert gemitt_zeta is None, 'Cannot provide both nemitt_zeta and gemitt_zeta'
            gemitt_zeta = nemitt_zeta / (beta0 * gamma0)

        gemitt_x = gemitt_x or 0
        gemitt_y = gemitt_y or 0
        gemitt_zeta = gemitt_zeta or 0

        Ws = self.W_matrix.copy()

        if self.method == '4d':
            Ws[:, 4:, 4:] = 0

        v1 = Ws[:,:,0] + 1j * Ws[:,:,1]
        v2 = Ws[:,:,2] + 1j * Ws[:,:,3]
        v3 = Ws[:,:,4] + 1j * Ws[:,:,5]

        Sigma1 = np.zeros(shape=(len(self.s), 6, 6), dtype=np.float64)
        Sigma2 = np.zeros(shape=(len(self.s), 6, 6), dtype=np.float64)
        Sigma3 = np.zeros(shape=(len(self.s), 6, 6), dtype=np.float64)

        for ii in range(6):
            for jj in range(6):
                Sigma1[:, ii, jj] = np.real(v1[:,ii] * v1[:,jj].conj())
                Sigma2[:, ii, jj] = np.real(v2[:,ii] * v2[:,jj].conj())
                Sigma3[:, ii, jj] = np.real(v3[:,ii] * v3[:,jj].conj())

        Sigma = gemitt_x * Sigma1 + gemitt_y * Sigma2 + gemitt_zeta * Sigma3
        res = _build_sigma_table(Sigma=Sigma, s=self.s, name=self.name)

        return res

    def get_ibs_growth_rates(
        self,
        formalism: str,
        total_beam_intensity: int = None,
        gemitt_x: float = None,
        nemitt_x: float = None,
        gemitt_y: float = None,
        nemitt_y: float = None,
        sigma_delta: float = None,
        bunch_length: float = None,
        bunched: bool = True,
        **kwargs,
    ):
        """
        Computes IntraBeam Scattering (amplitude) growth rates.

        Parameters
        ----------
        formalism : str
            Which formalism to use for the computation. Can be ``Nagaitsev``
            or ``Bjorken-Mtingwa`` (also accepts ``B&M``), case-insensitively.
        total_beam_intensity : int, optional
            The beam intensity. Required if ``particles`` is not provided.
        gemitt_x : float, optional
            Horizontal geometric emittance in [m]. If ``particles`` is not
            provided, either this parameter or ``nemitt_x`` is required.
        nemitt_x : float, optional
            Horizontal normalized emittance in [m]. If ``particles`` is not
            provided, either this parameter or ``gemitt_x`` is required.
        gemitt_y : float, optional
            Vertical geometric emittance in [m]. If ``particles`` is not
            provided, either this parameter or ``nemitt_y`` is required.
        nemitt_y : float, optional
            Vertical normalized emittance in [m]. If ``particles`` is not
            provided, either this parameter or ``gemitt_y`` is required.
        sigma_delta : float, optional
            The momentum spread. Required if ``particles`` is not provided.
        bunch_length : float, optional
            The bunch length in [m]. Required if ``particles`` is not provided.
        bunched : bool, optional
            Whether the beam is bunched or not (coasting). Defaults to ``True``.
            Required if ``particles`` is not provided.
        **kwargs : dict
            Keyword arguments are passed to the growth rates computation method of
            the chosen IBS formalism implementation. See the IBS details from the
            ``xfields`` package directly.

        Returns
        -------
        IBSGrowthRates
            An ``IBSGrowthRates`` object with the computed growth rates.
        """
        try:
            from xfields.ibs import get_intrabeam_scattering_growth_rates
        except ImportError:
            raise ImportError("Please install xfields to use this feature.")
        return get_intrabeam_scattering_growth_rates(
            self, formalism, total_beam_intensity,
            gemitt_x, nemitt_x, gemitt_y, nemitt_y,
            sigma_delta, bunch_length, bunched,
            **kwargs,
        )

    def get_ibs_and_synrad_emittance_evolution(
        self,
        formalism: Literal["Nagaitsev", "Bjorken-Mtingwa", "B&M"],
        total_beam_intensity: int,
        gemitt_x: float | None = None,
        nemitt_x: float | None = None,
        gemitt_y: float | None = None,
        nemitt_y: float | None = None,
        gemitt_zeta: float | None = None,
        nemitt_zeta: float | None = None,
        overwrite_sigma_zeta: float | None = None,
        overwrite_sigma_delta: float | None = None,
        emittance_coupling_factor: float = 0,
        emittance_constraint: Literal["coupling", "excitation"] | None = "coupling",
        rtol: float = 1e-6,
        tstep: float | None = None,
        max_steps: float | None = None,
        verbose: bool = True,
        **kwargs,
    ) -> Table:
        """
        Compute the evolution of emittances due to Synchrotron Radiation
        and Intra-Beam Scattering until convergence to equilibrium values.
        The equilibrium state is determined by an iterative process which
        consists in computing the IBS growth rates and the emittance time
        derivatives, then computing the emittances at the next time step,
        potentially including the effect of transverse constraints, and
        checking for convergence. The convergence criteria can be chosen
        by the user.

        Transverse emittances can be constrained to follow two scenarios:
            - An emittance exchange originating from betatron coupling.
            - A vertical emittance originating from an excitation.

        The impact from the longitudinal impedance (e.g. bunch lengthening
        or microwave instability) can be accounted for by specifying the RMS
        bunch length and momentum spread.

        Notes
        -----
            It is required that radiation has been configured in the line,
            and that this ``TwissTable`` holds information on the equilibrium
            state from Synchrotron Radiation. This means calling first
            ``line.configure_radiation(model="mean")`` and then the ``.twiss()``
            method with ``radiation_analysis=True``.

        Warning
        -------
            If the user does not provide a starting emittance, the program
            defaults to using the SR equilibrium value from this ``TwissTable``,
            which is a reasonable defaults for light sources. If a constraint
            is provided via ``emittance_constraint``  the starting emittances are
            re-computed to respect that constraint (this is logged to the user).

            If the user does provide starting emittances **and** a constraint, it
            is up to the user to make sure these provided values are consistent
            with the provided constraint!

        Parameters
        ----------
        formalism : str
            Which formalism to use for the computation of the IBS growth rates.
            Can be ``Nagaitsev`` or ``Bjorken-Mtingwa`` (also accepts ``B&M``),
            case-insensitively.
        total_beam_intensity : int
            The bunch intensity, in [particles per bunch].
        gemitt_x : float, optional
            Starting horizontal geometric emittance, in [m]. If neither this nor
            the normalized one is provided, the SR equilibrium value from this
            ``TwissTable`` is used.
        nemitt_x : float, optional
            Starting horizontal normalized emittance, in [m]. If neither this nor
            the geometric one is provided, the SR equilibrium value from this
            ``TwissTable`` is used.
        gemitt_y : float, optional
            Starting vertical geometric emittance, in [m]. If neither this nor
            the normalized one is provided, the SR equilibrium value from this
            ``TwissTable`` is used.
        nemitt_y : float, optional
            Starting vertical normalized emittance, in [m]. If neither this nor
            the geometric one is provided, the SR equilibrium value from this
            ``TwissTable`` is used.
        gemitt_zeta : float, optional
            Starting longitudinal geometric emittance, in [m]. If neither this
            nor the normalized one is provided, the SR equilibrium value from
            this ``TwissTable`` is used.
        nemitt_zeta : float, optional
            Starting longitudinal normalized emittance, in [m]. If neither this
            nor the geometric one is provided, the SR equilibrium value from this
            ``TwissTable`` is used.
        emittance_coupling_factor : float, optional
            The ratio of perturbed transverse emittances due to betatron coupling.
            If a value is provided, it is taken into account for the evolution of
            emittances and induces an emittance sharing between the two planes.
            See the next parameter for possible scenarios and how this value is
            used. Defaults to 0.
        emittance_constraint : str, optional
            If an accepted value is provided, enforces constraints on the transverse
            emittances. Can be either "coupling" or "excitation", case-insensitively.
            Defaults to "coupling".
            - If ``coupling``, vertical emittance is the result of linear coupling. In
                this case both the vertical and horizontal emittances are altered and
                determined based on the value of ``emittance_coupling_factor`` and the
                damping partition numbers. If the horizontal and vertical partition
                numbers are equal then the total transverse emittance is preserved.
            - If ``excitation``, vertical emittance is the result of an excitation
                (e.g. from a feedback system) and is determined from the horizontal
                emittance based on the value of ``emittance_coupling_factor``. In this
                case the total transverse emittance is NOT preserved.
            Providing ``None`` allows one to study a scenario without constraint. Note
            that as ``emittance_coupling_factor`` defaults to 0, the constraint has no
            effect unless a non-zero factor is provided.
        overwrite_sigma_zeta : float, optional
            The RMS bunch length, in [m]. If provided, overwrites the one computed from
            the longitudinal emittance and forces a recompute of the longitudinal
            emittance. Defaults to ``None``.
        overwrite_sigma_delta : float, optional
            The RMS momentum spread of the bunch. If provided, overwrites the one
            computed from the longitudinal emittance and forces a recompute of the
            longitudinal emittance. Defaults to ``None``.
        rtol : float, optional
            Relative tolerance to determine when convergence is reached: if the relative
            difference between the computed emittances and those at the previous step is
            below ``rtol``, then convergence is considered achieved. Defaults to 1e-6.
        tstep : float, optional
            Time step to use for each iteration, in [s]. If not provided, an
            adaptive time step is computed based on the IBS growth rates and
            the damping constants. Defaults to ``None``.
        max_steps : float, optional
            The maximum number of iterations to perform before stopping the iterative
            process. If not provided, the process continues until it reaches convergence
            (according to the provided ``rtol``). Defaults to ``None``.
        verbose : bool, optional
            Whether to print out information on the current iteration step and estimated
            convergence progress. Defaults to ``True``.
        **kwargs : dict
            Keyword arguments are passed to the growth rates computation method of
            the chosen IBS formalism implementation. See the formalism classes in
            the ``xfields.ibs._analytical`` for more details.

        Returns
        -------
        xtrack.Table
            The convergence calculations results. The table contains the following
            columns, as time-step by time-step quantities:
                - time: time values at which quantities are computed, in [s].
                - gemitt_x: horizontal geometric emittances, in [m].
                - nemitt_x: horizontal normalized emittances, in [m].
                - gemitt_y: vertical geometric emittances, in [m].
                - nemitt_y: vertical normalized emittances, in [m].
                - gemitt_zeta: longitudinal geometric emittances, in [m].
                - nemitt_zeta: longitudinal normalized emittances, in [m].
                - sigma_zeta: bunch lengths, in [m].
                - sigma_delta: momentum spreads, in [-].
                - Kx: horizontal IBS amplitude growth rates, in [s^-1].
                - Ky: vertical IBS amplitude growth rates, in [s^-1].
                - Kz: longitudinal IBS amplitude growth rates, in [s^-1].
            The table also contains the following global quantities:
                - damping_constants_s: radiation damping constants used, in [s].
                - partition_numbers: damping partition numbers used.
                - eq_gemitt_x: horizontal equilibrium geometric emittance from synchrotron radiation used, in [m].
                - eq_gemitt_y: vertical equilibrium geometric emittance from synchrotron radiation used, in [m].
                - eq_gemitt_zeta: longitudinal equilibrium geometric emittance from synchrotron radiation used, in [m].
                - eq_sr_ibs_gemitt_x: final horizontal equilibrium geometric emittance converged to, in [m].
                - eq_sr_ibs_nemitt_x: final horizontal equilibrium normalized emittance converged to, in [m].
                - eq_sr_ibs_gemitt_y: final vertical equilibrium geometric emittance converged to, in [m].
                - eq_sr_ibs_gemitt_y: final vertical equilibrium normalized emittance converged to, in [m].
                - eq_sr_ibs_gemitt_zeta: final longitudinal equilibrium geometric emittance converged to, in [m].
                - eq_sr_ibs_gemitt_zeta: final longitudinal equilibrium normalized emittance converged to, in [m].
        """
        try:
            from xfields.ibs import get_ibs_and_synrad_emittance_evolution
        except ImportError:
            raise ImportError("Please install xfields to use this feature.")
        return get_ibs_and_synrad_emittance_evolution(
            self, formalism=formalism, total_beam_intensity=total_beam_intensity,
            gemitt_x=gemitt_x, nemitt_x=nemitt_x, gemitt_y=gemitt_y, nemitt_y=nemitt_y,
            gemitt_zeta=gemitt_zeta, nemitt_zeta=nemitt_zeta,
            overwrite_sigma_zeta=overwrite_sigma_zeta,
            overwrite_sigma_delta=overwrite_sigma_delta,
            emittance_coupling_factor=emittance_coupling_factor,
            emittance_constraint=emittance_constraint,
            rtol=rtol, tstep=tstep, max_steps=max_steps, verbose=verbose, **kwargs,
        )

    def get_R_matrix(self, start, end):
        """
        Compute the transfer matrix between two table locations.

        The matrix is reconstructed from the W matrices and phase advances stored
        in the Twiss table. Both ``start`` and ``end`` identify rows in the table
        and are used as the boundary locations of the transfer. For tables with
        ``values_at == "entry"`` (default), this is the transfer from the entry of
        ``start`` to the entry of ``end``.

        Parameters
        ----------
        start : str or int
            Element name or row index at which the transfer starts.
        end : str or int
            Element name or row index at which the transfer ends. The end row must
            be after the start row in the table.

        Returns
        -------
        numpy.ndarray
            Six-by-six transfer matrix from ``start`` to ``end``.
        """

        assert self.values_at == 'entry', 'Not yet implemented for exit'

        if isinstance(start, str):
            start = np.where(self.name == start)[0][0]
        if isinstance(end, str):
            end = np.where(self.name == end)[0][0]

        if start > end:
            raise ValueError('start must be smaller than ele_end')

        W_start = self.W_matrix[start]
        W_end = self.W_matrix[end]

        phix_start = self.phix[start]
        phix_end = self.phix[end]
        phiy_start = self.phiy[start]
        phiy_end = self.phiy[end]
        phizeta_start = self.phizeta[start]
        phizeta_end = self.phizeta[end]

        phi_x = phix_end - phix_start
        phi_y = phiy_end - phiy_start
        phi_zeta = phizeta_end - phizeta_start

        Rot = np.zeros(shape=(6, 6), dtype=np.float64)

        Rot[0:2,0:2] = lnf.Rot2D(phi_x)
        Rot[2:4,2:4] = lnf.Rot2D(phi_y)
        Rot[4:6,4:6] = lnf.Rot2D(phi_zeta)

        R_matrix = W_end @ Rot @ np.linalg.inv(W_start)

        return R_matrix

    def get_R_matrix_table(self):
        """
        Compute transfer matrices from the first table row to all rows.

        For each row, the transfer matrix is reconstructed from the W matrix at
        that row, the W matrix at the first row, and the phase advances relative
        to the first row.

        Returns
        -------
        xtrack.Table
            Table with one row per Twiss-table row. It contains the element names,
            longitudinal positions, the full ``R_matrix`` array for each row, and
            scalar columns ``r11`` through ``r66`` with the individual matrix
            elements.
        """

        Rot = np.zeros(shape=(len(self.s), 6, 6), dtype=np.float64)

        cos_phix = np.cos(self.phix - self.phix[0])
        sin_phix = np.sin(self.phix - self.phix[0])
        cos_phiy = np.cos(self.phiy - self.phiy[0])
        sin_phiy = np.sin(self.phiy - self.phiy[0])
        cos_phizeta = np.cos(self.phizeta - self.phizeta[0])
        sin_phizeta = np.sin(self.phizeta - self.phizeta[0])

        Rot[:, 0, 0] = cos_phix
        Rot[:, 0, 1] = sin_phix
        Rot[:, 1, 0] = -sin_phix
        Rot[:, 1, 1] = cos_phix
        Rot[:, 2, 2] = cos_phiy
        Rot[:, 2, 3] = sin_phiy
        Rot[:, 3, 2] = -sin_phiy
        Rot[:, 3, 3] = cos_phiy
        Rot[:, 4, 4] = cos_phizeta
        Rot[:, 4, 5] = sin_phizeta
        Rot[:, 5, 4] = -sin_phizeta
        Rot[:, 5, 5] = cos_phizeta

        # Compute W @ Rot @ W_inv slice by slice
        WW = self.W_matrix
        R_matrix_ebe = np.einsum('ijk,ikl->ijl', WW, Rot) @ np.linalg.inv(WW[0, :, :])

        out_dct = {'s': self.s, 'name': self.name, 'R_matrix': R_matrix_ebe}
        for ii in range(6):
            for jj in range(6):
                out_dct[f'r{ii+1}{jj+1}'] = R_matrix_ebe[:, ii, jj]

        return Table(out_dct)

    def get_normalized_coordinates(self, particles, nemitt_x=None, nemitt_y=None,
                                   nemitt_zeta=None, _force_at_element=None):
        """
        Convert particle coordinates to normalized coordinates.

        Particle physical coordinates are transformed using the closed orbit and
        W matrix stored in this Twiss table at each particle's ``at_element``.
        If normalized emittances are provided, the normalized coordinates are
        scaled by the square root of the corresponding emittance.

        Parameters
        ----------
        particles : xtrack.Particles
            Particles whose coordinates are converted.
        nemitt_x : float, optional
            Horizontal normalized emittance used to scale ``x_norm`` and
            ``px_norm``.
        nemitt_y : float, optional
            Vertical normalized emittance used to scale ``y_norm`` and
            ``py_norm``.
        nemitt_zeta : float, optional
            Longitudinal normalized emittance used to scale ``zeta_norm`` and
            ``pzeta_norm``.

        Returns
        -------
        xtrack.Table
            Table indexed by ``particle_id`` with columns ``particle_id``,
            ``at_element``, ``x_norm``, ``px_norm``, ``y_norm``, ``py_norm``,
            ``zeta_norm``, and ``pzeta_norm``.
        """

        ctx2np = particles._context.nparray_from_context_array
        at_element_particles = ctx2np(particles.at_element)

        part_id = ctx2np(particles.particle_id).copy()
        at_element = part_id.copy() * 0 + xt.particles.LAST_INVALID_STATE
        x_norm = ctx2np(particles.x).copy() * 0 + xt.particles.LAST_INVALID_STATE
        px_norm = x_norm.copy()
        y_norm = x_norm.copy()
        py_norm = x_norm.copy()
        zeta_norm = x_norm.copy()
        pzeta_norm = x_norm.copy()

        at_element_no_rep = list(set(
            at_element_particles[part_id > xt.particles.LAST_INVALID_STATE]))

        for at_ele in at_element_no_rep:

            if _force_at_element is not None:
                at_ele = _force_at_element

            W = self.W_matrix[at_ele]

            mask_at_ele = at_element_particles == at_ele

            if _force_at_element is not None:
                mask_at_ele = ctx2np(particles.state) > xt.particles.LAST_INVALID_STATE


            XX_norm  = _W_phys2norm(x = ctx2np(particles.x)[mask_at_ele],
                                    px = ctx2np(particles.px)[mask_at_ele],
                                    y = ctx2np(particles.y)[mask_at_ele],
                                    py = ctx2np(particles.py)[mask_at_ele],
                                    zeta = ctx2np(particles.zeta)[mask_at_ele],
                                    pzeta = ctx2np(particles.ptau)[mask_at_ele]/ctx2np(particles.beta0)[mask_at_ele],
                                    W_matrix = W,
                                    co_dict = {'x': self.x[at_ele], 'px': self.px[at_ele],
                                               'y': self.y[at_ele], 'py': self.py[at_ele],
                                               'zeta': self.zeta[at_ele], 'ptau': self.ptau[at_ele],
                                               'beta0': self.particle_on_co._xobject.beta0[0],
                                               'gamma0': self.particle_on_co._xobject.gamma0[0]},
                                    nemitt_x = nemitt_x,
                                    nemitt_y = nemitt_y,
                                    nemitt_zeta = nemitt_zeta)

            x_norm[mask_at_ele] = XX_norm[0, :]
            px_norm[mask_at_ele] = XX_norm[1, :]
            y_norm[mask_at_ele] = XX_norm[2, :]
            py_norm[mask_at_ele] = XX_norm[3, :]
            zeta_norm[mask_at_ele] = XX_norm[4, :]
            pzeta_norm[mask_at_ele] = XX_norm[5, :]
            at_element[mask_at_ele] = at_ele

        return Table({'particle_id': part_id, 'at_element': at_element,
                      'x_norm': x_norm, 'px_norm': px_norm, 'y_norm': y_norm,
                      'py_norm': py_norm, 'zeta_norm': zeta_norm,
                      'pzeta_norm': pzeta_norm}, index='particle_id')

    def reverse(self):
        """
        Build a Twiss table for the reverse local reference frame.

        The returned table has the element order reversed and optics quantities
        transformed to the reverse local reference frame. The transverse and
        longitudinal coordinates are transformed as ``x -> -x``, ``y -> y``,
        and ``zeta -> -zeta``. The momenta and longitudinal position are
        transformed as ``px -> px``, ``py -> -py``, and
        ``s -> line_length - s``. The phase advances are transformed as
        ``mux -> mux[0] - mux`` and ``muy -> muy[0] - muy``. The reference frame
        is switched between ``"proper"`` and ``"reverse"``.

        Returns
        -------
        xtrack.TwissTable
            Twiss table corresponding to the reverse local reference frame.
        """

        assert self.values_at == 'entry', 'Not yet implemented for exit'
        assert self.name[-1] == '_end_point' # Needed for the present implementation

        new_data = {}
        for kk, vv in self._data.items():
            if hasattr(vv, 'copy'):
                new_data[kk] = vv.copy()
            else:
                new_data[kk] = vv

        if self.only_markers:
            itake = slice(None, -1, None)
        else:
            # To keep association name <-> quantities at elemement entry
            itake = slice(1, None, None)

        for kk in self._col_names:
            # Accessing with _data to avoid triggering deprecation warnings
            if (kk == 'name' or kk == 'env_name'
                    or kk in NORMAL_STRENGTHS_FROM_ATTR
                    or kk in SKEW_STRENGTHS_FROM_ATTR
                    or kk in OTHER_FIELDS_FROM_ATTR
                    or kk in OTHER_FIELDS_FROM_TABLE
            ):
                new_data[kk][:-1] = new_data[kk][:-1][::-1]
                new_data[kk][-1] = self._data[kk][-1]
            elif kk == 'W_matrix':
                new_data[kk][:-1, :, :] = new_data[kk][itake, :, :][::-1, :, :]
                new_data[kk][-1, :, :] = self._data[kk][0, :, :]
            else:
                if kk in ['kin_xprime', 'kin_yprime']:
                    # deprecated fields, to be removed in the future
                    continue # handled separately below for backward compatibility
                new_data[kk][:-1] = new_data[kk][itake][::-1]
                new_data[kk][-1] = self._data[kk][0]

        out = self.__class__(data=new_data, col_names=self._col_names)

        line_length = (
            out.line_length if hasattr(out, 'line_length') else np.max(out.s))

        out.s = line_length - out.s

        out.x = -out.x
        out.px = out.px # Dx/Ds
        out.y = out.y
        out.py = -out.py # Dy/Ds
        out.zeta = -out.zeta
        out.delta = out.delta
        out.ptau = out.ptau

        if 'kin_px' in out:
            out.kin_px = out.kin_px
            out.kin_py = -out.kin_py
            out.kin_xprime = out.kin_xp # deprecated
            out.kin_yprime = -out.kin_yp # deprecated
            out.kin_xp = out.kin_xp
            out.kin_yp = -out.kin_yp

        if 'betx' in out:
            # if optics calculation is not skipped
            out.betx = out.betx
            out.bety = out.bety
            out.alfx = -out.alfx # Dpx/Dx
            out.alfy = -out.alfy # Dpy/Dy
            out.gamx = out.gamx
            out.gamy = out.gamy

            out.dx = -out.dx
            out.dpx = out.dpx
            out.dy = out.dy
            out.dpy = -out.dpy
            if 'dzeta' in out:
                out.dzeta = -out.dzeta

            if 'dx_zeta' in out._col_names:
                out.dx_zeta = out.dx_zeta
                out.dpx_zeta = -out.dpx_zeta
                out.dy_zeta = -out.dy_zeta
                out.dpy_zeta = out.dpy_zeta

            if 'alfx2' in out._col_names:
                out.alfx1 = -out.alfx1
                out.alfx2 = -out.alfx2
                out.alfy1 = -out.alfy1
                out.alfy2 = -out.alfy2

            if 'alfx_edw_teng' in out._col_names:
                out.alfx_edw_teng = -out.alfx_edw_teng
                out.alfy_edw_teng = -out.alfy_edw_teng

            if 'f1001' in out._col_names:
                out.f1001 = np.conj(out.f1001)
                out.f1010 = np.conj(out.f1010)
                out.f0110 = np.conj(out.f0110)
                out.f0101 = np.conj(out.f0101)

            out.W_matrix[:, 0, :] = -out.W_matrix[:, 0, :]
            out.W_matrix[:, 1, :] = out.W_matrix[:, 1, :]
            out.W_matrix[:, 2, :] = out.W_matrix[:, 2, :]
            out.W_matrix[:, 3, :] = -out.W_matrix[:, 3, :]
            out.W_matrix[:, 4, :] = -out.W_matrix[:, 4, :]
            out.W_matrix[:, 5, :] = out.W_matrix[:, 5, :]

            out.mux = out.mux[0] - out.mux
            out.muy = out.muy[0] - out.muy
            out.muzeta = out.muzeta[0] - out.muzeta
            out.phix = -out.phix
            out.phiy = -out.phiy

            if 'dzeta' in out:
                out.dzeta = out.dzeta[0] - out.dzeta

        if 'ax_chrom' in out._col_names:
            out.ax_chrom = -out.ax_chrom
            out.ay_chrom = -out.ay_chrom
            out.ddx = -out.ddx
            out.ddpy = -out.ddpy

        if hasattr(out, 'R_matrix'): out.R_matrix = None # To be implemented
        if hasattr(out, 'particle_on_co'):
            out.particle_on_co = self.particle_on_co.copy()
            out.particle_on_co.x = -out.particle_on_co.x
            out.particle_on_co.py = -out.particle_on_co.py
            out.particle_on_co.zeta = -out.particle_on_co.zeta

        if 'qs' in self.keys() and self.qs == 0:
            # 4d calculation
            out.qs = 0
            out.muzeta[:] = 0

        if 'spin_x' in self.keys():
            out.spin_x *= -1
            out.spin_z *= -1

        _reverse_strengths(out._data)

        # Remove Edwards-Teng elements for now
        if 'r11_edw_teng' in out._col_names:
            out.pop('r11_edw_teng')
            out.pop('r12_edw_teng')
            out.pop('r21_edw_teng')
            out.pop('r22_edw_teng')

        out._data['reference_frame'] = {
            'proper': 'reverse', 'reverse': 'proper'}[self.reference_frame]

        return out

    ind_per_table = []

    def add_strengths(self, line=None):
        """
        Add integrated element strengths to the Twiss table.

        The strength columns are computed from the elements in ``line`` and added
        to this table in place. If ``line`` is not provided, the line stored in the
        Twiss action is used when available.

        Parameters
        ----------
        line : xtrack.Line, optional
            Line from which the element strengths are read.

        Returns
        -------
        xtrack.TwissTable
            This Twiss table, with strength columns added when a line is available.
        """
        if line is None and hasattr(self,"_action"):
            line = self._action.line
        if line is not None:
            _add_strengths_to_twiss_res(self, line)
        return self

    @classmethod
    def concatenate(cls, tables_to_concat):
        """
        Concatenate compatible Twiss tables.

        The input tables are joined in order. Common boundary rows are removed
        when needed to avoid duplicating the shared element, and cyclic quantities
        such as phase advances are shifted to remain continuous across table
        boundaries.

        Parameters
        ----------
        tables_to_concat : sequence of xtrack.TwissTable
            Twiss tables to concatenate. All tables must have the same
            ``values_at`` and ``reference_frame``.

        Returns
        -------
        xtrack.TwissTable
            Concatenated Twiss table.
        """

        # Check values_at compatibility
        assert len(set([tt.values_at for tt in tables_to_concat])) == 1, (
            'All tables must have the same values_at')

        # Check reference_frame compatibility
        assert len(set([tt.reference_frame for tt in tables_to_concat])) == 1, (
            'All tables must have the same reference_frame')

        # trim away common markers
        ind_per_table = []
        for ii, tt in enumerate(tables_to_concat):
            this_ind = [0, len(tt)]
            if ii > 0:
                if tt.name[0] in tables_to_concat[ii-1].name:
                    assert tt.name[0] == tables_to_concat[ii-1].name[ind_per_table[ii-1][1]-1]
                    ind_per_table[ii-1][1] -= 1
            if ii < len(tables_to_concat) - 1:
                if tt.name[-1] == '_end_point':
                    this_ind[1] -= 1

            ind_per_table.append(this_ind)

        n_elem = sum([ind[1] - ind[0] for ind in ind_per_table])

        new_data = {}
        for kk in tables_to_concat[0]._col_names:
            if kk == 'W_matrix':
                new_data[kk] = np.empty(
                    (n_elem, 6, 6), dtype=tables_to_concat[0]._data[kk].dtype)
                continue
            dtype=tables_to_concat[0]._data[kk].dtype
            if dtype.str.startswith('<U'):
                str_len = np.max([int(tables_to_concat[ii]._data[kk].dtype.str.split('<U')[-1])
                                    for ii in range(len(tables_to_concat))])
                dtype = f'<U{str_len}'
            new_data[kk] = np.empty(n_elem, dtype=dtype)

        i_start = 0
        for ii, tt in enumerate(tables_to_concat):
            i_end = i_start + ind_per_table[ii][1] - ind_per_table[ii][0]
            for kk in tt._col_names:
                if kk == 'W_matrix':
                    new_data[kk][i_start:i_end] = (
                        tt._data[kk][ind_per_table[ii][0]:ind_per_table[ii][1], :, :])
                    continue
                new_data[kk][i_start:i_end] = (
                    tt._data[kk][ind_per_table[ii][0]:ind_per_table[ii][1]])
                if kk in CYCLICAL_QUANTITIES:
                    new_data[kk][i_start:i_end] -= new_data[kk][i_start]
                    if ii > 0:
                        new_data[kk][i_start:i_end] += new_data[kk][i_start-1]
                        new_data[kk][i_start:i_end] += (
                            tables_to_concat[ii-1]._data[kk][-1]
                            - tables_to_concat[ii-1]._data[kk][ind_per_table[ii-1][1]-1])

            i_start = i_end

        new_table = cls(new_data)
        new_table._data['values_at'] = tables_to_concat[0].values_at
        new_table._data['reference_frame'] = tables_to_concat[0].reference_frame
        new_table._data['particle_on_co'] = tables_to_concat[0].particle_on_co

        return new_table

    def zero_at(self, name):
        """
        Shift cyclic quantities to be zero at an element.

        The values of cyclic columns, such as phase advances, are shifted in place
        by subtracting their value at ``name``.

        Parameters
        ----------
        name : str
            Element name at which cyclic quantities are set to zero.
        """
        for kk in CYCLICAL_QUANTITIES:
            if kk in self:
                self[kk] -= self[kk, name]

    def target(self, tars=None, value=None, at=None, **kwargs):
        """
        Build matching targets from this Twiss table.

        This is a convenience wrapper around :class:`xtrack.TargetSet`. If
        ``value`` is not provided, this Twiss table is used as the reference value
        for the targets.

        Parameters
        ----------
        tars : str or sequence of str, optional
            Twiss quantities to target.
        value : float, xdeps.GreaterThan, xdeps.LessThan, xtrack.TwissTable, optional
            Target value. If not provided, this Twiss table is used.
        at : str, optional
            Element name at which the targets are evaluated.
        **kwargs
            Additional keyword arguments passed to :class:`xtrack.TargetSet`.

        Returns
        -------
        xtrack.TargetSet
            Matching targets built from the requested quantities.
        """
        if value is None:
            value = self
        tarset = xt.TargetSet(tars=tars, value=value, at=at,
                              action=self._action, **kwargs)
        return tarset

    def plot(self,
            yl=None,
            yr=None,x='s',
            lattice=True,
            mask=None,
            labels=None,
            clist="k r b g c m",
            figure=None,
            figlabel=None,
            ax=None,
            axleft=None,
            axright=None,
            axlattice=None,
            hover=False,
            grid=True,
            figsize=(6.4*1.2, 4.8),
            lattice_only=False
            ):
        """
        Plot columns of the TwissTable

        Parameters:
        -----------
        yl: str
            space separated columns or expressions to plot on the left y-axis
        yr: str
            space separated columns or expressions to plot on the right y-axis
        x: str
            column to plot on the x-axis
        lattice: bool
            if True, the lattice is plotted
        mask: slice
            mask to select the elements to plot
        labels: str
            mask to select the elements to label
        clist: str
            colors to use
        ax: matplotlib axis
            axis to plot on
        figlabel: str
            label to use for the figure
        """

        if yl is None and yr is None:
            yl='betx bety'
            yr='dx dy'
        if yl is None:
            yl=""
        if yr is None:
            yr=""

        if lattice and 'k2l' not in self.keys():
            self.add_strengths()

        if mask is not None:
            if isinstance(mask,str):
                idx=self.mask[mask]
            else:
                idx=mask
        else:
            idx=slice(None)

        self._is_s_begin=True

        if lattice_only:
            yl = ''
            yr = ''

        pl=TwissPlot(self,
                x=x,
                yl=yl,
                yr=yr,
                idx=idx,
                lattice=lattice,
                figure=figure,
                figlabel=figlabel,clist=clist,
                ax=ax,
                axleft=axleft,
                axright=axright,
                axlattice=axlattice,
                hover=hover,
                figsize=figsize,
                grid=grid,
                )

        if labels is not None:
            mask=self.mask[labels]
            labels=self[self._index][mask]
            xs=self[x][mask]
            pl.left.set_xticks(xs,labels)

        if lattice_only:
            ax1 = pl.lattice.twinx()
            ax1.yaxis.set_label_position("left")
            ax1.yaxis.set_ticks_position("left")
            ax1.set_autoscale_on(True)
            pl.left = ax1

        return pl

    def _get_radiation_integrals(self, add_to_tw=False):

        out = _compute_radiation_integrals(self)

        if add_to_tw:
            for nn in out._col_names:
                if nn.startswith('rad_int_'):
                    self[nn] = out[nn]
            for nn, vv in out._data.items():
                if nn.startswith('rad_int_') and nn not in out._col_names:
                    self._data[nn] = vv

        return out

    def _sort_col_names(self):
        old_col_names = self._col_names
        col_name_set = set(old_col_names)
        new_col_names = []
        for nn in DEFAULT_COL_ORDER:
            if nn in col_name_set:
                new_col_names.append(nn)
        set_sorted_col_names = set(new_col_names)
        for nn in old_col_names:
            if nn not in set_sorted_col_names:
                new_col_names.append(nn)
        self._col_names = new_col_names
