# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np

from .. import linear_normal_form as lnf
from ..general import _print
from .chromatic_functions import trapz
from .closed_orbit import ClosedOrbitSearchError
from .transfer_matrices import _complete_steps_r_matrix_with_default
from .twiss import twiss_line
from .twiss_table import TwissTable


class MultiBunchTwiss:

    """
    Container for the per-bunch Twiss results of a multi-bunch beam, as returned
    by :func:`twiss_line_multibunch`.

    Each bunch of the beam sits at a distinct longitudinal position ``zeta`` and,
    through a multi-bunch beam-beam element, experiences a different force. As a
    consequence its closed orbit and linear optics (in particular the tunes)
    differ from bunch to bunch. This object holds one :class:`TwissTable` per
    bunch and offers convenient per-bunch access to scalar quantities.

    - ``mbtw[i]`` returns the :class:`TwissTable` of bunch ``i``.
    - ``mbtw['qx']`` (or any scalar Twiss quantity) returns a numpy array with
      the value for each bunch; for a column quantity (e.g. ``mbtw['x']``)
      the full per-bunch columns are stacked into an (n_bunches, n_rows)
      array.
    - ``mbtw['x', 'ip1']`` / ``mbtw['x', list_of_names]`` return the column
      values at the given element(s) for every bunch, shape (n_bunches,) /
      (n_bunches, n_names). The row positions are resolved once (all bunch
      tables share the same row order), so this is the FAST way to extract
      per-bunch values at many elements.
    - Scalar quantities are also available as attributes, e.g. ``mbtw.qx``.
    - ``mbtw.zeta_bunches`` is the array of the bunch longitudinal positions.
    """

    def __init__(self, bunch_twiss, zeta_bunches, bunch_names=None):
        self.bunch_twiss = list(bunch_twiss)
        self.zeta_bunches = np.atleast_1d(np.asarray(zeta_bunches, dtype=float))
        self.num_bunches = len(self.bunch_twiss)
        if bunch_names is None:
            bunch_names = [f'bunch_{i}' for i in range(self.num_bunches)]
        self.bunch_names = list(bunch_names)
        self._name_pos = None

    def __len__(self):
        return self.num_bunches

    def __iter__(self):
        return iter(self.bunch_twiss)

    def rows_index(self, names):
        """Row index (or index array) of the given element name(s), shared by
        all bunch tables."""
        if self._name_pos is None:
            self._name_pos = {nn: ii for ii, nn in
                              enumerate(self.bunch_twiss[0].name)}
        if isinstance(names, str):
            return self._name_pos[names]
        return np.array([self._name_pos[nn] for nn in names])

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer, slice)):
            return self.bunch_twiss[key]
        if isinstance(key, tuple):
            col, names = key
            idx = self.rows_index(names)
            return np.array([tw[col][idx] for tw in self.bunch_twiss])
        return np.array([tw[key] for tw in self.bunch_twiss])

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            values = [tw[name] for tw in self.bunch_twiss]
        except (KeyError, NameError):
            raise AttributeError(name)
        if np.isscalar(values[0]):
            return np.array(values)
        raise AttributeError(
            f"'{name}' is not a per-bunch scalar quantity; use "
            f"mbtw['{name}'] or mbtw['{name}', element_names]")

    def bunch(self, name):
        """Return the TwissTable of the bunch with the given name."""
        return self.bunch_twiss[self.bunch_names.index(name)]

    def __repr__(self):
        return (f'MultiBunchTwiss({self.num_bunches} bunches, '
                f'zeta={np.array2string(self.zeta_bunches, precision=3)})')


def _mb_co_search(line, zeta_t, delta_t, Z_init, hs, co_tol, max_iter_co,
                  continue_on_closed_orbit_error=False):

    """Batched Newton closed-orbit search: for every target (a bunch at fixed
    ``zeta`` and ``delta``) track the closed-orbit candidate plus 8 transverse
    central-difference probes, all targets in ONE tracking call per iteration.
    Returns ``(Z, J, dzeta_turn)``: converged 4D closed orbits, one-turn 4x4
    Jacobians, and the one-turn zeta slippage of each closed orbit.

    ``continue_on_closed_orbit_error`` has the same meaning as in
    :func:`twiss_line`: if the search does not reach ``co_tol`` within
    ``max_iter_co`` iterations, return the last iterate instead of raising
    :class:`ClosedOrbitSearchError`.
    """

    ctx2np = line._context.nparray_from_context_array
    n_t = len(zeta_t)
    ZZ = Z_init.copy()
    converged = False
    for _ in range(max_iter_co):
        XX = np.repeat(ZZ[:, None, :], 9, axis=1)
        for kk in range(4):
            XX[:, 1 + 2 * kk, kk] += hs[kk]
            XX[:, 2 + 2 * kk, kk] -= hs[kk]
        pp = line.build_particles(
            x=XX[..., 0].ravel(), px=XX[..., 1].ravel(),
            y=XX[..., 2].ravel(), py=XX[..., 3].ravel(),
            zeta=np.repeat(zeta_t[:, None], 9, axis=1).ravel(),
            delta=np.repeat(delta_t[:, None], 9, axis=1).ravel())
        line.track(pp, num_turns=1)
        if not np.all(ctx2np(pp.state) > 0):
            raise ClosedOrbitSearchError(
                'Particles lost while tracking the multibunch finite-'
                'difference probes')
        order = np.argsort(ctx2np(pp.particle_id))
        out = np.stack([ctx2np(pp.x), ctx2np(pp.px),
                        ctx2np(pp.y), ctx2np(pp.py)], axis=-1)[order]
        out = out.reshape(n_t, 9, 4)
        zeta_out = ctx2np(pp.zeta)[order].reshape(n_t, 9)[:, 0]

        FF = out[:, 0, :] - ZZ
        JJ = np.empty((n_t, 4, 4))
        for kk in range(4):
            JJ[:, :, kk] = (out[:, 1 + 2 * kk, :]
                            - out[:, 2 + 2 * kk, :]) / (2 * hs[kk])
        res = np.abs(FF).max(axis=1)
        if res.max() < co_tol:
            converged = True
            break
        dZZ = np.linalg.solve(JJ - np.eye(4)[None, :, :], -FF[:, :, None])
        ZZ += dZZ[:, :, 0]
    if not converged:
        n_bad = int((res > co_tol).sum())
        if not continue_on_closed_orbit_error:
            raise ClosedOrbitSearchError(
                f'Multibunch closed-orbit search did not converge in '
                f'{max_iter_co} iterations (residual {res.max():.2e} on '
                f'{n_bad} of {n_t} bunches)')
        _print(f'  closed-orbit search: {max_iter_co} iterations reached, '
               f'residual {res.max():.2e} on {n_bad} of {n_t} bunches '
               f'-- continuing on closed-orbit error')
    return ZZ, JJ, zeta_out - zeta_t


def _mb_fractional_tunes(JJ):

    """Per-target fractional tunes from the 4x4 one-turn Jacobians
    (eigenvalues; modes classified by eigenvector dominance; the sign of
    R12/R34 resolves the q <-> 1-q mirror ambiguity)."""

    n_t = len(JJ)
    qx = np.full(n_t, np.nan)
    qy = np.full(n_t, np.nan)
    eigvals, eigvecs = np.linalg.eig(JJ)
    for nn in range(n_t):
        for ii in np.where(np.imag(eigvals[nn]) > 0)[0]:
            vv = eigvecs[nn][:, ii]
            is_x = (abs(vv[0])**2 + abs(vv[1])**2
                    >= abs(vv[2])**2 + abs(vv[3])**2)
            qq = np.angle(eigvals[nn][ii]) / (2 * np.pi)
            if is_x:
                qx[nn] = qq if JJ[nn][0, 1] >= 0 else 1 - qq
            else:
                qy[nn] = qq if JJ[nn][2, 3] >= 0 else 1 - qq
    return qx, qy


def _twiss_multibunch_fast(line, zeta_bunches, steps_R_matrix=None,
                           co_tol=1e-11, max_iter_co=20,
                           continue_on_closed_orbit_error=False,
                           compute_optics=False, delta_chrom=5e-5):

    """Batched per-bunch 4D closed solution (closed orbit + tunes + element-by-
    element orbit, optionally linear optics and global quantities) for a
    multi-bunch beam.

    Instead of one full twiss per bunch (each with its own closed-orbit search
    and finite-difference R-matrix probes), the finite-difference probes of ALL
    bunches are tracked together: per Newton iteration one single tracking call
    with ``n_bunches x 9`` particles (closed-orbit candidate + central-
    difference probes in x, px, y, py, each at its own fixed ``zeta``). The
    per-bunch 4x4 one-turn Jacobian from the probes drives a batched Newton
    update of all closed orbits simultaneously; the same Jacobians give the
    per-bunch fractional tunes. A final single tracking call with an element-
    by-element monitor records each bunch's closed orbit at every element.

    With ``compute_optics=False`` (mode='fast_orbit') the per-bunch
    :class:`TwissTable` has columns ``name, s, x, px, y, py`` and scalars
    ``qx``/``qy`` == ``qx_frac``/``qy_frac`` (only the FRACTIONAL tunes are
    available without the optics pass) and ``zeta0``.

    With ``compute_optics=True`` (mode='fast') the final monitored pass tracks
    the probes (plus two dispersion probes in delta) instead of only the
    closed orbits. The per-element transfer Jacobians propagate the Courant-
    Snyder normal form W (from :mod:`xtrack.linear_normal_form`, same
    conventions as standard twiss), adding columns ``betx, alfx, bety, alfy,
    mux, muy, dx, dpx, dy, dpy``. Global scalars follow the standard twiss
    naming: ``qx``/``qy`` are the ACCUMULATED tunes (integer part included,
    from mux/muy -- meaningful only if no single element advances the phase by
    more than one turn), ``qx_frac``/``qy_frac`` the exact fractional tunes,
    ``dqx``/``dqy`` the chromaticities (from two additional off-momentum
    closed-orbit searches at delta = +/- ``delta_chrom``), ``c_minus`` the
    closest-tune-approach coupling coefficient, and ``slip_factor``/
    ``slip_factor_dzeta_ddelta``/``momentum_compaction_factor`` the
    longitudinal slippage quantities of the off-momentum closed orbits.
    """

    ctx = line._context
    ctx2np = ctx.nparray_from_context_array
    zeta_bunches = np.atleast_1d(np.asarray(zeta_bunches, dtype=float))
    n_bunches = len(zeta_bunches)

    steps = _complete_steps_r_matrix_with_default(steps_R_matrix)
    hs = np.array([steps['dx'], steps['dpx'], steps['dy'], steps['dpy']])

    ZZ, JJ, _ = _mb_co_search(line, zeta_bunches, np.zeros(n_bunches),
                              np.zeros((n_bunches, 4)), hs, co_tol,
                              max_iter_co, continue_on_closed_orbit_error)

    qx_frac, qy_frac = _mb_fractional_tunes(JJ)

    ltab = line.get_table()
    names = np.asarray(ltab.name)
    s_all = np.asarray(ltab.s)
    n_pts = len(names)

    if not compute_optics:
        pp = line.build_particles(
            x=ZZ[:, 0], px=ZZ[:, 1], y=ZZ[:, 2], py=ZZ[:, 3],
            zeta=zeta_bunches, delta=0.)
        line.track(pp, num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
        rec = line.record_last_track
        x_ebe = ctx2np(rec.x)
        px_ebe = ctx2np(rec.px)
        y_ebe = ctx2np(rec.y)
        py_ebe = ctx2np(rec.py)

        bunch_twiss = []
        for nn in range(n_bunches):
            tab = TwissTable(dict(name=names, s=s_all,
                                  x=x_ebe[nn].copy(), px=px_ebe[nn].copy(),
                                  y=y_ebe[nn].copy(), py=py_ebe[nn].copy()),
                             periodic=True)
            tab._data['qx'] = float(qx_frac[nn])
            tab._data['qy'] = float(qy_frac[nn])
            tab._data['qx_frac'] = float(qx_frac[nn])
            tab._data['qy_frac'] = float(qy_frac[nn])
            tab._data['zeta0'] = float(zeta_bunches[nn])
            bunch_twiss.append(tab)
        return bunch_twiss

    dd = steps['ddelta']
    W0 = np.empty((n_bunches, 4, 4))
    for nn in range(n_bunches):
        M6 = np.eye(6)
        M6[:4, :4] = JJ[nn]
        WW, _, _, _ = lnf.get_linear_normal_form(M6, only_4d_block=True)
        W0[nn] = WW[:4, :4]

    zeta2 = np.concatenate([zeta_bunches, zeta_bunches])
    delta2 = np.concatenate([np.full(n_bunches, +delta_chrom),
                             np.full(n_bunches, -delta_chrom)])
    _, J2, dzeta2 = _mb_co_search(line, zeta2, delta2,
                                  np.concatenate([ZZ, ZZ]), hs, co_tol,
                                  max_iter_co)
    qxp, qyp = _mb_fractional_tunes(J2[:n_bunches])
    qxm, qym = _mb_fractional_tunes(J2[n_bunches:])

    def _wrap_half(v):
        return (v + 0.5) % 1.0 - 0.5

    dqx = _wrap_half(qxp - qxm) / (2 * delta_chrom)
    dqy = _wrap_half(qyp - qym) / (2 * delta_chrom)
    slip_dzeta_ddelta = (dzeta2[:n_bunches]
                         - dzeta2[n_bunches:]) / (2 * delta_chrom)
    line_length = line.get_length()
    gamma0 = float(ctx2np(line.particle_ref.gamma0)[0])
    if line_length > 0:
        slip_factor = -slip_dzeta_ddelta / line_length
        momentum_compaction = slip_factor + 1 / gamma0**2
    else:
        slip_factor = np.full(n_bunches, np.nan)
        momentum_compaction = np.full(n_bunches, np.nan)

    out_cols = {kk: np.empty((n_bunches, n_pts)) for kk in
                ['x', 'px', 'y', 'py', 'betx', 'alfx', 'bety', 'alfy',
                 'mux', 'muy', 'dx', 'dpx', 'dy', 'dpy']}
    cmin_pts = np.empty((n_bunches, n_pts))

    n_probes = 11
    chunk = max(1, int(4e6 // (n_probes * n_pts)))
    for c0 in range(0, n_bunches, chunk):
        c1 = min(c0 + chunk, n_bunches)
        nb = c1 - c0
        XX = np.repeat(ZZ[c0:c1, None, :], n_probes, axis=1)
        for kk in range(4):
            XX[:, 1 + 2 * kk, kk] += hs[kk]
            XX[:, 2 + 2 * kk, kk] -= hs[kk]
        delta = np.zeros((nb, n_probes))
        delta[:, 9] = +dd
        delta[:, 10] = -dd
        zz = np.repeat(zeta_bunches[c0:c1, None], n_probes, axis=1)
        pp = line.build_particles(
            x=XX[..., 0].ravel(), px=XX[..., 1].ravel(),
            y=XX[..., 2].ravel(), py=XX[..., 3].ravel(),
            zeta=zz.ravel(), delta=delta.ravel())
        line.track(pp, num_turns=1, turn_by_turn_monitor='ONE_TURN_EBE')
        if not np.all(ctx2np(pp.state) > 0):
            raise ClosedOrbitSearchError(
                'Particles lost while tracking the multibunch optics probes')
        rec = line.record_last_track
        traj = np.stack([ctx2np(rec.x), ctx2np(rec.px),
                         ctx2np(rec.y), ctx2np(rec.py)], axis=-1)
        traj = traj.reshape(nb, n_probes, n_pts, 4)

        for kk, key in enumerate(['x', 'px', 'y', 'py']):
            out_cols[key][c0:c1] = traj[:, 0, :, kk]

        MM = np.empty((nb, n_pts, 4, 4))
        for kk in range(4):
            MM[:, :, :, kk] = (
                traj[:, 1 + 2 * kk] - traj[:, 2 + 2 * kk]) / (2 * hs[kk])

        Tdelta = (traj[:, 9] - traj[:, 10]) / (2 * dd)
        D0 = np.linalg.solve(np.eye(4)[None, :, :] - JJ[c0:c1],
                             Tdelta[:, -1, :, None])[:, :, 0]
        DD = np.einsum('bpij,bj->bpi', MM, D0) + Tdelta
        for kk, key in enumerate(['dx', 'dpx', 'dy', 'dpy']):
            out_cols[key][c0:c1] = DD[:, :, kk]

        Ws = np.einsum('bpij,bjk->bpik', MM, W0[c0:c1])
        out_cols['betx'][c0:c1] = Ws[..., 0, 0]**2 + Ws[..., 0, 1]**2
        out_cols['alfx'][c0:c1] = -(Ws[..., 0, 0] * Ws[..., 1, 0]
                                    + Ws[..., 0, 1] * Ws[..., 1, 1])
        out_cols['bety'][c0:c1] = Ws[..., 2, 2]**2 + Ws[..., 2, 3]**2
        out_cols['alfy'][c0:c1] = -(Ws[..., 2, 2] * Ws[..., 3, 2]
                                    + Ws[..., 2, 3] * Ws[..., 3, 3])
        for th_key, (r0, r1) in [('mux', (0, 1)), ('muy', (2, 3))]:
            th = np.arctan2(Ws[..., r0, r1], Ws[..., r0, r0])
            dth = np.diff(th, axis=1) % (2 * np.pi)
            mu = np.concatenate(
                [np.zeros((nb, 1)), np.cumsum(dth, axis=1)], axis=1)
            out_cols[th_key][c0:c1] = mu / (2 * np.pi)

        c_r1 = (np.sqrt(Ws[..., 2, 0]**2 + Ws[..., 2, 1]**2)
                / np.sqrt(out_cols['betx'][c0:c1]))
        c_r2 = (np.sqrt(Ws[..., 0, 2]**2 + Ws[..., 0, 3]**2)
                / np.sqrt(out_cols['bety'][c0:c1]))
        dq_frac = np.abs(np.mod(out_cols['mux'][c0:c1, -1], 1)
                         - np.mod(out_cols['muy'][c0:c1, -1], 1))[:, None]
        cmin_pts[c0:c1] = (2 * np.sqrt(c_r1 * c_r2) * dq_frac
                           / (1 + c_r1 * c_r2))

    if line_length > 0:
        c_minus = trapz(cmin_pts, s_all, axis=1) / line_length
    else:
        c_minus = np.mean(cmin_pts, axis=1)

    bunch_twiss = []
    for nn in range(n_bunches):
        data = dict(name=names, s=s_all)
        for key, arr in out_cols.items():
            data[key] = arr[nn].copy()
        tab = TwissTable(data, periodic=True)
        tab._data['qx'] = float(out_cols['mux'][nn][-1])
        tab._data['qy'] = float(out_cols['muy'][nn][-1])
        tab._data['qx_frac'] = float(qx_frac[nn])
        tab._data['qy_frac'] = float(qy_frac[nn])
        tab._data['dqx'] = float(dqx[nn])
        tab._data['dqy'] = float(dqy[nn])
        tab._data['c_minus'] = float(c_minus[nn])
        tab._data['slip_factor'] = float(slip_factor[nn])
        tab._data['slip_factor_dzeta_ddelta'] = float(slip_dzeta_ddelta[nn])
        tab._data['momentum_compaction_factor'] = float(
            momentum_compaction[nn])
        tab._data['line_length'] = float(line_length)
        tab._data['zeta0'] = float(zeta_bunches[nn])
        bunch_twiss.append(tab)
    return bunch_twiss


def twiss_line_multibunch(line, zeta_bunches=None, particles=None,
                          method='4d', bunch_names=None,
                          show_progress=True, mode='fast', **kwargs):

    """
    Compute a closed (periodic) Twiss solution for each bunch of a multi-bunch
    beam.

    In a multi-bunch beam every bunch occupies a distinct longitudinal position
    ``zeta`` and, through a multi-bunch beam-beam element (e.g.
    :class:`xfields.BeamBeamBiGaussianRigidBunch2D`), sees a different force.
    This function fixes ``zeta`` to each bunch position in turn and computes the
    corresponding periodic solution, so that the per-bunch closed orbit and
    optics (including the coherent beam-beam tune shift) are obtained.

    The opposing beam is assumed frozen during the computation (as stored in the
    beam-beam element). For a self-consistent solution of two colliding beams,
    call this function alternately on each beam, updating the opposing-beam
    state from the freshly computed per-bunch closed orbits and iterating until
    convergence.

    Parameters
    ----------
    line : Line
        The beam line (containing the multi-bunch beam-beam element(s)).
    zeta_bunches : array_like, optional
        Longitudinal positions of the bunches. Mutually exclusive with
        ``particles``.
    particles : xpart.Particles, optional
        Particles object of the beam in which each active macroparticle is one
        bunch; the bunch positions are read from its ``zeta``. Mutually
        exclusive with ``zeta_bunches``.
    method : {'4d', '6d'}, optional
        Twiss method. Defaults to ``'4d'`` (``zeta`` is held fixed at each bunch
        position, which is the relevant regime for the coherent multi-bunch
        problem).
    bunch_names : list of str, optional
        Names for the bunches (used for labelling). Defaults to
        ``['bunch_0', 'bunch_1', ...]``.
    show_progress : bool, optional
        If True (default), display a ``tqdm`` progress bar over the bunches
        (``mode='full'`` only).
    mode : {'fast', 'fast_orbit', 'full'}, optional
        ``'fast'`` (default): the finite-difference probes of ALL bunches are
        tracked together (one tracking call per Newton iteration of the
        batched closed-orbit search, plus one monitored element-by-element
        call and two off-momentum closed-orbit searches), which is orders of
        magnitude faster than one twiss per bunch. The per-bunch result is a
        :class:`TwissTable` with the element-by-element closed orbit and
        linear optics (``name, s, x, px, y, py, betx, alfx, bety, alfy, mux,
        muy, dx, dpx, dy, dpy``) and global quantities following the standard
        twiss naming: ``qx``/``qy`` (tunes including integer),
        ``qx_frac``/``qy_frac`` (fractional tunes), ``dqx``/``dqy``
        (chromaticities), ``c_minus``, ``slip_factor``,
        ``momentum_compaction_factor``. Requires ``method='4d'``.
        ``'fast_orbit'``: as ``'fast'`` but the optics pass and the global
        quantities are skipped; the per-bunch :class:`TwissTable` contains
        only the closed orbit (``name, s, x, px, y, py``) and the fractional
        tunes (``qx`` == ``qx_frac``: the integer part is not available
        without the optics pass). About a third of the cost of ``'fast'`` --
        useful in iteration loops that only feed back orbits.
        ``'full'``: one full :func:`twiss_line` per bunch (complete
        :class:`TwissTable`); slow.
    **kwargs :
        Additional keyword arguments forwarded to :func:`twiss_line` /
        :meth:`Line.twiss` (e.g. ``nemitt_x``, ``chrom``, ...) in
        ``mode='full'``. ``zeta0`` must not be given (it is set internally to
        each bunch position). In ``mode='fast'``/``'fast_orbit'`` only
        ``chrom`` (accepted and ignored), ``co_tol``, ``max_iter_co`` and
        ``continue_on_closed_orbit_error`` are accepted; other kwargs raise.

    Returns
    -------
    MultiBunchTwiss
        Container with one table per bunch (a full :class:`TwissTable` in
        ``mode='full'``, a lightweight orbit/tunes table in ``mode='fast'``).
        See :class:`MultiBunchTwiss`.
    """

    if 'zeta0' in kwargs:
        raise ValueError(
            '`zeta0` cannot be provided to twiss_line_multibunch; it is set '
            'internally to each bunch position.')

    if (zeta_bunches is None) == (particles is None):
        raise ValueError('Provide exactly one of `zeta_bunches` or `particles`.')

    if particles is not None:
        state = particles._context.nparray_from_context_array(particles.state)
        mask = state > 0
        zeta_bunches = particles._context.nparray_from_context_array(
            particles.zeta)[mask]

    zeta_bunches = np.atleast_1d(np.asarray(zeta_bunches, dtype=float))
    num_bunches = len(zeta_bunches)

    if num_bunches == 0:
        raise ValueError('No bunches to twiss.')

    if mode in ('fast', 'fast_orbit'):
        if method != '4d':
            raise ValueError(f"mode='{mode}' requires method='4d'")
        unsupported = set(kwargs) - {'chrom', 'co_tol', 'max_iter_co',
                                     'continue_on_closed_orbit_error'}
        if unsupported:
            raise ValueError(
                f'kwargs {sorted(unsupported)} are not supported in '
                f"mode='{mode}'; use mode='full'")
        co_kwargs = {kk: kwargs[kk] for kk in
                     ('co_tol', 'max_iter_co', 'continue_on_closed_orbit_error')
                     if kk in kwargs}
        bunch_twiss = _twiss_multibunch_fast(
            line, zeta_bunches, compute_optics=(mode == 'fast'), **co_kwargs)
    elif mode == 'full':
        from tqdm.auto import tqdm
        bunch_twiss = []
        iterator = tqdm(zeta_bunches, desc='twiss_multibunch',
                        unit='bunch', disable=not show_progress)
        for zb in iterator:
            tw = twiss_line(line, method=method, zeta0=float(zb), **kwargs)
            bunch_twiss.append(tw)
    else:
        raise ValueError(
            f'Unknown mode {mode!r} (use "fast", "fast_orbit" or "full")')

    return MultiBunchTwiss(
        bunch_twiss, zeta_bunches, bunch_names=bunch_names)
