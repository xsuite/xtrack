"""TPSA backend of ``line.twiss``: exact derivatives where the regular twiss takes finite differences.

The ``TpsaTwiss`` backend is selected by calling ``line.twiss(tpsa=True)``.
It provides functions to calculate the closed orbit, the one-turn matrix,
the element-by-element W and the chromatic functions from one
element-by-element TPSA track. Everything else is taken from the regular twiss.

``M[i]`` is the Jacobian from the range start to the entry of element ``i``, and
``R = M[-1]`` the one of the whole range (the one-turn matrix when it is periodic).
``W`` is the normalizing matrix of ``R = W Rot W^-1`` (``linear_normal_form``), the
matrix the twiss reads betx, alfx, the dispersion .. off, propagated as
``W_i = M_i W``. At order 2 the same recording carries ``H[i]``, the second
derivatives, and

    dW_i = (H_i . direction) W + M_i dW

with ``direction = (u1, 1)`` the delta-derivative of the off-momentum closed orbit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

import xtrack as xt

from .. import linear_normal_form as lnf
from ..twiss.chromatic_functions import _chromatic_functions_requested
from ..twiss.optics_propagation import AT_TURN_FOR_TWISS
from ..twiss.periodic_solution import _set_4d_dispersion_columns
from ..twiss.twiss_backend import FiniteDifferenceTwiss
from .particles import COORDS, ParticlesTpsa

if TYPE_CHECKING:
    import madng_tpsa

    from ..twiss.twiss_init import TwissInit
    from ..twiss.twiss_table import TwissTable

_DELTA: int = 5                            # index of delta among the coordinates

# (j, k) of the quadratic monomials, in the order they are recorded.
_QUADRATIC_PAIRS: list[tuple[int, int]] = [(j, k) for j in range(6) for k in range(j, 6)]

# the second closed-orbit attempt of the regular search starts from here
_SECOND_ATTEMPT_SHIFT = np.array([1e-5, 1e-7, 1e-5, 1e-7, 1e-4, 1e-5])


def _scalar(particles: xt.Particles, name: str) -> float:
    return float(np.atleast_1d(getattr(particles, name))[0])


def _coordinates(particles: xt.Particles) -> np.ndarray:
    return np.array([_scalar(particles, c) for c in COORDS])


def _unsupported(**options: Any) -> None:
    """Raise for regular-twiss options the TPSA backend does not implement."""
    for name, value in options.items():
        if value:
            raise NotImplementedError(f"``{name}`` is not supported by the TPSA twiss")


def _range_indices(
    line: xt.Line, start: str | int | None, end: str | int | None
) -> tuple[int, int]:
    """First and last tracked element index. ``None`` is the whole line."""
    last = len(line._element_names_unique) - 1
    if start is None:
        start = 0
    elif isinstance(start, str):
        start = line._element_names_unique.index(start)
    if end is None or end == "_end_point":
        end = last
    elif isinstance(end, str):
        end = line._element_names_unique.index(end)
    return int(start), int(end)


def _monomials(descriptor: madng_tpsa.Descriptor, order: int) -> np.ndarray:
    """The monomials the monitor records: linear, plus quadratic at order 2."""
    length = descriptor.monomial_length
    linear = np.zeros((6, length), dtype=int)
    linear[np.arange(6), np.arange(6)] = 1
    if order < 2:
        return linear
    quadratic = np.zeros((len(_QUADRATIC_PAIRS), length), dtype=int)
    for row, (j, k) in enumerate(_QUADRATIC_PAIRS):
        quadratic[row, j] += 1
        quadratic[row, k] += 1
    return np.vstack([linear, quadratic])


def _new_map(
    reference_particle: xt.Particles, coordinates: np.ndarray, order: int
) -> ParticlesTpsa:
    return ParticlesTpsa(
        order=order,
        mass0=_scalar(reference_particle, "mass0"),
        q0=_scalar(reference_particle, "q0"),
        p0c=_scalar(reference_particle, "p0c"),
        **dict(zip(COORDS, coordinates)),
    )


def _one_turn(
    line: xt.Line,
    reference_particle: xt.Particles,
    coordinates: np.ndarray,
    start: int,
    end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Orbit and Jacobian after ``start..end`` from an unmonitored order-1 track."""
    tpsa_map = _new_map(reference_particle, coordinates, order=1)
    line.track(tpsa_map, ele_start=start, ele_stop=end + 1)
    return tpsa_map.const_part, tpsa_map.jacobian()


def _scalar_turn(
    line: xt.Line,
    reference_particle: xt.Particles,
    coordinates: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray | None:
    """Orbit after ``start..end`` from one scalar particle, ``None`` if it is lost."""
    particle = xt.Particles(
        _context=line._buffer.context,
        mass0=_scalar(reference_particle, "mass0"),
        q0=_scalar(reference_particle, "q0"),
        p0c=_scalar(reference_particle, "p0c"),
        **dict(zip(COORDS, coordinates)),
    )
    line.track(particle, ele_start=start, ele_stop=end + 1)
    if _scalar(particle, "state") <= 0:
        return None
    return _coordinates(particle)


def _newton_start_orbit(
    line: xt.Line,
    reference_particle: xt.Particles,
    guess: np.ndarray,
    start: int,
    end: int,
    periodic: bool,
    transverse_only: bool,
    tol: float,
    max_iter: int,
) -> tuple[np.ndarray | None, int, float]:
    """Start orbit whose image after ``end`` is itself (periodic) or ``guess``.

    Residuals come from scalar tracks, the Jacobian from an
    order-1 TPSA track taken once and again only if the contraction stalls.
    Returns ``None`` when not converged.
    """
    orbit = np.array(guess, dtype=float)
    block = slice(0, 4) if transverse_only else slice(0, 6)
    size = 4 if transverse_only else 6
    R = None
    previous_residual = np.inf
    for num_iter in range(1, max_iter + 1):
        image = _scalar_turn(line, reference_particle, orbit, start, end)
        if image is None:
            return None, num_iter, np.inf
        error = image - (orbit if periodic else guess)
        residual = np.max(np.abs(error[block]))
        if residual < tol:
            return orbit, num_iter, residual
        if R is None or residual > 0.5 * previous_residual:
            _, R = _one_turn(line, reference_particle, orbit, start, end)
        previous_residual = residual
        jacobian = R[block, block] - (np.eye(size) if periodic else 0.0)
        orbit[block] -= np.linalg.solve(jacobian, error[block])
    return None, num_iter, residual


class TpsaEbeTrack:
    """One element-by-element TPSA track over the elements ``start..end``, and its coefficients.

    Rows are the entries of the tracked elements plus a final row after ``end``.
    ``M`` is ``(rows, 6, 6)``, ``H`` is ``(rows, 6, 6, 6)`` and ``None`` below order 2.
    """

    def __init__(
        self,
        line: xt.Line,
        reference_particle: xt.Particles,
        coordinates: np.ndarray,
        order: int,
        start: int,
        end: int,
    ) -> None:
        self.start, self.end, self.order = start, end, order
        self.coordinates = np.array(coordinates, dtype=float)

        self.map = _new_map(reference_particle, self.coordinates, order)
        monomials = _monomials(self.map.descriptor, order)
        line.track(self.map,
                   ele_start=start, ele_stop=end + 1,
                   multi_element_monitor_at=np.arange(start, end + 1),
                   monitor_monomials=monomials)
        monitor = line.tracker.record_multi_element_last_track

        num_rows = end - start + 2
        self.M = np.zeros((num_rows, 6, 6))
        for i in range(6):
            for j in range(6):
                self.M[:-1, i, j] = monitor.coefficient(i, monomials[j], turn=0)
        self.M[-1] = self.map.jacobian()

        self.H = None
        if order >= 2:
            self.H = np.zeros((num_rows, 6, 6, 6))
            for i in range(6):
                for row, (j, k) in enumerate(_QUADRATIC_PAIRS):
                    coefficients = np.empty(num_rows)
                    coefficients[:-1] = monitor.coefficient(
                        i, monomials[6 + row], turn=0)
                    coefficients[-1] = self.map.coefficient(i, monomials[6 + row])
                    # the coefficient of z_j**2 is half the second derivative
                    if j == k:
                        coefficients *= 2.0
                    self.H[:, i, j, k] = coefficients
                    self.H[:, i, k, j] = coefficients

        self.orbit = np.zeros((num_rows, 6))
        for i, coord in enumerate(COORDS):
            self.orbit[:-1, i] = monitor.get(coord, turn=0)[0]
        self.orbit[-1] = self.map.const_part

        # vector potential, nonzero at boundaries inside a sliced solenoid
        self.ax = np.append(monitor.get("ax", turn=0)[0],
                            self.map._local_series["ax"].const_part)
        self.ay = np.append(monitor.get("ay", turn=0)[0],
                            self.map._local_series["ay"].const_part)

        # not the map's s: it starts at 0 whatever the range and is reset at the turn end
        tracker_data = line.tracker._tracker_data_base
        s_locations = tracker_data.element_s_locations
        s_after_end = (s_locations[end + 1] if end + 1 < len(s_locations)
                       else tracker_data.line_length)
        self.s = np.append(s_locations[start:end + 1], s_after_end)

    @property
    def R(self) -> np.ndarray:
        """The transfer matrix of the whole tracked range."""
        return self.M[-1]

    def covers(self, coordinates: np.ndarray, order: int, start: int, end: int) -> bool:
        return (self.order >= order and self.start == start and self.end == end
                and np.allclose(self.coordinates, coordinates, rtol=0, atol=1e-14))


def _chromatic_functions_from_map(twiss_config: dict[str, Any]) -> bool:
    """Whether the chromatic functions come from the order-2 map.

    6d has no delta variable to differentiate against, and an open twiss would need
    the delta-derivative of the supplied init. Both stay finite differences.
    """
    return (_chromatic_functions_requested(twiss_config)
            and twiss_config["periodic"] and twiss_config["method"] == "4d")


class TpsaTwiss:
    """Twiss derivatives from element-by-element TPSA tracks.

    One instance lives for one ``line.twiss`` call and keeps its last recording, so
    R, W and the chromatic functions share one track. Options that reach the three
    backend methods are checked there, the rest in ``from_twiss_config``.
    """

    def __init__(self, order: int) -> None:
        self.order = order
        self._recording = None

    @classmethod
    def from_twiss_config(cls, twiss_config: dict[str, Any]) -> TpsaTwiss:
        """Reject what TPSA tracking cannot do and pick the map order."""
        _unsupported(polarization_analysis=twiss_config["polarization_analysis"],
                     radiation_analysis=twiss_config["radiation_analysis"])
        if twiss_config["line"]._radiation_model is not None:
            raise NotImplementedError("the TPSA twiss does not support radiation")
        if twiss_config["tpsa"] is not True:
            raise ValueError("``tpsa`` must be True or False")
        return cls(order=2 if _chromatic_functions_from_map(twiss_config) else 1)

    def _track(
        self,
        line: xt.Line,
        reference_particle: xt.Particles,
        coordinates: np.ndarray,
        start: str | int | None,
        end: str | int | None,
    ) -> TpsaEbeTrack:
        """The element-by-element recording, reused if the last one covers the request."""
        start, end = _range_indices(line, start, end)
        if (self._recording is None
                or not self._recording.covers(coordinates, self.order, start, end)):
            self._recording = TpsaEbeTrack(line, reference_particle, coordinates,
                                           self.order, start, end)
        return self._recording

    def find_closed_orbit(
        self,
        line: xt.Line,
        method: str,
        co_guess: xt.Particles | None,
        particle_ref: xt.Particles | None,
        co_search_settings: dict[str, Any] | None,
        continue_on_closed_orbit_error: bool,
        delta0: float | None,
        zeta0: float | None,
        zeta_shift: float | None,
        start: str | int | None,
        end: str | int | None,
        num_turns: int,
        co_search_at: str | None,
        search_for_t_rev: bool,
        spin: bool,
        num_turns_search_t_rev: int,
        symmetrize: bool,
        include_collective: bool,
        tol: float = 1e-13,
        max_iter: int = 20,
    ) -> xt.Particles:
        """Newton search for the periodic orbit, Jacobian from an order-1 map.

        In ``4d`` zeta and delta are held and only the transverse block is solved.
        """
        from ..twiss.closed_orbit import ClosedOrbitSearchError

        _unsupported(co_search_settings=co_search_settings, zeta_shift=zeta_shift,
                     co_search_at=co_search_at, search_for_t_rev=search_for_t_rev,
                     spin=spin, symmetrize=symmetrize,
                     include_collective=include_collective)
        if num_turns > 1:
            raise NotImplementedError("``num_turns`` > 1 is not supported by the TPSA twiss")

        if co_guess is None:
            if particle_ref is None:
                particle_ref = line.particle_ref
            if particle_ref is None:
                raise ValueError(
                    "Either ``co_guess`` or ``particle_ref`` must be provided")
            co_guess = particle_ref.copy()
            for name in COORDS:
                setattr(co_guess, name, 0)
            co_guess.s = 0
            co_guess.at_element = _range_indices(line, start, end)[0]
        particle_on_co = co_guess.copy(_context=line._buffer.context)
        particle_on_co.at_turn = AT_TURN_FOR_TWISS

        guess = _coordinates(particle_on_co)
        if delta0 is not None:
            guess[_DELTA] = delta0
        if zeta0 is not None:
            guess[4] = zeta0

        start_index, end_index = _range_indices(line, start, end)
        for attempt in (0.0, 1.0):
            orbit, num_iter, residual = _newton_start_orbit(
                line, particle_on_co, guess + attempt * _SECOND_ATTEMPT_SHIFT,
                start_index, end_index, periodic=True,
                transverse_only=(method == "4d"), tol=tol, max_iter=max_iter)
            if orbit is not None:
                break
            xt.general._print("Warning! Need second attempt on closed orbit search")
        else:
            if not continue_on_closed_orbit_error:
                raise ClosedOrbitSearchError(
                    f"TPSA closed orbit did not converge in {max_iter} iterations "
                    f"(residual {residual:.3e})")
            orbit = guess

        for name, value in zip(COORDS, orbit):
            setattr(particle_on_co, name, value)
        particle_on_co._fsolve_info = {"num_iter": num_iter, "residual": residual}
        return particle_on_co

    def get_R_matrix(
        self,
        line: xt.Line,
        particle_on_co: xt.Particles,
        start: str | int | None,
        end: str | int | None,
        method: str,
        steps_R_matrix: dict[str, float],
        num_turns: int,
        element_by_element: bool,
        only_markers: bool,
        include_collective: bool,
        symplectify: bool,
        matrix_responsiveness_tol: float | None,
        nemitt_x: float,
        nemitt_y: float,
        factor_adapt_steps: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Read ``R`` off the recording, with the per-element ``M`` if asked."""
        _unsupported(only_markers=only_markers, include_collective=include_collective)
        if num_turns > 1:
            raise NotImplementedError("``num_turns`` > 1 is not supported by the TPSA twiss")
        track = self._track(line, particle_on_co, _coordinates(particle_on_co),
                            start, end)
        if matrix_responsiveness_tol is not None:
            lnf._assert_matrix_responsiveness(track.R, matrix_responsiveness_tol,
                                              only_4d=(method == "4d"))
        return track.R, (track.M if element_by_element else None)

    def track_orbit_and_W(
        self,
        line: xt.Line,
        init: TwissInit,
        start: str | int,
        end: str | int,
        twiss_orientation: str,
        nemitt_x: float,
        nemitt_y: float,
        step_W_sigma: float,
        delta_disp: float,
        spin: bool,
        continue_if_lost: bool,
        keep_tracking_data: bool,
        keep_initial_particles: bool,
        initial_particles: xt.Particles | None,
        ebe_monitor: Any,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, int, int, dict[str, Any]]:
        """Propagate the orbit and the normalizing matrix to every row.

        A backward twiss is given ``W`` after the last element, so the ``W`` the rows
        propagate from is the solution of ``R W_start = W_end``.

        Returns the orbit columns, ``W`` per row, the first and one-past-last element index
        the table names its rows from, and the ``extra_data`` the regular twiss puts its tracking data in
        (always empty here, that option is unsupported).
        """
        _unsupported(spin=spin, keep_tracking_data=keep_tracking_data,
                     keep_initial_particles=keep_initial_particles,
                     initial_particles=initial_particles, ebe_monitor=ebe_monitor)
        particle_on_co = init.particle_on_co
        if twiss_orientation == "forward":
            track = self._track(line, particle_on_co, _coordinates(particle_on_co),
                                start, end)
            W_start = init.W_matrix
        else:
            track = self._shoot_to_end(line, particle_on_co, start, end)
            W_start = np.linalg.solve(track.R, init.W_matrix)

        # ptau and the kin_* columns are xsuite's own algebra over the tracked orbit
        orbit_particles = xt.Particles(
            mass0=_scalar(particle_on_co, "mass0"),
            q0=_scalar(particle_on_co, "q0"),
            p0c=_scalar(particle_on_co, "p0c"),
            ax=track.ax, ay=track.ay,
            **{c: track.orbit[:, i] for i, c in enumerate(COORDS)},
        )
        orbit = {c: track.orbit[:, i].copy() for i, c in enumerate(COORDS)}
        orbit["s"] = track.s.copy()
        for name in ("ptau", "kin_px", "kin_py", "kin_ps", "kin_xp", "kin_yp"):
            orbit[name] = np.array(getattr(orbit_particles, name))

        return orbit, track.M @ W_start, track.start, track.end + 1, {}

    def _shoot_to_end(
        self,
        line: xt.Line,
        particle_on_co: xt.Particles,
        start: str | int,
        end: str | int,
        tol: float = 1e-13,
        max_iter: int = 20,
    ) -> TpsaEbeTrack:
        """The recording from the start orbit that reaches ``particle_on_co`` after ``end``."""
        start_index, end_index = _range_indices(line, start, end)
        orbit, _, residual = _newton_start_orbit(
            line, particle_on_co, _coordinates(particle_on_co), start_index, end_index,
            periodic=False, transverse_only=False, tol=tol, max_iter=max_iter)
        if orbit is None:
            raise RuntimeError(
                f"TPSA backward twiss: no start orbit reaches the init in {max_iter} "
                f"iterations (residual {residual:.3e})")
        return self._track(line, particle_on_co, orbit, start, end)

    def chromatic_functions(
        self, twiss_config: dict[str, Any], twiss_res: TwissTable
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """The chromatic columns from the order-2 coefficients of the twiss's own track."""
        if not _chromatic_functions_from_map(twiss_config):
            return FiniteDifferenceTwiss().chromatic_functions(twiss_config, twiss_res)

        particle_on_co = twiss_config["init"].particle_on_co
        track = self._track(twiss_config["line"], particle_on_co,
                            _coordinates(particle_on_co),
                            twiss_config["start"], twiss_config["end"])
        R = track.R
        _, u2, direction, dR = _off_momentum_expansion(track.H[-1], R)
        W, dW, dq = _4d_w_matrix_and_derivatives(R, dR, u2)

        dWs = np.einsum("nijk,k->nij", track.H, direction) @ W + track.M @ dW
        columns = _chromatic_columns(track.M @ W, dWs)
        return columns, {"dqx": dq[0], "dqy": dq[1]}


def _off_momentum_expansion(
    H: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The delta-expansion of the closed orbit and of the one-turn matrix.

    ``x_co(delta) = u1 delta + u2 delta**2`` about the on-momentum orbit, and the
    linear map about that orbit is ``R + delta * dR``. Only the transverse block is
    made periodic, as in a 4d twiss.
    """
    A = R[:4, :4] - np.eye(4)
    u1 = -np.linalg.solve(A, R[:4, _DELTA])
    direction = np.zeros(6)                    # first-order shift per unit delta
    direction[:4], direction[_DELTA] = u1, 1.0
    dR = np.einsum("ijk,k->ij", H, direction)
    # delta**2 of the periodicity condition: the map's own quadratic part
    # contracted twice with the first-order shift.
    quadratic = 0.5 * np.einsum("ijk,j,k->i", H, direction, direction)
    u2 = -np.linalg.solve(A, quadratic[:4])
    return u1, u2, direction, dR


def _4d_w_matrix_and_derivatives(
    R: np.ndarray, dR: np.ndarray, u2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """The 4d W of ``lnf``, its delta-derivative and ``dqx, dqy``.

    First-order eigenvector perturbation, so it needs distinct eigenvalues: two planes
    on the same tune make the derivative blow up.
    """
    M = lnf._with_dummy_longitudinal_block(R)
    dM = dR.copy()
    dM[4:, :] = dM[:, 4:] = 0

    eigenvalues, eigenvectors = np.linalg.eig(M)
    # the dummy modes live in the longitudinal block and dM does not touch them.
    # A transverse tune on the dummy tune mixes the eigenvectors instead.
    dummy = np.all(np.abs(eigenvectors[:4, :]) < 1e-12, axis=0)
    if dummy.sum() != 2:
        raise ValueError("a transverse tune coincides with the dummy longitudinal tune")
    projected = np.linalg.solve(eigenvectors, dM @ eigenvectors)
    projected[dummy, :] = projected[:, dummy] = 0.0
    gaps = eigenvalues[None, :] - eigenvalues[:, None]   # gaps[l, k] = lam_k - lam_l
    np.fill_diagonal(gaps, np.inf)             # own-mode component is a free scale
    transverse_gaps = np.abs(gaps[~dummy][:, ~dummy])
    if transverse_gaps.min() < 1e-9:
        raise ValueError("degenerate transverse tunes, the chromatic W is undefined")
    deigenvectors = eigenvectors @ np.divide(
        projected, gaps, out=np.zeros_like(projected), where=projected != 0)

    modes = lnf.sort_modes(eigenvectors, eigenvalues)
    W, dW = lnf._build_w_matrix_and_derivative_from_eigenvectors(
        eigenvectors, deigenvectors, modes, only_4d_block=True)
    dq = [(projected[mode, mode] / eigenvalues[mode]).imag / (2 * np.pi)
          for mode in modes[:2]]

    # the longitudinal columns hold the dispersion instead
    _set_4d_dispersion_columns(W, R)
    A = R[:4, :4] - np.eye(4)
    dW[4:, :] = dW[:, 4:] = 0
    dW[:4, _DELTA] = 2.0 * u2                  # d(dispersion)/ddelta
    dW[:4, 4] = -np.linalg.solve(A, dR[:4, 4] + dR[:4, :4] @ W[:4, 4])
    return W, dW, dq


def _chromatic_columns(Ws: np.ndarray, dWs: np.ndarray) -> dict[str, np.ndarray]:
    """The chromatic functions from W and ``dW/ddelta`` at every element (MAD-8 6.3)."""
    columns = {}
    for plane, i0 in (("x", 0), ("y", 2)):
        w00, w01 = Ws[:, i0, i0], Ws[:, i0, i0 + 1]
        w10, w11 = Ws[:, i0 + 1, i0], Ws[:, i0 + 1, i0 + 1]
        d00, d01 = dWs[:, i0, i0], dWs[:, i0, i0 + 1]
        d10, d11 = dWs[:, i0 + 1, i0], dWs[:, i0 + 1, i0 + 1]

        bet = w00**2 + w01**2
        dbet = 2 * (w00 * d00 + w01 * d01)
        alf = -(w00 * w10 + w01 * w11)
        dalf = -(d00 * w10 + w00 * d10 + d01 * w11 + w01 * d11)

        b_chrom = dbet / bet
        a_chrom = dalf - dbet * alf / bet
        columns[f"b{plane}_chrom"] = b_chrom
        columns[f"a{plane}_chrom"] = a_chrom
        columns[f"w{plane}_chrom"] = np.sqrt(a_chrom**2 + b_chrom**2)
        columns[f"dmu{plane}"] = (w00 * d01 - w01 * d00) / bet / (2 * np.pi)

    dzeta = Ws[:, 4, _DELTA]
    columns["dzeta"] = dzeta - dzeta[0]
    for i, name in enumerate(("ddx", "ddpx", "ddy", "ddpy")):
        columns[name] = dWs[:, i, _DELTA]      # d(dispersion)/ddelta
    return columns
