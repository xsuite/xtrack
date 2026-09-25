"""TPSA backend of ``line.twiss``: exact derivatives where the regular twiss takes finite differences.

The ``TpsaTwiss`` backend is selected by calling ``line.twiss(tpsa=True)``.
It provides functions to calculate the closed orbit, the one-turn matrix,
the element-by-element W and the chromatic functions from one
element-by-element TPSA track. Everything else is taken from the regular twiss.

``R_matrices_ebe[i]`` is the Jacobian from the range start to the entry of element ``i``,
``R_matrix = R_matrices_ebe[-1]`` the one of the whole range (one-turn matrix when it is periodic).
``W_matrix`` is the normalizing matrix of ``R_matrix = W Rot W^-1`` (``linear_normal_form``).
This is the matrix the twiss reads betx, alfx, the dispersion, etc. off, propagated as
``W_matrices_ebe[i] = R_matrices_ebe[i] W_matrix``.
At order 2 the same recording carries ``hessians_ebe[i]``, the second derivatives, and

    dW_i/ddelta = (hessians_ebe[i] . orbit_shift_per_delta) W_matrix
                  + R_matrices_ebe[i] dW_matrix/ddelta

with ``orbit_shift_per_delta`` the delta-derivative of the off-momentum closed orbit.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import madng_tpsa
import numpy as np

import xtrack as xt

from .. import linear_normal_form as lnf
from ..twiss.chromatic_functions import _chromatic_functions_requested
from ..twiss.optics_propagation import AT_TURN_FOR_TWISS
from ..twiss.periodic_solution import _set_4d_dispersion_columns
from ..twiss.twiss_backend import FiniteDifferenceTwiss
from ._knobs import KnobParameters
from .particles import COORDS, ParticlesTpsa
from .particles import _SPIN_COORDS as SPIN_COORDS

if TYPE_CHECKING:
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


def _monomial(length: int, *variables: int) -> np.ndarray:
    """Build the exponent vector of the product of ``variables``, a repeat raises the power."""
    return np.bincount(variables, minlength=length)


def _derivative_weights(monomials: np.ndarray) -> np.ndarray:
    """Factor from a Taylor coefficient to the derivative, the product of the exponents' factorials."""
    return np.array([math.prod(math.factorial(n) for n in row) for row in monomials])


def _recorded_monomials(
    descriptor: madng_tpsa.Descriptor, hessians: bool
) -> dict[str, np.ndarray]:
    """The monitor's monomials, by the quantity they give.

    ``linear``: R. ``quadratic``: the Hessians. ``var_knob``: dR/dknob, variable-major.
    ``knob``: dorbit/dknob.
    """
    length = descriptor.monomial_length
    knobs = range(6, 6 + descriptor.num_params)
    blocks = {"linear": np.array([_monomial(length, j) for j in range(6)])}
    if hessians:
        blocks["quadratic"] = np.array([_monomial(length, j, k) for j, k in _QUADRATIC_PAIRS])
    if descriptor.num_params > 0:
        blocks["var_knob"] = np.array([_monomial(length, j, knob)
                                       for j in range(6) for knob in knobs])
        blocks["knob"] = np.array([_monomial(length, knob) for knob in knobs])
    return blocks


def _block_slices(blocks: dict[str, np.ndarray]) -> dict[str, slice]:
    """Where each block sits in ``np.vstack(list(blocks.values()))``."""
    slices, start = {}, 0
    for name, monomials in blocks.items():
        slices[name] = slice(start, start + len(monomials))
        start += len(monomials)
    return slices


def _linear_and_knob_blocks(
    coefficients: np.ndarray, blocks: dict[str, np.ndarray], num_knobs: int
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Split ``(rows, 6, monomials)`` coefficients into R, dR/dknob and dorbit/dknob."""
    block = _block_slices(blocks)
    R_matrices = coefficients[:, :, block["linear"]]
    if not num_knobs:
        return R_matrices, None, None
    dR_dknob = coefficients[:, :, block["var_knob"]].reshape(
        len(coefficients), 6, 6, num_knobs)
    return R_matrices, dR_dknob, coefficients[:, :, block["knob"]]


def _new_map(
    reference_particle: xt.Particles, coordinates: np.ndarray, order: int,
    spin: bool = False, descriptor: madng_tpsa.Descriptor | None = None,
) -> ParticlesTpsa:
    # Zero spin skips the spin code, so seed it only when asked
    spin_seed = ({name: _scalar(reference_particle, name) for name in SPIN_COORDS}
                 if spin else {})
    return ParticlesTpsa(
        order=order,
        descriptor=descriptor,
        mass0=_scalar(reference_particle, "mass0"),
        q0=_scalar(reference_particle, "q0"),
        p0c=_scalar(reference_particle, "p0c"),
        anomalous_magnetic_moment=_scalar(reference_particle, "anomalous_magnetic_moment"),
        **spin_seed,
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
    R_matrix = None
    previous_residual = np.inf
    for num_iter in range(1, max_iter + 1):
        image = _scalar_turn(line, reference_particle, orbit, start, end)
        if image is None:
            return None, num_iter, np.inf
        error = image - (orbit if periodic else guess)
        residual = np.max(np.abs(error[block]))
        if residual < tol:
            return orbit, num_iter, residual
        if R_matrix is None or residual > 0.5 * previous_residual:
            _, R_matrix = _one_turn(line, reference_particle, orbit, start, end)
        previous_residual = residual
        jacobian = R_matrix[block, block] - (np.eye(size) if periodic else 0.0)
        orbit[block] -= np.linalg.solve(jacobian, error[block])
    return None, num_iter, residual


class TpsaEbeTrack:
    """One element-by-element TPSA track over the elements ``start..end``, and its coefficients.

    Rows are the entries of the tracked elements plus a final row after ``end``.
    ``R_matrices_ebe`` is ``(rows, 6, 6)``, ``hessians_ebe`` is ``(rows, 6, 6, 6)`` or
    ``None`` when not asked for.
    ``spin`` is the ``(rows, 3)`` spin of the reference particle along the orbit.
    With ``knobs``, ``dR_dknob_ebe`` is ``(rows, 6, 6, knobs)`` and ``dorbit_dknob_ebe``
    ``(rows, 6, knobs)``, both at fixed start orbit.
    """

    def __init__(
        self,
        line: xt.Line,
        reference_particle: xt.Particles,
        coordinates: np.ndarray,
        order: int,
        start: int,
        end: int,
        spin: bool = False,
        knobs: list[str] | None = None,
        descriptor: madng_tpsa.Descriptor | None = None,
        hessians: bool = False,
    ) -> None:
        if hessians and order < 2:
            raise ValueError("the Hessians need an order-2 map")
        self.start, self.end, self.order = start, end, order
        self.knobs = list(knobs) if knobs else []
        self.coordinates = np.array(coordinates, dtype=float)

        self.map = _new_map(reference_particle, self.coordinates, order, spin=spin,
                            descriptor=descriptor)
        blocks = _recorded_monomials(self.map.descriptor, hessians)
        # Only around this track: scalar tracks cannot run on TPSA strengths
        knob_parameters = (KnobParameters(line, self.knobs, self.map.descriptor)
                           if self.knobs else None)
        try:
            if knob_parameters is not None:
                knob_parameters.apply()
            line.track(self.map,
                       ele_start=start, ele_stop=end + 1,
                       multi_element_monitor_at=np.arange(start, end + 1),
                       monitor_monomials=np.vstack(list(blocks.values())))
        finally:
            if knob_parameters is not None:
                knob_parameters.teardown()
        monitor = line.tracker.record_multi_element_last_track

        num_rows = end - start + 2
        # (rows, coord, monomial), the row after the last element off the map
        coefficients = np.concatenate([
            monitor.coefficients_by_coord(turn=0),
            self.map.coefficient_table_at_indices(monitor.coefficient_indices_by_monomial())[None]])
        self.R_matrices_ebe, self.dR_dknob_ebe, self.dorbit_dknob_ebe = (
            _linear_and_knob_blocks(coefficients, blocks, len(self.knobs)))

        self.hessians_ebe = None
        if hessians:
            block = _block_slices(blocks)
            quadratic = (coefficients[:, :, block["quadratic"]]
                         * _derivative_weights(blocks["quadratic"]))
            self.hessians_ebe = np.zeros((num_rows, 6, 6, 6))
            for row, (j, k) in enumerate(_QUADRATIC_PAIRS):
                self.hessians_ebe[:, :, j, k] = self.hessians_ebe[:, :, k, j] = quadratic[:, :, row]

        self.orbit = np.zeros((num_rows, 6))
        for i, coord in enumerate(COORDS):
            self.orbit[:-1, i] = monitor.get(coord, turn=0)[0]
        self.orbit[-1] = self.map.const_part
        self.spin = np.zeros((num_rows, 3))
        for i, name in enumerate(SPIN_COORDS):
            self.spin[:-1, i] = monitor.get(name, turn=0)[0]
            self.spin[-1, i] = getattr(self.map, name).const_part

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
    def R_matrix(self) -> np.ndarray:
        """The transfer matrix of the whole tracked range."""
        return self.R_matrices_ebe[-1]

    def covers(
        self, coordinates: np.ndarray, spin: np.ndarray, order: int, start: int, end: int,
        hessians: bool,
    ) -> bool:
        """Whether this recording answers the request, so it need not be tracked again."""
        same_range = self.start == start and self.end == end
        same_start_point = (np.allclose(self.coordinates, coordinates, rtol=0, atol=1e-14)
                            and np.allclose(self.spin[0], spin, rtol=0, atol=1e-14))
        has_hessians = self.hessians_ebe is not None or not hessians
        return self.order >= order and same_range and same_start_point and has_hessians


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
    the R matrix, the W matrices and the chromatic functions share one track. Options that reach the three
    backend methods are checked there, the rest in ``from_twiss_config``.
    """

    def __init__(self, chromatic: bool, spin: bool = False, knobs: list[str] | None = None,
                 periodic: bool = True) -> None:
        self.knobs = list(knobs) if knobs else []
        # var * knob for dR/dknob
        self.map_order = 2 if chromatic or self.knobs else 1
        # a periodic knob twiss moves the closed orbit, which enters dR/dknob through the Hessians
        self.record_hessians = chromatic or (periodic and bool(self.knobs))
        self.spin = spin
        self.descriptor = (madng_tpsa.Descriptor(variables=COORDS, order=self.map_order,
                                                 params=self.knobs, param_order=1)
                           if self.knobs else None)
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
        return cls(chromatic=_chromatic_functions_from_map(twiss_config),
                   spin=twiss_config["spin"], knobs=twiss_config["knobs"],
                   periodic=twiss_config["periodic"])

    def _track(
        self,
        line: xt.Line,
        reference_particle: xt.Particles,
        coordinates: np.ndarray,
        start: str | int | None,
        end: str | int | None,
    ) -> TpsaEbeTrack:
        """The element-by-element recording, reused if the last one covers the request.

        A spin twiss seeds the map with the reference particle's spin. Spin shares
        the radiation compile flag, so a spin twiss switches it on for the track.
        """
        start, end = _range_indices(line, start, end)
        spin = (np.array([_scalar(reference_particle, name) for name in SPIN_COORDS])
                if self.spin else np.zeros(len(SPIN_COORDS)))
        if (self._recording is None
                or not self._recording.covers(coordinates, spin, self.map_order, start, end,
                                              self.record_hessians)):
            with xt.line._preserve_config(line):
                if self.spin:
                    line.config.XTRACK_MULTIPOLE_NO_SYNRAD = False
                self._recording = TpsaEbeTrack(line, reference_particle, coordinates,
                                               self.map_order, start, end, spin=self.spin,
                                               knobs=self.knobs, descriptor=self.descriptor,
                                               hessians=self.record_hessians)
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
                     symmetrize=symmetrize, include_collective=include_collective)
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
        if spin:
            from ..twiss.spin import _find_spin_fixed_point
            for name, value in zip(SPIN_COORDS,
                                   _find_spin_fixed_point(line, particle_on_co)):
                setattr(particle_on_co, name, value)
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
        """Read the R matrix off the recording, with the per-element ones if asked."""
        _unsupported(only_markers=only_markers, include_collective=include_collective)
        if num_turns > 1:
            raise NotImplementedError("``num_turns`` > 1 is not supported by the TPSA twiss")
        track = self._track(line, particle_on_co, _coordinates(particle_on_co),
                            start, end)
        if matrix_responsiveness_tol is not None:
            lnf._assert_matrix_responsiveness(track.R_matrix, matrix_responsiveness_tol,
                                              only_4d=(method == "4d"))
        return track.R_matrix, (track.R_matrices_ebe if element_by_element else None)

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

        A backward twiss is given the W matrix after the last element, so the one the
        rows propagate from is the solution of ``R_matrix W_matrix_start = W_matrix_end``.

        Returns the orbit columns, the W matrix per row, the first and one-past-last element index
        the table names its rows from, and the ``extra_data`` the regular twiss puts its tracking data in
        (always empty here). ``keep_initial_particles`` is ignored, there is nothing to keep.
        """
        _unsupported(keep_tracking_data=keep_tracking_data,
                     initial_particles=initial_particles, ebe_monitor=ebe_monitor)
        particle_on_co = init.particle_on_co
        if twiss_orientation == "forward":
            track = self._track(line, particle_on_co, _coordinates(particle_on_co),
                                start, end)
            W_matrix_start = init.W_matrix
        else:
            if spin:
                raise NotImplementedError(
                    "``spin`` with a backward twiss is not supported by the TPSA twiss")
            track = self._shoot_to_end(line, particle_on_co, start, end)
            W_matrix_start = np.linalg.solve(track.R_matrix, init.W_matrix)

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
        if spin:
            for i, name in enumerate(SPIN_COORDS):
                orbit[name] = track.spin[:, i].copy()

        W_matrices_ebe = track.R_matrices_ebe @ W_matrix_start   # (rows, 6, 6) @ (6, 6)
        return orbit, W_matrices_ebe, track.start, track.end + 1, {}

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
        R_matrix = track.R_matrix
        _, orbit_delta2_coefficient, orbit_shift_per_delta, dR_matrix_ddelta = (
            _off_momentum_expansion(track.hessians_ebe[-1], R_matrix))
        W_matrix, dW_matrix_ddelta, chromaticity = _4d_w_matrix_and_derivatives(
            R_matrix, dR_matrix_ddelta, 2.0 * orbit_delta2_coefficient)

        # (rows, 6, 6, 6) @ (6,) -> (rows, 6, 6), then @ (6, 6) -> (rows, 6, 6)
        dW_matrices_ebe_ddelta = (
            (track.hessians_ebe @ orbit_shift_per_delta) @ W_matrix
            + track.R_matrices_ebe @ dW_matrix_ddelta)     # (rows, 6, 6) @ (6, 6)
        W_matrices_ebe = track.R_matrices_ebe @ W_matrix   # (rows, 6, 6) @ (6, 6)
        columns = _chromatic_columns(W_matrices_ebe, dW_matrices_ebe_ddelta)
        return columns, {"dqx": chromaticity[0], "dqy": chromaticity[1]}

    def knob_derivatives(
        self, twiss_config: dict[str, Any], twiss_res: TwissTable
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """``<column>_dknob`` columns ``(rows, knobs)``, and ``q<plane>_dknob`` when periodic."""
        periodic = twiss_config["periodic"]
        _unsupported(reverse=twiss_config["reverse"], zero_at=twiss_config["zero_at"])
        if twiss_res._data["_orientation"] != "forward":
            raise NotImplementedError("``knobs`` with a backward twiss is not supported")
        if periodic and twiss_config["method"] != "4d":
            raise NotImplementedError("``knobs`` with a periodic twiss needs ``method='4d'``")

        init = twiss_config["init"]
        track = self._track(twiss_config["line"], init.particle_on_co,
                            _coordinates(init.particle_on_co),
                            twiss_config["start"], twiss_config["end"])
        W_matrices_ebe, dW_matrices_ebe, dorbit_ebe, tune_derivatives = _knob_directions(
            track, init.W_matrix, periodic)

        # the trailing axis broadcasts W against the knob axis of dW
        columns = {f"{name}_dknob": column for name, column in
                   _derivative_columns(W_matrices_ebe[..., None], dW_matrices_ebe).items()}
        for i, coord in enumerate(COORDS):
            columns[f"{coord}_dknob"] = dorbit_ebe[:, i, :]
        scalars = {"knob_names": list(self.knobs)}
        if periodic:
            scalars["qx_dknob"], scalars["qy_dknob"] = tune_derivatives
        return columns, scalars


def _off_momentum_expansion(
    one_turn_hessian: np.ndarray, R_matrix: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand the closed orbit and the one-turn matrix in delta.

    ``x_co(delta) = orbit_delta_coefficient delta + orbit_delta2_coefficient delta**2``
    about the on-momentum orbit, and the linear map about that orbit is
    ``R_matrix + delta * dR_matrix_ddelta``. Only the transverse block is made
    periodic, as in a 4d twiss.
    """
    transverse_R_minus_identity = R_matrix[:4, :4] - np.eye(4)
    orbit_delta_coefficient = -np.linalg.solve(transverse_R_minus_identity,
                                               R_matrix[:4, _DELTA])
    orbit_shift_per_delta = np.zeros(6)
    orbit_shift_per_delta[:4] = orbit_delta_coefficient
    orbit_shift_per_delta[_DELTA] = 1.0
    dR_matrix_ddelta = one_turn_hessian @ orbit_shift_per_delta   # (6, 6, 6) @ (6,) -> (6, 6)
    # delta**2 term of the periodicity condition, 0.5 * H[shift, shift]
    quadratic = 0.5 * dR_matrix_ddelta @ orbit_shift_per_delta     # (6, 6) @ (6,) -> (6,)
    orbit_delta2_coefficient = -np.linalg.solve(transverse_R_minus_identity, quadratic[:4])
    return (orbit_delta_coefficient, orbit_delta2_coefficient,
            orbit_shift_per_delta, dR_matrix_ddelta)


def _eigen_with_dummy_block(R_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagonalize R with ``lnf``'s dummy longitudinal rotation.

    Returns the eigenvalues, the eigenvectors and the mask of the two dummy modes.
    """
    eigenvalues, eigenvectors = np.linalg.eig(lnf._with_dummy_longitudinal_block(R_matrix))
    # A transverse tune on the dummy tune mixes the eigenvectors of both blocks
    dummy = np.all(np.abs(eigenvectors[:4, :]) < 1e-12, axis=0)
    if dummy.sum() != 2:
        raise ValueError("a transverse tune coincides with the dummy longitudinal tune")
    return eigenvalues, eigenvectors, dummy


def _project_transverse(
    dR_matrix: np.ndarray, eigenvectors: np.ndarray, dummy: np.ndarray
) -> np.ndarray:
    """Express the transverse block of ``dR_matrix`` in the eigenbasis, ``V^-1 dR V``.

    The dummy block does not move, so its rows and columns are zero.
    """
    transverse_block = np.zeros((6, 6))
    transverse_block[:4, :4] = dR_matrix[:4, :4]
    projected = np.linalg.solve(eigenvectors, transverse_block @ eigenvectors)
    projected[dummy, :] = projected[:, dummy] = 0.0
    return projected


def _4d_w_matrix_and_derivatives(
    R_matrix: np.ndarray, dR_matrix: np.ndarray,
    dispersion_derivative: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Build the 4d W matrix of ``lnf`` and the derivatives of W and of the tunes along ``dR_matrix``.

    ``dispersion_derivative`` defaults to the derivative of ``-(R4 - I)^-1 R[:4, delta]``
    along ``dR_matrix``, right for a knob. Along delta the orbit's delta**2 term adds to it,
    so the chromatic caller passes its own.
    First-order eigenvector perturbation, so it needs distinct eigenvalues: two planes
    on the same tune make the derivative blow up.
    """
    eigenvalues, eigenvectors, dummy = _eigen_with_dummy_block(R_matrix)
    projected = _project_transverse(dR_matrix, eigenvectors, dummy)
    gaps = eigenvalues[None, :] - eigenvalues[:, None]   # gaps[l, k] = lam_k - lam_l
    np.fill_diagonal(gaps, np.inf)             # own-mode component is a free scale
    transverse_gaps = np.abs(gaps[~dummy][:, ~dummy])
    if transverse_gaps.min() < 1e-9:
        raise ValueError("degenerate transverse tunes, the chromatic W is undefined")
    deigenvectors = eigenvectors @ np.divide(
        projected, gaps, out=np.zeros_like(projected), where=projected != 0)

    modes = lnf.sort_modes(eigenvectors, eigenvalues)
    W_matrix, dW_matrix = lnf._build_w_matrix_and_derivative_from_eigenvectors(
        eigenvectors, deigenvectors, modes, only_4d_block=True)
    tune_derivatives = [(projected[mode, mode] / eigenvalues[mode]).imag / (2 * np.pi)
                        for mode in modes[:2]]

    # the longitudinal columns hold the dispersion instead
    _set_4d_dispersion_columns(W_matrix, R_matrix)
    transverse_R_minus_identity = R_matrix[:4, :4] - np.eye(4)
    if dispersion_derivative is None:
        dispersion_derivative = -np.linalg.solve(
            transverse_R_minus_identity,
            dR_matrix[:4, _DELTA] + dR_matrix[:4, :4] @ W_matrix[:4, _DELTA])
    dW_matrix[4:, :] = dW_matrix[:, 4:] = 0
    dW_matrix[:4, _DELTA] = dispersion_derivative
    dW_matrix[:4, 4] = -np.linalg.solve(
        transverse_R_minus_identity,
        dR_matrix[:4, 4] + dR_matrix[:4, :4] @ W_matrix[:4, 4])
    return W_matrix, dW_matrix, tune_derivatives


def _chromatic_columns(
    W_matrices_ebe: np.ndarray, dW_matrices_ebe_ddelta: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute the chromatic functions from W and ``dW/ddelta`` at every element (MAD-8 6.3)."""
    columns = {}
    for plane, plane_index in (("x", 0), ("y", 2)):
        # entries of the plane's 2x2 block of W and dW/ddelta
        i, j = plane_index, plane_index + 1
        w00, w01 = W_matrices_ebe[:, i, i], W_matrices_ebe[:, i, j]
        w10, w11 = W_matrices_ebe[:, j, i], W_matrices_ebe[:, j, j]
        d00, d01 = dW_matrices_ebe_ddelta[:, i, i], dW_matrices_ebe_ddelta[:, i, j]
        d10, d11 = dW_matrices_ebe_ddelta[:, j, i], dW_matrices_ebe_ddelta[:, j, j]

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

    dzeta = W_matrices_ebe[:, 4, _DELTA]
    columns["dzeta"] = dzeta - dzeta[0]
    for i, name in enumerate(("ddx", "ddpx", "ddy", "ddpy")):
        columns[name] = dW_matrices_ebe_ddelta[:, i, _DELTA]   # d(dispersion)/ddelta
    return columns


def _closed_orbit_knob_shifts(track: TpsaEbeTrack) -> np.ndarray:
    """``(6, knobs)`` 4d closed-orbit derivative ``(I - R4)^-1 df/dknob``."""
    orbit_shifts = np.zeros((6, len(track.knobs)))
    orbit_shifts[:4] = np.linalg.solve(np.eye(4) - track.R_matrix[:4, :4],
                                       track.dorbit_dknob_ebe[-1, :4])
    return orbit_shifts


def _knob_directions(
    track: TpsaEbeTrack, W_matrix: np.ndarray, periodic: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """W, ``dW/dknob`` and ``dorbit/dknob`` at every row, and ``dq/dknob`` when periodic.

    Open, the start orbit and W are fixed. Periodic (4d), the start orbit moves with the
    closed orbit and W is rebuilt from the one-turn matrix, so ``W_matrix`` is unused.
    """
    R_matrices_ebe = track.R_matrices_ebe
    dR_matrices_ebe = track.dR_dknob_ebe      # (rows, 6, 6, knobs)
    dorbit_ebe = track.dorbit_dknob_ebe
    if not periodic:
        return (R_matrices_ebe @ W_matrix,
                np.einsum("rijk,jl->rilk", dR_matrices_ebe, W_matrix), dorbit_ebe, None)
    num_knobs = len(track.knobs)
    orbit_shifts = _closed_orbit_knob_shifts(track)
    dR_matrices_ebe = dR_matrices_ebe + np.einsum(
        "rijl,lk->rijk", track.hessians_ebe, orbit_shifts)
    dW_matrix = np.zeros((6, 6, num_knobs))
    tune_derivatives = np.zeros((2, num_knobs))
    for k in range(num_knobs):
        W_matrix, dW_matrix[..., k], tune_derivatives[:, k] = (
            _4d_w_matrix_and_derivatives(track.R_matrix, dR_matrices_ebe[-1, ..., k]))
    dW_matrices_ebe = (np.einsum("rijk,jl->rilk", dR_matrices_ebe, W_matrix)
                       + np.einsum("rij,jlk->rilk", R_matrices_ebe, dW_matrix))
    dorbit_ebe = dorbit_ebe + np.einsum("rij,jk->rik", R_matrices_ebe, orbit_shifts)
    return R_matrices_ebe @ W_matrix, dW_matrices_ebe, dorbit_ebe, tune_derivatives


def _derivative_columns(
    W_matrices_ebe: np.ndarray, dW_matrices_ebe: np.ndarray
) -> dict[str, np.ndarray]:
    """Differentiate the lattice functions of ``_get_lattice_functions`` along ``dW``."""
    W, dW = W_matrices_ebe, dW_matrices_ebe
    columns = {}
    for plane, i in (("x", 0), ("y", 2)):
        j = i + 1
        w00, w01, w10, w11 = W[:, i, i], W[:, i, j], W[:, j, i], W[:, j, j]
        d00, d01, d10, d11 = dW[:, i, i], dW[:, i, j], dW[:, j, i], dW[:, j, j]
        bet = w00**2 + w01**2
        columns[f"bet{plane}"] = 2 * (w00 * d00 + w01 * d01)
        columns[f"alf{plane}"] = -(d00 * w10 + w00 * d10 + d01 * w11 + w01 * d11)
        # the table subtracts the phase at the first row
        dphase = (w00 * d01 - w01 * d00) / bet / (2 * np.pi)
        columns[f"mu{plane}"] = dphase - dphase[0]

    # dispersion: the vector of span(W[:, 4], W[:, 5]) with zeta 0 and pzeta 1
    ratio = W[:, 4, 5] / W[:, 4, 4]
    dratio = (dW[:, 4, 5] * W[:, 4, 4] - W[:, 4, 5] * dW[:, 4, 4]) / W[:, 4, 4]**2
    denominator = W[:, 5, 5] - W[:, 5, 4] * ratio
    ddenominator = dW[:, 5, 5] - dW[:, 5, 4] * ratio - W[:, 5, 4] * dratio
    for i, name in enumerate(("dx", "dpx", "dy", "dpy")):
        numerator = W[:, i, 5] - W[:, i, 4] * ratio
        dnumerator = dW[:, i, 5] - dW[:, i, 4] * ratio - W[:, i, 4] * dratio
        columns[name] = (dnumerator * denominator - numerator * ddenominator) / denominator**2
    return columns
