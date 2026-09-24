"""ActionTpsaTrack: a native-GTPSA match action (sibling of ActionTwissMadngTPSA).

One parametric ``line.track`` per merit evaluation yields both the target values and the
analytic Jacobian d(target)/d(knob), read off the coefficients recorded at the targets.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import madng_tpsa
import xtrack as xt

from ..match import Action, ActionTwiss, TargetRelPhaseAdvance
from ..twiss.lattice_functions_from_W import _get_lattice_functions
from ..twiss.twiss_init import _6d_w_matrix
from ._knobs import KnobParameters, _scalar_value
from .particles import COORDS, ParticlesTpsa
from .twiss import _derivative_columns, _linear_and_knob_blocks, _recorded_monomials

_OPTICS_QTYS = ("betx", "bety", "alfx", "alfy", "dx", "dpx", "dy", "dpy")
_ORBIT_QTYS = ("x", "px", "y", "py", "zeta", "delta")
_PHASE_QTYS = ("mux", "muy")
_PHASE_ERROR = ("phase-advance targets need the continuous phase, "
                "use tpsa_backend='twiss'")


class ActionTpsaTrack(Action):
    """Match action for the GTPSA backend.

    ``vary_names`` are held as parametric maps (``KnobParameters``). The dependencies are
    resolved by propagating through xdeps into the element strengths. The map is tracked once through
    the range, recording A and dA/dknob at the target locations. Values and gradients come from the
    same functions as the TPSA twiss.

    Phase targets are rejected: the map only gives the fractional phase (``atan2``).

    The element fields hold TPSA handles for the whole match. Call ``teardown()`` to put
    plain doubles back.
    """

    def __init__(self, line, vary_names, targets=(), tw_kwargs=None, order=2,
                 **kwargs):
        self.line = line
        self.vary_names = list(vary_names)
        self.targets = list(targets)
        self.order = order
        # The value-only map carries no parameters and can be one order less
        # as derivatives are not required.
        self.plain_order = max(1, order - 1)
        self.tw_kwargs = dict(tw_kwargs or {})
        self.tw_kwargs.update(kwargs)
        self.optics_target_locations = None   # all observed locations (ordered, unique)
        self._col_names = None                # result-table columns to fill
        self._target_meta = None              # per-target (qty, loc)
        self._already_prepared = False
        self._init = None
        self._seed_name = None
        self._knobs = None
        self._last_res = None
        # A merit evaluation only needs the values,
        # knob columns are needed at Jacobian points. The solver says which through the _build_parametric flag.
        self._build_parametric = True
        self._last_parametric = None
        self._plain_descriptor = None

    def set_build_parametric(self, flag):
        """Solver hint: parametric map (Jacobian point) or value-only (line search)."""
        self._build_parametric = bool(flag)

    def prepare(self, force=False):
        if self._already_prepared and not force:
            return

        init = self.tw_kwargs.get("init", None)
        if init is None:
            tw_kw = {k: v for k, v in self.tw_kwargs.items() if k != "init"}
            init = self.line.twiss(**tw_kw)
        self._init = init
        self._seed_name = self.tw_kwargs.get("start", None) or init.name[0]

        # Classify targets, collect observed locations + result columns.
        locs, cols, meta = [], set(), []

        def add_loc(loc):
            if loc not in locs:
                locs.append(loc)

        for target in self.targets:
            if isinstance(target, TargetRelPhaseAdvance):
                raise ValueError(_PHASE_ERROR)
            elif isinstance(target.tar, tuple):
                qty, loc = target.tar
                if qty in _PHASE_QTYS:
                    raise ValueError(_PHASE_ERROR)
                if qty not in _OPTICS_QTYS and qty not in _ORBIT_QTYS:
                    raise ValueError(f"target quantity {qty!r} not supported")
                add_loc(loc)
                cols.add(qty)
                meta.append((qty, loc))
            else:
                raise NotImplementedError(f"unsupported target {target!r}")

        self.optics_target_locations = locs
        self._col_names = cols
        self._target_meta = meta

        # Resolve the tracked range. For a ring cut at the segment end (e.g. lhcb1 cut
        # at ip1), ``end`` resolves to an index at/before ``start``: the segment is the
        # tail of the line, so track to the physical end and read wrap-around locations
        # (index <= start) at the line's last element (same physical point).
        # Without start and end the range is the whole ring: the seed is the cut point,
        # reached again at the line end, so a target there means the tracked value.
        names = list(self.line.element_names)
        end = self.tw_kwargs.get("end", None)
        self._start_idx = names.index(self._seed_name)
        end_idx = names.index(end) if end is not None else len(names)
        self._wrap = end_idx <= self._start_idx
        # ele_stop is exclusive, and a monitor records at the entry of an element the track reaches
        self._track_stop = None if self._wrap or end_idx + 1 >= len(names) else end_idx + 1
        full_ring = self.tw_kwargs.get("start", None) is None and end is None

        def observed_at(loc):
            idx = names.index(loc)
            if self._wrap and idx < self._start_idx:
                return names[-1]
            if full_ring and idx <= self._start_idx:
                return names[-1]
            return loc

        self._obs_name = {loc: observed_at(loc)
                          for loc in self.optics_target_locations}

        # One descriptor for the whole match: the line variables, the element fields the
        # expressions reach, the tracked map and the recorded maps all live in it.
        # Parameter k+1 is vary_names[k], so the gradients come out in vary order.
        self._descriptor = madng_tpsa.Descriptor(
            6, self.order, params=list(self.vary_names), param_order=1)
        # The value-only map carries no parameters and only plain_order, which is what
        # makes it cheap.
        self._plain_descriptor = madng_tpsa.Descriptor(6, self.plain_order)
        self._blocks = _recorded_monomials(self._descriptor, hessians=False)
        self._plain_blocks = _recorded_monomials(self._plain_descriptor, hessians=False)
        self._knobs = KnobParameters(self.line, self.vary_names, self._descriptor)
        self._knobs.apply()
        self._already_prepared = True

    def teardown(self):
        """Put plain doubles back in the line variables and the fields they drive."""
        if self._knobs is not None:
            self._knobs.teardown()
            self._knobs = None
        self._already_prepared = False

    def _seed_map(self, parametric=True):
        """A fresh map seeded with the periodic orbit + W-matrix at the start."""
        init = self._init
        pref = self.line.particle_ref

        def at(qty):
            return float(init[qty, self._seed_name])

        m = ParticlesTpsa(
            order=self.order if parametric else self.plain_order,
            descriptor=self._descriptor if parametric else self._plain_descriptor,
            mass0=float(pref.mass0),
            q0=float(pref.q0),
            p0c=float(pref.p0c[0]),
            x=at("x"), px=at("px"), y=at("y"), py=at("py"),
            zeta=at("zeta"), delta=at("delta"),
        )
        m.set_jacobian(_6d_w_matrix(
            at("betx"), at("bety"), at("alfx"), at("alfy"), 1.0,
            at("dx"), at("dpx"), at("dy"), at("dpy"),
        ))
        return m

    def run(self):
        if not self._already_prepared:
            self.prepare()

        # Rewrite the driven fields every call: the optimizer also writes line.vars
        # directly (reload, clipping), so their current contents are unknown.
        # Parametric needs the knob parameters, value-only needs plain doubles.
        parametric = self._build_parametric
        values = [_scalar_value(self.line[n]) for n in self.vary_names]
        if parametric:
            self._knobs.apply(values)
        else:
            self._knobs.apply_doubles(values)

        blocks = self._blocks if parametric else self._plain_blocks
        try:
            # Unique physical positions (distinct logical locations may resolve to the same
            # element, e.g. 'ip1' and 'ip1.l1' after the ring-cut remap).
            obs = list(dict.fromkeys(self._obs_name[loc]
                                     for loc in self.optics_target_locations))
            self.line.track(
                self._seed_map(parametric),
                ele_start=self.tw_kwargs.get("start", 0),
                ele_stop=self._track_stop,
                multi_element_monitor_at=obs,
                monitor_monomials=np.vstack(list(blocks.values())),
            )
        finally:
            # xdeps reads the line variables after every action evaluation.
            self._knobs.teardown()
        monitor = self.line.tracker.record_multi_element_last_track
        self._last_parametric = parametric

        # monitor rows follow the line, the tables follow optics_target_locations
        rows = [monitor._obs_index(self._obs_name[loc]) for loc in self.optics_target_locations]
        W_matrices, dW_matrices, dorbit = _linear_and_knob_blocks(
            monitor.coefficients_by_coord(turn=0)[rows], blocks,
            len(self.vary_names) if parametric else 0)
        # strictly increasing s, so no thin-group merging
        lattice_functions, _ = _get_lattice_functions(
            W_matrices.copy(), False, np.arange(len(rows), dtype=float))
        for coord in COORDS:
            lattice_functions[coord] = monitor.get(coord, turn=0)[0][rows]
        if parametric:
            self._gradients = _derivative_columns(W_matrices[..., None], dW_matrices)
            for i, coord in enumerate(COORDS):
                self._gradients[coord] = dorbit[:, i, :]

        cols = {"name": np.array(self.optics_target_locations, dtype=object)}
        for c in self._col_names:
            cols[c] = lattice_functions[c]
        self._last_res = xt.TwissTable(data=cols)
        return self._last_res

    def acquire_jacobian(self):
        """(n_targets, n_vary) analytic d(target)/d(knob) from the last tracked map.
            If the last map was value-only, re-track to get a parametric one."""
        if not self._last_parametric:
            self._build_parametric = True    # last map was value-only: re-track
            self.run()
        row = {loc: i for i, loc in enumerate(self.optics_target_locations)}
        return np.array([self._gradients[qty][row[loc]] for qty, loc in self._target_meta])


class ActionTwissTpsa(ActionTwiss):
    """``ActionTwiss`` over ``line.twiss(tpsa=True, knobs=vary_names)``.

    The Jacobian is read off the ``<column>_dknob`` columns. Line-search evaluations run
    without knobs, at order 1.
    """

    def __init__(self, line, vary_names, targets, **kwargs):
        super().__init__(line, **kwargs)
        self.kwargs["tpsa"] = True
        self.vary_names = list(vary_names)
        self.targets = targets
        # evaluations before the solver's first hint only need values, acquire_jacobian re-runs
        self._build_parametric = False
        self._last_parametric = False
        self._last_res = None

    def set_build_parametric(self, flag):
        """Solver hint: Jacobian point (knob columns) or value-only evaluation."""
        self._build_parametric = bool(flag)

    def run(self, allow_failure=True):
        self.kwargs["knobs"] = self.vary_names if self._build_parametric else None
        out = super().run(allow_failure=allow_failure)
        self._last_res = out
        self._last_parametric = self._build_parametric and not isinstance(out, str)
        return out

    def acquire_jacobian(self):
        """``(n_targets, n_vary)`` d(target)/d(knob) of the last evaluation."""
        if not self._last_parametric:
            self._build_parametric = True
            self.run(allow_failure=False)
        return np.array([self._target_gradient(target) for target in self.targets])

    def _target_gradient(self, target):
        if target.action is not self:
            raise NotImplementedError("targets of other actions are not supported")
        if hasattr(target.value, "auxtarget") or target.optimize_log:
            raise NotImplementedError(
                "inequality and optimize_log targets are not supported by the TPSA twiss")
        res = self._last_res
        if isinstance(target, TargetRelPhaseAdvance):
            end = -1 if target.end == "__ele_stop__" else target.end
            start = 0 if target.start == "__ele_start__" else target.start
            column = f"{target.var}_dknob"
            return res[column, end] - res[column, start]
        name, at = target.tar if isinstance(target.tar, tuple) else (target.tar, None)
        if not isinstance(name, str):
            raise NotImplementedError("callable targets are not supported by the TPSA twiss")
        column = f"{name}_dknob"
        if column not in res._data:
            available = sorted(k[:-len("_dknob")] for k in res._data if k.endswith("_dknob"))
            raise KeyError(f"no knob derivative for {name!r}, available: {available}")
        if at is None:
            return np.asarray(res._data[column])
        return res[column, at]
