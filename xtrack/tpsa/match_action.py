"""ActionTpsaTrack: a native-GTPSA match action (sibling of ActionTwissMadngTPSA).

One parametric ``line.track`` per merit evaluation yields both the target values and the
analytic Jacobian d(target)/d(knob), read off the tracked map's optics.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import xgtpsa
import xtrack as xt

from ..match import Action, TargetRelPhaseAdvance
from ..twiss import _6d_w_matrix
from ._knobs import KnobParameters
from .particles import ParticlesTpsa

# Optics quantities served by TpsaOptics, orbit quantities by the map's param_jacobian.
_OPTICS_QTYS = ("betx", "bety", "alfx", "alfy", "mux", "muy", "dx", "dpx", "dy", "dpy")
_ORBIT_QTYS = ("x", "px", "y", "py", "zeta", "delta")
_PHASE_QTYS = ("mux", "muy")


class ActionTpsaTrack(Action):
    """Match action for the GTPSA backend.

    ``vary_names`` are held as parametric maps (``KnobParameters``). The dependencies are
    resolved propagated through xdeps into the element strengths. The map is tracked once through
    the range, recording the full map at every target location. Values come from
    ``TpsaOptics``, from which the Jacobian is read.

    Optics targets and phase-advance targets are both supported.
    Caveat for the phase-advances: the map Jacobian yields only the fractional betatron phase (``atan2``).
    The continuous phase advance is recovered by unwrapping against the reference twiss ``init``, which is
    exact whenever the phase stays within half a unit of the reference. Whenever this is not the case,
    the offset is an integer.
    The knob gradient is exact regardless (the integer offset is knob-independent).

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
        self._target_meta = None              # per-target ('located'|'phase', ...)
        self._already_prepared = False
        self._init = None
        self._seed_name = None
        self._knobs = None
        self._monitor = None
        self._last_res = None
        # A merit evaluation only needs the values,
        # knob columns are needed at Jacobian points. The solver says which through the _build_parametric flag.
        self._build_parametric = True
        self._last_parametric = None
        self._plain_descriptor = None

    def set_build_parametric(self, flag):
        """Solver hint: parametric map (Jacobian point) or value-only (line search)."""
        self._build_parametric = bool(flag)

    # ------------------------------------------------------------------ prepare
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
        has_phase = False

        def add_loc(loc):
            if loc not in locs:
                locs.append(loc)

        for target in self.targets:
            if isinstance(target, TargetRelPhaseAdvance):
                has_phase = True
                var = target.var
                if var not in _PHASE_QTYS:
                    raise ValueError(f"phase-advance quantity {var!r} not supported")
                start = self._seed_name if target.start == "__ele_start__" else target.start
                end = (self.tw_kwargs.get("end") or init.name[-1]) \
                    if target.end == "__ele_stop__" else target.end
                add_loc(start)
                add_loc(end)
                cols.add(var)
                meta.append(("phase", var, start, end))
            elif isinstance(target.tar, tuple):
                qty, loc = target.tar
                if qty not in _OPTICS_QTYS and qty not in _ORBIT_QTYS:
                    raise ValueError(f"target quantity {qty!r} not supported")
                add_loc(loc)
                cols.add(qty)
                meta.append(("located", qty, loc))
            else:
                raise NotImplementedError(f"unsupported target {target!r}")

        if has_phase or (cols & set(_PHASE_QTYS)):
            # Any mux/muy column is unwrapped against the seed phase (phi0), also when
            # it comes from a plain located target rather than a phase-advance one.
            add_loc(self._seed_name)
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
        self._track_stop = None if self._wrap else end
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
        # Parameter k+1 is vary_names[k], which is what makes TpsaOptics.gradient() come
        # out in vary order.
        self._descriptor = xgtpsa.Descriptor(
            6, self.order, params=list(self.vary_names), param_order=1)
        # The value-only map carries no parameters and only plain_order, which is what
        # makes it cheap.
        self._plain_descriptor = xgtpsa.Descriptor(6, self.plain_order)
        self._knobs = KnobParameters(self.line, self.vary_names, self._descriptor)
        self._knobs.apply()
        self._already_prepared = True

    def teardown(self):
        """Put plain doubles back in the line variables and the fields they drive."""
        if self._knobs is not None:
            self._knobs.teardown()
            self._knobs = None
        self._already_prepared = False

    # ------------------------------------------------------------------- seed
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

    # -------------------------------------------------------------------- run
    def run(self):
        if not self._already_prepared:
            self.prepare()

        # Rewrite the driven fields every call: the optimizer also writes line.vars
        # directly (reload, clipping), so their current contents are unknown.
        # Parametric needs the knob parameters, value-only needs plain doubles.
        parametric = self._build_parametric
        values = [float(self.line[n]) for n in self.vary_names]
        if parametric:
            self._knobs.refresh(values)
        else:
            self._knobs.apply_doubles(values)

        m = self._seed_map(parametric)
        # Unique physical positions (distinct logical locations may resolve to the same
        # element, e.g. 'ip1' and 'ip1.l1' after the ring-cut remap).
        obs = list(dict.fromkeys(self._obs_name[loc]
                                 for loc in self.optics_target_locations))
        self.line.track(
            m,
            ele_start=self.tw_kwargs.get("start", 0),
            ele_stop=self._track_stop,
            multi_element_monitor_at=obs,
        )
        self._monitor = self.line.tracker.record_multi_element_last_track
        self._last_parametric = parametric

        # One map view + one TpsaOptics per location, reused for all quantities and the Jacobian.
        self._views = {loc: self._monitor.map_at(self._obs_name[loc])
                       for loc in self.optics_target_locations}
        self._optics = {loc: v.optics() for loc, v in self._views.items()}
        self._phase_cont = self._continuous_phases()

        # Result table: fill every requested column at every observed location.
        cols: dict[str, Any] = {"name": np.array(self.optics_target_locations, dtype=object)}
        for c in self._col_names:
            cols[c] = np.array([self._value(loc, c) for loc in self.optics_target_locations])
        res = xt.TwissTable(data=cols)
        self._last_res = res
        return res

    def _continuous_phases(self):
        """mux/muy per observed location, unwrapped vs the reference twiss (phase from
        the seed). ``cont[var][loc]`` compares as ``cont[end] - cont[start]``."""
        cont = {}
        for var in (self._col_names & set(_PHASE_QTYS)):
            phi0 = getattr(self._optics[self._seed_name], var)     # seed map phase
            mu_ref0 = float(self._init[var, self._seed_name])
            d = {}
            for loc in self.optics_target_locations:
                frac_adv = getattr(self._optics[loc], var) - phi0   # from seed, fractional
                ref_adv = float(self._init[var, self._obs_name[loc]]) - mu_ref0
                d[loc] = frac_adv + round(ref_adv - frac_adv)        # snap integer to ref
            cont[var] = d
        return cont

    def _value(self, loc, qty):
        if qty in _PHASE_QTYS:
            return self._phase_cont[qty][loc]
        if qty in _ORBIT_QTYS:
            return self._views[loc].const_part[_ORBIT_QTYS.index(qty)]
        return getattr(self._optics[loc], qty)

    # -------------------------------------------------------------- jacobian
    def acquire_jacobian(self):
        """(n_targets, n_vary) analytic d(target)/d(knob) from the last tracked map.
            If the last map was value-only, re-track to get a parametric one."""
        if not self._last_parametric:
            self._build_parametric = True    # last map was value-only: re-track
            self.run()
        jac = np.zeros((len(self.targets), len(self.vary_names)))
        for i, meta in enumerate(self._target_meta):
            if meta[0] == "phase":
                _, var, start, end = meta
                # integer offset is knob-independent -> difference of fractional grads
                jac[i, :] = self._optics[end].gradient(var) - self._optics[start].gradient(var)
            else:
                _, qty, loc = meta
                if qty in _ORBIT_QTYS:
                    jac[i, :] = self._views[loc].param_jacobian()[_ORBIT_QTYS.index(qty)]
                else:
                    jac[i, :] = self._optics[loc].gradient(qty)
        return jac
