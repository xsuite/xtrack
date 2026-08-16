# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Generic (machine-independent) multi-bunch beam-beam tools.

Install coherent (rigid-bunch) 2D beam-beam elements
(:class:`xfields.BeamBeamBiGaussianMultibunch2D`) for the head-on and
long-range (LR) encounters at an arbitrary set of interaction points (IPs) of
two counter-rotating rings, and find the per-bunch self-consistent closed orbit
of the two multi-bunch beams by iterating the multi-bunch twiss.

Everything is driven through one entry point and a small state object:

    setup = env.xfields.install_multibunch_beambeam(
        clockwise_line, anticlockwise_line, ips=[...],
        num_long_range_encounters_per_side=..., harmonic_number=...,
        bunch_spacing_buckets=..., nemitt_x=..., nemitt_y=...,
        filling_clockwise=..., filling_anticlockwise=...)
    mbtw_cw, mbtw_acw = setup.solve()

``install_multibunch_beambeam`` places one beam-beam element per encounter
DIRECTLY on the two lines (the element is its own twiss/survey observation
point -- there are no separate markers), computes the encounter geometry
(per-encounter bunch-pairing offset, convolved sizes, survey separation) and
returns a :class:`MultibunchBBSetup`. All further operations are methods on that
object:

* :meth:`MultibunchBBSetup.solve` -- self-consistent per-bunch closed orbit;
* :meth:`MultibunchBBSetup.second_order_maps` -- a fast sector-map copy: the arcs
  between the encounters are replaced by second-order maps (splitting the lines
  at the beam-beam elements, which stay exact) and a NEW setup on the reduced
  lines is returned; solving it is orders of magnitude faster and gives the same
  per-bunch orbit and tunes;
* :meth:`MultibunchBBSetup.load_solution` -- load a converged solution (from a
  reduced-model solve) onto this setup's lattice, e.g. to compute footprints on
  the full thick lattice;
* :meth:`MultibunchBBSetup.set_filling` -- change the per-beam bunch filling.

Nothing is LHC specific: the IPs (a ``{ip: offset}`` mapping, or a list of IP
element names for which the head-on offsets are derived from the ring geometry
as ``round(2 * (s_ip - s_ref) / slot_len)``), the RF harmonic number and the
bunch spacing (in RF buckets) are all inputs. The number of slots on the ring is
``n_slots = harmonic_number / bunch_spacing_buckets`` and the longitudinal slot
spacing is ``slot_len = circumference / n_slots``.

The two lines are the usual xsuite two-ring setup: the ``clockwise_line`` runs
in ``+s`` and the ``anticlockwise_line`` is the *reversed* line (also running in
``+s``); a given encounter element name is the same physical point in both
beams, mirrored on the reversed line.
"""

import numpy as np

from .general import _print


def _resolve_line(env, line):
    """Accept a Line or a line name and return the Line."""
    import xtrack as xt
    if isinstance(line, xt.Line):
        return line
    return env[line]


def _encounter_specs(ip_names, nlr_per_side):
    """Yield ``(base_name, ip, signed_n)``; ``signed_n == 0`` is the head-on
    encounter. ``nlr_per_side`` is an int or a ``{ip: int}`` mapping."""
    for ip in ip_names:
        n_side = (nlr_per_side[ip] if isinstance(nlr_per_side, dict)
                  else nlr_per_side)
        yield f'bb_{ip}_ho', ip, 0
        for n in range(1, n_side + 1):
            yield f'bb_{ip}_r{n:02d}', ip, +n
            yield f'bb_{ip}_l{n:02d}', ip, -n


def _gamma0(line):
    return float(line.particle_ref.gamma0[0])


def _beta0(line):
    return float(line.particle_ref.beta0[0])


def _bind_beambeam_scale(line, bb_names):
    """Create a per-line ``beambeam_scale`` knob and bind the ``scale_strength``
    of all the beam-beam elements to it (as in the xfields beam-beam config
    tools), e.g. for footprints with a linear rescale of the beam-beam strength.
    """
    if 'beambeam_scale' not in line.vars:
        line['beambeam_scale'] = 1.0
    for name in bb_names:
        line[name].scale_strength = 'beambeam_scale'


class MultibunchBBSetup:
    """State and operations of one multi-bunch beam-beam problem.

    Returned by :func:`install_multibunch_beambeam` /
    :meth:`xtrack.environment.EnvXfields.install_multibunch_beambeam`. Holds the
    two lines, the encounter geometry (per-encounter pairing offset, beta
    functions and survey separation), the installed beam-beam elements
    (``bb_cw`` / ``bb_acw``, keyed by encounter base name) and the per-beam bunch
    filling. The self-consistent solve, the sector-map reduction and the
    solution transfer are methods (:meth:`solve`, :meth:`second_order_maps`,
    :meth:`load_solution`, :meth:`set_filling`).

    Beam-beam element names are the encounter base names plus the beam suffix
    (default ``'_cw'`` / ``'_acw'``), e.g. ``bb_ip1_ho_cw``. The element itself
    is the observation point used for the geometry and the orbit feedback.
    """

    def __init__(self, clockwise_line, anticlockwise_line, ips,
                 num_long_range_encounters_per_side,
                 harmonic_number, bunch_spacing_buckets,
                 nemitt_x, nemitt_y,
                 bb_suffix_cw='_cw', bb_suffix_acw='_acw'):
        self.cw_line = clockwise_line
        self.acw_line = anticlockwise_line
        self.ips = ips                          # dict {ip: offset} or list
        self.ip_names = list(ips)
        self.ip_offsets = None                  # resolved by _compute_geometry
        self.num_long_range_encounters_per_side = \
            num_long_range_encounters_per_side
        self.harmonic_number = int(harmonic_number)
        self.bunch_spacing_buckets = int(bunch_spacing_buckets)
        self.n_slots = int(harmonic_number) // int(bunch_spacing_buckets)
        self.slot_len = clockwise_line.get_length() / self.n_slots
        self.b_h_dist = self.slot_len / 2.0          # LR half-slot step [m]
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.bb_suffix_cw = bb_suffix_cw
        self.bb_suffix_acw = bb_suffix_acw

        self.enc_specs = list(_encounter_specs(
            self.ip_names, num_long_range_encounters_per_side))
        self.enc_names = [b for b, _, _ in self.enc_specs]
        self.bb_names_cw = [b + bb_suffix_cw for b in self.enc_names]
        self.bb_names_acw = [b + bb_suffix_acw for b in self.enc_names]

        self.geom = {}               # base_name -> geometry dict
        self.meta = {}
        self.bb_cw = {}              # base_name -> element (in cw line)
        self.bb_acw = {}             # base_name -> element (in acw line)
        # filling: per-slot populations (length n_slots) and derived quantities
        self.filling_cw = None
        self.filling_acw = None
        self.bunches_cw = None       # slot indices of populated cw bunches
        self.bunches_acw = None
        self.num_particles_cw = None  # populations of the populated cw bunches
        self.num_particles_acw = None

    # ------------------------------------------------------------------
    # Naming / bookkeeping
    # ------------------------------------------------------------------
    def bb_name(self, base, mirror):
        """Beam-beam element name of one beam (``mirror=True`` -> acw)."""
        return base + (self.bb_suffix_acw if mirror else self.bb_suffix_cw)

    @property
    def zeta_per_slot(self):
        return self.slot_len

    def bunch_zeta(self, mirror):
        """Longitudinal positions of the populated bunches of a beam."""
        slots = self.bunches_acw if mirror else self.bunches_cw
        return np.asarray(slots) * self.slot_len

    def __repr__(self):
        return (f'MultibunchBBSetup({len(self.enc_names)} encounters, '
                f'n_slots={self.n_slots}, '
                f'B1={0 if self.bunches_cw is None else len(self.bunches_cw)} '
                f'B2={0 if self.bunches_acw is None else len(self.bunches_acw)} '
                f'bunches)')

    def set_filling(self, filling_clockwise, filling_anticlockwise):
        """Set the per-beam bunch filling from two arrays of length ``n_slots``
        holding the population (number of particles, zero = empty) of each slot.
        Derives the populated slot indices and their intensities.

        If the beam-beam elements are already installed, a change in either
        bunch count rebuilds them with arrays sized exactly for the new filling.
        A change that preserves both counts updates the bunch grids and resets
        the opposing state in place."""
        old_n_cw = (None if self.bunches_cw is None
                    else len(self.bunches_cw))
        old_n_acw = (None if self.bunches_acw is None
                     else len(self.bunches_acw))
        fcw = np.asarray(filling_clockwise, dtype=float)
        facw = np.asarray(filling_anticlockwise, dtype=float)
        if len(fcw) != self.n_slots or len(facw) != self.n_slots:
            raise ValueError(
                f'filling arrays must have length n_slots={self.n_slots} '
                f'(got {len(fcw)} and {len(facw)})')
        bunches_cw = np.where(fcw > 0)[0]
        bunches_acw = np.where(facw > 0)[0]
        if len(bunches_cw) == 0 or len(bunches_acw) == 0:
            raise ValueError(
                'Rigid-bunch beam-beam requires at least one filled slot in '
                'each beam.')
        self.filling_cw = fcw
        self.filling_acw = facw
        self.bunches_cw = bunches_cw
        self.bunches_acw = bunches_acw
        self.num_particles_cw = fcw[self.bunches_cw]
        self.num_particles_acw = facw[self.bunches_acw]

        # Skipped during installation, before elements and geometry exist.
        if self.bb_cw and self.geom:
            count_changed = (old_n_cw != len(self.bunches_cw)
                             or old_n_acw != len(self.bunches_acw))
            if count_changed:
                self._rebuild_bb_elements()
            else:
                self._configure_bb()

    # ------------------------------------------------------------------
    # Building (used by install_multibunch_beambeam)
    # ------------------------------------------------------------------
    def _representative_other_beam(self, line, mirror):
        """One inactive-kick representative per opposing bunch.

        The particles remain active so their count determines exact storage,
        but zero weight preserves the bare-lattice cold start until a solution
        update loads the physical bunch populations.
        """
        import xtrack as xt
        other_line = self.cw_line if mirror else self.acw_line
        slots = self.bunches_cw if mirror else self.bunches_acw
        return xt.Particles(
            _context=line._context,
            p0c=other_line.particle_ref.p0c[0],
            mass0=other_line.particle_ref.mass0,
            q0=other_line.particle_ref.q0,
            x=np.zeros(len(slots)),
            y=np.zeros(len(slots)),
            zeta=np.asarray(slots) * self.slot_len,
            weight=np.zeros(len(slots)))

    def _make_bb(self, line, mirror):
        """Build one exactly sized rigid-bunch beam-beam element."""
        import xfields as xf
        beta0_other = _beta0(self.acw_line if not mirror else self.cw_line)
        q0_other = float((self.acw_line if not mirror
                          else self.cw_line).particle_ref.q0)
        own_slots = self.bunches_acw if mirror else self.bunches_cw
        own_zeta = np.asarray(own_slots) * self.slot_len
        return xf.BeamBeamBiGaussianMultibunch2D(
            other_particles=self._representative_other_beam(line, mirror),
            own_beam_zeta=own_zeta,
            zeta_offset=0.0,
            zeta_match_tol=0.1 * self.slot_len,
            zeta_period=self.n_slots * self.slot_len,
            other_beam_q0=q0_other, other_beam_beta0=beta0_other,
            coherent=True,
            sigma_x=1.0, sigma_y=1.0,
            other_beam_sigma_x=1.0, other_beam_sigma_y=1.0,
            _context=line._context)

    def _place_bb(self, line, mirror):
        """Place one (still un-sized) beam-beam element per encounter DIRECTLY
        at the encounter positions of ``line`` (no separate markers). The
        element is named ``bb_name(base, mirror)`` and is the observation point
        for the geometry. Sizes/offsets are set later by ``_configure_bb``.

        The own-beam bunch zeta grid (this line's bunches) is registered on each
        element so the kernel can match every tracked particle to its own bunch
        for the coherent convolution; the own per-bunch sizes are indexed by it.
        """
        env = line.env
        length = line.get_length()
        tab = line.get_table()
        s_ip = {ip: float(tab['s', ip]) for ip in self.ip_names}
        places, names = [], []
        for base, ip, sn in self.enc_specs:
            at = (s_ip[ip] + (-sn if mirror else sn) * self.b_h_dist + 1e-6) % length
            elname = self.bb_name(base, mirror)
            bb = self._make_bb(line, mirror)
            places.append(env.place(elname, bb, at=at))
            names.append((base, elname))
        line.insert(places)
        _bind_beambeam_scale(line, [elname for _, elname in names])
        return {base: line[elname] for base, elname in names}

    def _rebuild_bb_elements(self):
        """Replace installed elements after a change in either bunch count."""
        rebuilt = []
        for line, mirror in ((self.cw_line, False), (self.acw_line, True)):
            for base in self.enc_names:
                elname = self.bb_name(base, mirror)
                bb = self._make_bb(line, mirror)
                line.env.elements.remove(elname)
                line.env.elements[elname] = bb
            names = (self.bb_names_acw if mirror else self.bb_names_cw)
            _bind_beambeam_scale(line, names)
            rebuilt.append({base: line[self.bb_name(base, mirror)]
                            for base in self.enc_names})
        self.bb_cw, self.bb_acw = rebuilt
        self._configure_bb()

    def _resolve_ip_offsets(self, tw_cw):
        """Head-on pairing offset (in slots) of each IP: from ``self.ips`` if a
        mapping, else from the ring geometry (first IP as the reference),
        ``round(2 * (s_ip - s_ref) / slot_len)``."""
        if isinstance(self.ips, dict):
            return {ip: int(v) % self.n_slots for ip, v in self.ips.items()}
        ref = self.ip_names[0]
        s_ref = tw_cw['s', self.bb_name(f'bb_{ref}_ho', False)]
        return {ip: int(round(2 * (tw_cw['s', self.bb_name(f'bb_{ip}_ho',
                                                           False)] - s_ref)
                              / self.slot_len)) % self.n_slots
                for ip in self.ip_names}

    def _compute_geometry(self, survey_separation=True):
        """Twiss (and, if requested, survey) both beams and fill ``self.geom``
        with the per-encounter pairing offset, beta functions and survey
        separation; then configure the beam-beam element sizes/offsets. The
        beam-beam elements are the observation points (they must already be
        placed and are inactive, so the twiss is the bare optics)."""
        tw_cw = self.cw_line.twiss()
        tw_acw = self.acw_line.twiss()
        n_slots = self.n_slots
        self.ip_offsets = self._resolve_ip_offsets(tw_cw)

        if survey_separation:
            # One survey per IP, in that IP's own frame (:func:`_local_survey`).
            # The encounters of an IP are then read off directly: their
            # coordinates are already relative to their IP and expressed in the
            # local frame there, so nothing has to be subtracted afterwards and
            # the bends of the insertion are followed exactly.
            enc_of_ip = {ip: [b for b, i, _ in self.enc_specs if i == ip]
                         for ip in self.ip_names}
            frames_cw = {ip: _local_survey(
                self.cw_line, ip,
                [self.bb_name(b, False) for b in enc_of_ip[ip]])
                for ip in self.ip_names}
            frames_acw = {ip: _local_survey(
                self.acw_line, ip,
                [self.bb_name(b, True) for b in enc_of_ip[ip]])
                for ip in self.ip_names}

        geom = {}
        for j, (base, ip, sn) in enumerate(self.enc_specs):
            ncw, nacw = self.bb_names_cw[j], self.bb_names_acw[j]
            offset = (self.ip_offsets[ip] + sn) % n_slots
            sep_x = sep_y = 0.0
            if survey_separation:
                # Geometric survey separation of the two rings at this
                # encounter, in the clockwise beam's frame. Both positions are
                # already in their IP's local frame; the anticlockwise ring is
                # traversed the other way, so its frame is rotated 180 deg
                # about the vertical (X,Z -> -X,-Z) before differencing. The
                # separation is then PROJECTED on the local unit vectors of the
                # clockwise beam at this encounter, which gives the horizontal
                # and vertical components with their sign directly -- no
                # assumption that the reference direction follows the mean ring
                # azimuth, so a bend between the IP and the encounter is
                # accounted for.
                s1, frame = frames_cw[ip][ncw]
                s2, _ = frames_acw[ip][nacw]
                d = s1 - np.array([-s2[0], s2[1], -s2[2]])
                sep_x = float(d @ frame[:, 0])
                sep_y = float(d @ frame[:, 1])
            geom[base] = dict(
                ip=ip, offset=offset, signed_n=sn,
                betx_cw=float(tw_cw['betx', ncw]),
                bety_cw=float(tw_cw['bety', ncw]),
                betx_acw=float(tw_acw['betx', nacw]),
                bety_acw=float(tw_acw['bety', nacw]),
                sep_x=sep_x, sep_y=sep_y,
            )
        self.geom = geom
        self.meta = dict(
            qx_cw=float(tw_cw.qx), qy_cw=float(tw_cw.qy),
            qx_acw=float(tw_acw.qx), qy_acw=float(tw_acw.qy))
        self._configure_bb()

    def _configure_bb(self):
        """Set the pairing ``zeta_offset`` and the (static) opposing sizes on the
        placed beam-beam elements from the computed geometry, then register the
        own bunch grid and own sizes (:meth:`_register_own_sizes`). With the
        design (static) optics the beta functions are the same for all bunches, so
        the opposing per-bunch sizes (indexed by the OTHER beam) are filled with a
        single value broadcast over the opposing bunches."""
        ex, ey = self.nemitt_x, self.nemitt_y
        for mirror, bb_dict in ((False, self.bb_cw), (True, self.bb_acw)):
            oth = 'cw' if mirror else 'acw'
            n_oth = (len(self.bunches_cw) if mirror
                     else len(self.bunches_acw))
            gamma0 = _gamma0(self.cw_line if not mirror else self.acw_line)
            for base in self.enc_names:
                e = self.geom[base]
                bb = bb_dict[base]
                bb.zeta_offset = (-e['offset'] if mirror else e['offset']) \
                    * self.slot_len
                bb.other_beam_sigma_x = np.full(
                    n_oth, np.sqrt(e[f'betx_{oth}'] * ex / gamma0))
                bb.other_beam_sigma_y = np.full(
                    n_oth, np.sqrt(e[f'bety_{oth}'] * ey / gamma0))
        self._register_own_sizes()
        for line, mirror, bb_dict in (
                (self.cw_line, False, self.bb_cw),
                (self.acw_line, True, self.bb_acw)):
            particles = self._representative_other_beam(line, mirror)
            for bb in bb_dict.values():
                bb.update_from_other_beam(particles)

    def _register_own_sizes(self):
        """(Re)register each element's OWN bunch grid (``own_beam_zeta``) and
        static design sizes (``sigma_x``/``sigma_y``, indexed by THIS beam) for
        the CURRENT filling. Called at install and again whenever the filling
        changes (:meth:`set_filling`), so the per-bunch own arrays always match
        ``self.bunches_*``. Uses the bare-optics betas cached in ``self.geom``
        (uniform over bunches). A count change rebuilds the elements before this
        method is called, so their array lengths match the current filling."""
        ex, ey = self.nemitt_x, self.nemitt_y
        for mirror, bb_dict in ((False, self.bb_cw), (True, self.bb_acw)):
            own = 'acw' if mirror else 'cw'
            gamma0 = _gamma0(self.cw_line if not mirror else self.acw_line)
            own_slots = self.bunches_acw if mirror else self.bunches_cw
            own_zeta = np.asarray(own_slots) * self.slot_len
            for base in self.enc_names:
                e = self.geom[base]
                bb_dict[base].update_from_own_beam(
                    own_zeta,
                    sigma_x=np.sqrt(e[f'betx_{own}'] * ex / gamma0),
                    sigma_y=np.sqrt(e[f'bety_{own}'] * ey / gamma0))

    # ------------------------------------------------------------------
    # Sector-map reduction
    # ------------------------------------------------------------------
    def second_order_maps(self, keep_extra_cw=None, keep_extra_acw=None,
                          context=None):
        """Return a NEW :class:`MultibunchBBSetup` on second-order-map copies of
        the two lines: the arcs between the encounters are replaced by
        second-order maps (the beam-beam elements, kept as split points, stay
        exact), so solving the returned setup is much faster and gives the same
        per-bunch orbit and tunes. This setup (the full lattice) is left
        untouched; transfer a converged reduced solution back with
        :meth:`load_solution`.

        ``keep_extra_cw`` / ``keep_extra_acw`` are extra element names to
        preserve exactly (e.g. lattice octupoles for amplitude-detuning
        studies). ``context`` selects the CPU context for the reduced trackers
        (default: the clockwise line's context).
        """
        if context is None:
            context = self.cw_line._context
        method = self.cw_line.twiss_default.get('method', '4d')
        split_cw = self.bb_names_cw + list(keep_extra_cw or [])
        split_acw = self.bb_names_acw + list(keep_extra_acw or [])
        red_cw = self.cw_line.get_line_with_second_order_maps(split_at=split_cw)
        red_acw = self.acw_line.get_line_with_second_order_maps(
            split_at=split_acw)
        for rl in (red_cw, red_acw):
            rl.twiss_default['method'] = method
            rl.build_tracker(_context=context)

        new = MultibunchBBSetup(
            red_cw, red_acw, self.ips,
            self.num_long_range_encounters_per_side, self.harmonic_number,
            self.bunch_spacing_buckets, self.nemitt_x, self.nemitt_y,
            bb_suffix_cw=self.bb_suffix_cw, bb_suffix_acw=self.bb_suffix_acw)
        new.geom = self.geom
        new.meta = self.meta
        new.ip_offsets = self.ip_offsets
        new.filling_cw = self.filling_cw
        new.filling_acw = self.filling_acw
        new.bunches_cw = self.bunches_cw
        new.bunches_acw = self.bunches_acw
        new.num_particles_cw = self.num_particles_cw
        new.num_particles_acw = self.num_particles_acw
        new.bb_cw = {b: red_cw[new.bb_name(b, False)] for b in new.enc_names}
        new.bb_acw = {b: red_acw[new.bb_name(b, True)] for b in new.enc_names}
        # the reduced lines have their own env: re-create the beambeam_scale knob
        _bind_beambeam_scale(red_cw, new.bb_names_cw)
        _bind_beambeam_scale(red_acw, new.bb_names_acw)
        return new

    # ------------------------------------------------------------------
    # Solve / solution feed-in
    # ------------------------------------------------------------------
    def _compute_sigmas(self, mbtw, bb_names, gamma0):
        """Per-encounter transverse sizes from LIVE per-bunch beta functions
        (dynamic beta). Returns (sigma_x, sigma_y), each (n_bunches, n_enc)."""
        sigma_x = np.sqrt(mbtw['betx', bb_names] * self.nemitt_x / gamma0)
        sigma_y = np.sqrt(mbtw['bety', bb_names] * self.nemitt_y / gamma0)
        return sigma_x, sigma_y

    def _sigma_vector(self, bb_dict):
        """Own-beam per-bunch sizes laid out to match :func:`_orbit_vector` (x
        then y, each the ``(n_bunches, n_enc)`` array raveled). :meth:`set_filling`
        keeps every element's own arrays in sync with the solved bunches, so the
        active per-bunch sizes (the first ``num_own_bunches`` entries, in the same
        bunch order as the twiss) stack directly, one column per encounter"""
        def active(bb):
            n = int(bb.num_own_bunches)
            return np.asarray(bb.sigma_x)[:n], np.asarray(bb.sigma_y)[:n]
        cols = [active(bb_dict[b]) for b in self.enc_names]
        sx = np.stack([c[0] for c in cols], axis=1)   # (n_bunches, n_enc)
        sy = np.stack([c[1] for c in cols], axis=1)
        return np.concatenate([sx.ravel(), sy.ravel()])

    def _update_opposing(self, bb_dict, mbtw_other, slots_other, bb_names_other,
                         num_particles_other, sigmas_other=None,
                         sigmas_own=None):
        """Write the opposing beam's per-bunch orbit (+ survey separation) into
        the beam-beam elements ``bb_dict`` (optionally also the dynamic-beta
        sizes). Between the two (opposite-parity) beam lines x flips and y does
        not; matching TRAIN/pytrain and the reversed-line x-flip, the survey
        separation enters as ``-sep_x`` in x for BOTH beams.

        The opposing sizes (``sigmas_other``) are indexed by the OTHER beam; the
        own sizes (``sigmas_own``) are indexed by THIS beam."""
        import xtrack as xt
        xs = -mbtw_other['x', bb_names_other]
        ys = mbtw_other['y', bb_names_other]
        zeta_other = np.asarray(slots_other) * self.slot_len
        ref = self.cw_line.particle_ref
        p = xt.Particles(
            p0c=ref.p0c[0], mass0=ref.mass0, q0=ref.q0,
            x=np.zeros(len(zeta_other)), y=np.zeros(len(zeta_other)),
            zeta=zeta_other, weight=num_particles_other)
        for j, base in enumerate(self.enc_names):
            p.x[:] = xs[:, j] - self.geom[base]['sep_x']
            p.y[:] = ys[:, j] - self.geom[base]['sep_y']
            kw = {}
            if sigmas_other is not None:
                kw = dict(other_beam_sigma_x=sigmas_other[0][:, j],
                          other_beam_sigma_y=sigmas_other[1][:, j])
            bb = bb_dict[base]
            bb.update_from_other_beam(p, **kw)
            if sigmas_own is not None:
                bb.update_from_own_beam(sigma_x=sigmas_own[0][:, j],
                                        sigma_y=sigmas_own[1][:, j])

    def load_solution(self, mbtw_clockwise, mbtw_anticlockwise,
                      dynamic_beta=False):
        """Load a converged per-bunch solution (e.g. from a reduced-model
        :meth:`solve`) into this setup's beam-beam elements, so a subsequent
        ``line.twiss_multibunch(...)`` / footprint on this setup's lattice
        reproduces it. ``mbtw_clockwise`` / ``mbtw_anticlockwise`` are the two
        beams' multi-bunch twiss (their orbits are read at the beam-beam
        elements). With ``dynamic_beta`` the per-bunch sizes are taken from the
        live beta functions of the solution."""
        sizes_cw = sizes_acw = None
        if dynamic_beta:
            sizes_cw = self._compute_sigmas(mbtw_clockwise, self.bb_names_cw,
                                            _gamma0(self.cw_line))
            sizes_acw = self._compute_sigmas(mbtw_anticlockwise,
                                             self.bb_names_acw,
                                             _gamma0(self.acw_line))
        self._update_opposing(self.bb_cw, mbtw_anticlockwise, self.bunches_acw,
                              self.bb_names_acw, self.num_particles_acw,
                              sigmas_other=sizes_acw, sigmas_own=sizes_cw)
        self._update_opposing(self.bb_acw, mbtw_clockwise, self.bunches_cw,
                              self.bb_names_cw, self.num_particles_cw,
                              sigmas_other=sizes_cw, sigmas_own=sizes_acw)

    def solve(self, max_iterations=5, tol_sigma=1e-4, dynamic_beta=False,
              method='4d', chrom=False, twiss_mode=None, show_progress=True,
              continue_on_closed_orbit_error=False):
        """Find the per-bunch self-consistent closed orbit: iterate the
        multi-bunch twiss on both beams, feeding each beam's per-bunch closed
        orbit (plus the survey separation) into the other beam's elements, until
        the closed orbit at every beam-beam element stops changing.

        The elements are left holding the converged opposing-beam state, so a
        subsequent ``line.twiss_multibunch(...)`` (or plain ``line.twiss()`` for
        one bunch) reproduces the solution without re-iterating.

        Parameters
        ----------
        max_iterations : int
            Maximum number of iterations (default 5).
        tol_sigma : float
            Convergence tolerance, in units of the local beam size: stop once
            the maximum change of the x/y closed orbit at all beam-beam elements
            (over all bunches of both beams) between two successive iterations,
            each normalised by that element's own-beam transverse size, is below
            this (default 1e-4).
        dynamic_beta : bool
            If True, recompute the per-bunch effective (convolved) sizes from the
            live per-bunch beta functions at each iteration. Forces the
            optics-carrying twiss. Default False.
        method : str
            Twiss method, ``'4d'`` (default) or ``'6d'``.
        chrom : bool
            Whether to compute chromatic properties in the multi-bunch twiss.
        twiss_mode : str, optional
            ``'fast_orbit'`` (orbit only, the default when ``dynamic_beta`` is
            False), ``'fast'`` (adds per-bunch optics, forced when
            ``dynamic_beta`` is True) or ``'full'``.
        show_progress : bool
            Print per-iteration convergence information (default True).
        continue_on_closed_orbit_error : bool
            If True, the closed-orbit search of the INTERMEDIATE iterations may
            return its last iterate instead of raising
            :class:`ClosedOrbitSearchError` (same meaning as in
            :meth:`Line.twiss`); a FINAL iteration is then always run without
            it, and it must converge. That last orbit is the one returned and
            the one left on the elements, so the result is exactly as strict as
            without this option -- only the intermediate rounds are relaxed.
            Use it on lattices where the search cannot reach ``co_tol`` from a
            cold start but does once the opposing beam has settled. Default
            False (every iteration strict).

        Returns
        -------
        tuple of xtrack.MultiBunchTwiss
            ``(mbtw_clockwise, mbtw_anticlockwise)``.
        """
        if self.bunches_cw is None or self.bunches_acw is None:
            raise RuntimeError('bunch filling not set; call set_filling first')
        if twiss_mode is None:
            twiss_mode = 'fast' if dynamic_beta else 'fast_orbit'
        if dynamic_beta and twiss_mode == 'fast_orbit':
            twiss_mode = 'fast'

        cw, acw = self.cw_line, self.acw_line
        zeta_cw = np.asarray(self.bunches_cw) * self.slot_len
        zeta_acw = np.asarray(self.bunches_acw) * self.slot_len

        # The intermediate rounds may keep going on a closed-orbit error: their
        # orbit is only an input to the next round, so a few bunches short of
        # `co_tol` cost nothing, and on some machines the search cannot reach
        # it from a cold start at all. The FINAL round below is always strict.
        co_kwargs = (dict(continue_on_closed_orbit_error=True)
                     if continue_on_closed_orbit_error else {})

        mbtw_cw = mbtw_acw = None
        prev = None
        err = np.inf
        for it in range(max_iterations):
            mbtw_cw = cw.twiss_multibunch(
                zeta_bunches=zeta_cw, method=method, chrom=chrom,
                mode=twiss_mode, show_progress=show_progress, **co_kwargs)
            mbtw_acw = acw.twiss_multibunch(
                zeta_bunches=zeta_acw, method=method, chrom=chrom,
                mode=twiss_mode, show_progress=show_progress, **co_kwargs)

            cur = np.concatenate([_orbit_vector(mbtw_cw, self.bb_names_cw),
                                  _orbit_vector(mbtw_acw, self.bb_names_acw)])
            sig = np.concatenate([self._sigma_vector(self.bb_cw),
                                  self._sigma_vector(self.bb_acw)])
            err = (np.inf if prev is None
                   else float(np.max(np.abs(cur - prev) / sig)))
            prev = cur

            self.load_solution(mbtw_cw, mbtw_acw, dynamic_beta=dynamic_beta)

            if show_progress:
                _print(f'  multibunch orbit iteration {it}: '
                       f'max orbit change = {err:.2e} sigma')
            if err < tol_sigma:
                if show_progress:
                    _print(f'  converged after {it + 1} iterations '
                           f'(< {tol_sigma:.1e} sigma)')
                break
        else:
            if show_progress:
                _print(f'  reached max_iterations={max_iterations} '
                       f'(last change {err:.2e} sigma)')

        if continue_on_closed_orbit_error:
            # One last pass that must converge, so what is returned -- and what
            # the elements are left holding -- is a genuine closed orbit of the
            # converged state. If this raises, the solution is NOT usable and
            # the caller must know.
            if show_progress:
                _print('  final closed-orbit pass (strict)')
            mbtw_cw = cw.twiss_multibunch(
                zeta_bunches=zeta_cw, method=method, chrom=chrom,
                mode=twiss_mode, show_progress=show_progress)
            mbtw_acw = acw.twiss_multibunch(
                zeta_bunches=zeta_acw, method=method, chrom=chrom,
                mode=twiss_mode, show_progress=show_progress)
            self.load_solution(mbtw_cw, mbtw_acw, dynamic_beta=dynamic_beta)

        return mbtw_cw, mbtw_acw


def _local_survey(line, ref_name, names):
    """Survey of the region AROUND ``ref_name``, in that element's own frame:
    the reference sits at the origin with the identity orientation and the
    survey is integrated outwards from it, forwards and backwards, over just
    enough of the ring to reach ``names``.

    This is what makes the geometry independent of where the line is cut. A
    survey of the line as stored integrates from the line's first element, so
    an encounter on the far side of the seam from its IP -- which is exactly
    what happens once the line is cycled at a beam-beam IP and the upstream
    encounters wrap past s = 0 -- would be reached the long way round and pick
    up the whole ring closure defect (0.374 m for the LHC sequence as loaded,
    whose rbend lengths stay chords because ``option, -rbarc`` is not applied).
    Here the element sequence is first ROLLED so that the reference sits in the
    middle, then cut down to the span actually needed, so the seam is never
    crossed, no closure enters, and the reference frame is the local one: any
    bend between the reference and an element -- the separation and
    recombination dipoles of the insertion, in particular -- is followed
    exactly instead of being approximated by the mean ring azimuth.

    Returns ``{name: (v, W)}``: the position of each requested element in the
    reference frame, and its orientation matrix there. The columns of ``W`` are
    the local unit vectors -- ``W[:, 0]`` the horizontal transverse direction
    (the beam's ``x``), ``W[:, 1]`` the vertical, ``W[:, 2]`` the direction of
    travel.
    """
    from .survey import get_survey

    tab = line.get_table(attr=True)
    elements = list(line._elements)
    n_el = len(elements)
    length = np.array(tab.length[:n_el], dtype=float)
    length[~np.array(tab.isthick[:n_el], dtype=bool)] = 0.
    angle = np.array(tab.angle[:n_el], dtype=float)
    tilt = np.array(tab.rot_s_rad[:n_el], dtype=float)

    index = {nn: i for i, nn in enumerate(line.element_names)}
    shift = (index[ref_name] - n_el // 2) % n_el   # reference to the middle
    rolled = {nn: (index[nn] - shift) % n_el for nn in list(names) + [ref_name]}
    lo, hi = min(rolled.values()), max(rolled.values())

    def window(arr):
        return np.concatenate([arr[shift:], arr[:shift]])[lo:hi + 1]

    order = elements[shift:] + elements[:shift]
    V, W = get_survey(
        elements=order[lo:hi + 1],
        X0=0., Y0=0., Z0=0., theta0=0., phi0=0., psi0=0.,
        drift_length=window(length), angle=window(angle), tilt=window(tilt),
        element0=rolled[ref_name] - lo)

    return {nn: (np.array(V[rolled[nn] - lo]), np.array(W[rolled[nn] - lo]))
            for nn in names}


def _orbit_vector(mbtw, bb_names):
    """Flat (x then y) per-bunch orbit at all elements, for convergence."""
    x = mbtw['x', bb_names]
    y = mbtw['y', bb_names]
    return np.concatenate([np.asarray(x).ravel(), np.asarray(y).ravel()])


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def install_multibunch_beambeam(env, clockwise_line, anticlockwise_line,
                                ips,
                                num_long_range_encounters_per_side,
                                harmonic_number, bunch_spacing_buckets,
                                nemitt_x, nemitt_y,
                                filling_clockwise, filling_anticlockwise,
                                survey_separation=True,
                                bb_suffix_cw='_cw', bb_suffix_acw='_acw'):
    """Install multi-bunch beam-beam elements at N IPs of two counter-rotating
    rings and compute the encounter geometry. See
    :meth:`xtrack.environment.EnvXfields.install_multibunch_beambeam` for the
    full documentation. Returns a :class:`MultibunchBBSetup`."""
    cw = _resolve_line(env, clockwise_line)
    acw = _resolve_line(env, anticlockwise_line)

    setup = MultibunchBBSetup(
        cw, acw, ips, num_long_range_encounters_per_side,
        harmonic_number, bunch_spacing_buckets, nemitt_x, nemitt_y,
        bb_suffix_cw=bb_suffix_cw, bb_suffix_acw=bb_suffix_acw)
    setup.set_filling(filling_clockwise, filling_anticlockwise)
    setup.bb_cw = setup._place_bb(cw, mirror=False)
    setup.bb_acw = setup._place_bb(acw, mirror=True)
    setup._compute_geometry(survey_separation=survey_separation)
    return setup
