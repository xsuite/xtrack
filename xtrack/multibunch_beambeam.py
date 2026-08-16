# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""
Generic (machine-independent) multi-bunch beam-beam tools.

Install coherent (rigid-bunch) 2D beam-beam elements
(:class:`xfields.BeamBeamBiGaussianRigidBunch2D`) for the head-on and
long-range (LR) encounters at an arbitrary set of interaction points (IPs) of
two counter-rotating rings, and find the per-bunch self-consistent closed orbit
of the two multi-bunch beams by iterating the multi-bunch twiss.

The public workflow uses the standard beam-beam install/configure entry points
and returns a small state object:

    env.xfields.install_beambeam_interactions(
        clockwise_line, anticlockwise_line, ip_names=[...],
        num_long_range_encounters_per_side=..., harmonic_number=...,
        bunch_spacing_buckets=..., mode='rigid_bunch')
    setup = env.xfields.configure_beambeam_interactions(
        nemitt_x=..., nemitt_y=...,
        filling_scheme_cw=..., filling_scheme_acw=...,
        bunch_intensity_particles_cw=...,
        bunch_intensity_particles_acw=...)
    mbtw_cw, mbtw_acw = setup.solve()

Installation places one beam-beam element per encounter DIRECTLY on the two
lines (the element is its own twiss/survey observation point -- there are no
separate markers), with arrays covering every RF slot. Configuration loads the
filling and populations, computes the encounter geometry (per-encounter
bunch-pairing offset, convolved sizes, survey separation) and returns a
:class:`RigidBunchBBSetup`. All further operations are methods on that object:

* :meth:`RigidBunchBBSetup.solve` -- self-consistent per-bunch closed orbit;
* :meth:`RigidBunchBBSetup.second_order_maps` -- a fast sector-map copy: the arcs
  between the encounters are replaced by second-order maps (splitting the lines
  at the beam-beam elements, which stay exact) and a NEW setup on the reduced
  lines is returned; solving it is orders of magnitude faster and gives the same
  per-bunch orbit and tunes;
* :meth:`RigidBunchBBSetup.load_solution` -- load a converged solution (from a
  reduced-model solve) onto this setup's lattice, e.g. to compute footprints on
  the full thick lattice;
* :meth:`RigidBunchBBSetup.set_filling` -- change the per-beam bunch filling.

Nothing is LHC specific: the IPs (a ``{ip: offset}`` mapping, or a list of IP
element names for which the head-on offsets are derived from the ring geometry
as ``round(2 * (s_ip - s_ref) / bunch_spacing_zeta)``), the RF harmonic number
and the bunch spacing (in RF buckets) are all inputs. The number of slots is
``n_slots = harmonic_number / bunch_spacing_buckets`` and the positive physical
slot spacing is ``bunch_spacing_zeta = circumference / n_slots``. Physical slot
``i`` is centred at ``zeta = -i * bunch_spacing_zeta``, consistently with the
Xpart, Xwakes and :class:`BeamStatsMonitor` bunch-pattern APIs.

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


def _encounter_specs(encounters):
    """Yield ``(base_name, ip, signed_n)``; ``signed_n == 0`` is the head-on
    encounter.

    Encounter enumeration is shared with the conventional beam-beam installer;
    only the rigid-bunch element names are rendered here.
    """
    for encounter in encounters.itertuples(index=False):
        ip = encounter.ip_name
        signed_n = encounter.identifier
        if encounter.encounter_type == 'head_on':
            base_name = f'bb_{ip}_ho'
        else:
            side = 'r' if signed_n > 0 else 'l'
            base_name = f'bb_{ip}_{side}{abs(signed_n):02d}'
        yield base_name, ip, signed_n


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


def _normalize_filling(filling_scheme, bunch_intensity_particles, n_slots,
                       beam_name):
    """Normalize one beam's occupancy and return compact filled-bunch data."""
    scheme = np.asarray(filling_scheme)
    if scheme.ndim != 1 or len(scheme) != n_slots:
        raise ValueError(
            f'`filling_scheme_{beam_name}` must be a one-dimensional array '
            f'of length n_slots={n_slots}.')
    if not np.all(np.isfinite(scheme)):
        raise ValueError(f'`filling_scheme_{beam_name}` must be finite.')
    scheme = (scheme != 0).astype(np.int64)
    filled_slots = np.nonzero(scheme)[0].astype(np.int64)
    if len(filled_slots) == 0:
        raise ValueError(
            f'`filling_scheme_{beam_name}` must contain at least one filled '
            'slot.')

    intensity = np.asarray(bunch_intensity_particles, dtype=float)
    if intensity.ndim == 0:
        intensity = np.full(len(filled_slots), float(intensity))
    elif intensity.ndim == 1 and len(intensity) == n_slots:
        intensity = intensity[filled_slots]
    else:
        raise ValueError(
            f'`bunch_intensity_particles_{beam_name}` must be a scalar or a '
            f'one-dimensional slot-indexed array of length n_slots={n_slots}.')
    if not np.all(np.isfinite(intensity)) or np.any(intensity <= 0):
        raise ValueError(
            f'`bunch_intensity_particles_{beam_name}` must be finite and '
            'strictly positive at every filled slot.')
    return scheme, filled_slots, intensity


class RigidBunchBBSetup:
    """State and operations of one rigid-bunch beam-beam problem.

    Returned by
    :meth:`xtrack.environment.EnvXfields.configure_beambeam_interactions` in
    rigid-bunch mode (and temporarily by the compatibility entry point
    :meth:`xtrack.environment.EnvXfields.install_multibunch_beambeam`). Holds
    the two lines, the encounter geometry (per-encounter pairing offset, beta
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
                 nemitt_x=None, nemitt_y=None,
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
        self.bunch_spacing_zeta = clockwise_line.get_length() / self.n_slots
        self.b_h_dist = self.bunch_spacing_zeta / 2.0
        self.nemitt_x = nemitt_x
        self.nemitt_y = nemitt_y
        self.bb_suffix_cw = bb_suffix_cw
        self.bb_suffix_acw = bb_suffix_acw

        import xfields as xf
        self.encounter_table = xf.generate_beambeam_encounter_table(
            self.ip_names, num_long_range_encounters_per_side,
            bunch_spacing_zeta=self.bunch_spacing_zeta)
        self.enc_specs = list(_encounter_specs(self.encounter_table))
        self.enc_names = [b for b, _, _ in self.enc_specs]
        self.bb_names_cw = [b + bb_suffix_cw for b in self.enc_names]
        self.bb_names_acw = [b + bb_suffix_acw for b in self.enc_names]

        self.geom = {}               # base_name -> geometry dict
        self.meta = {}
        self.bb_cw = {}              # base_name -> element (in cw line)
        self.bb_acw = {}             # base_name -> element (in acw line)
        # Occupancy stays slot-indexed; intensities are compact arrays aligned
        # with the corresponding physical filled-slot arrays.
        self.filling_scheme_cw = None
        self.filling_scheme_acw = None
        self.filled_slots_cw = None
        self.filled_slots_acw = None
        self.bunch_intensity_particles_cw = None
        self.bunch_intensity_particles_acw = None

    # ------------------------------------------------------------------
    # Naming / bookkeeping
    # ------------------------------------------------------------------
    def bb_name(self, base, mirror):
        """Beam-beam element name of one beam (``mirror=True`` -> acw)."""
        return base + (self.bb_suffix_acw if mirror else self.bb_suffix_cw)

    def bunch_zeta(self, mirror):
        """Bunch centres in ascending physical-slot order."""
        slots = self.filled_slots_acw if mirror else self.filled_slots_cw
        return -np.asarray(slots) * self.bunch_spacing_zeta

    def __repr__(self):
        n_cw = 0 if self.filled_slots_cw is None else len(self.filled_slots_cw)
        n_acw = (0 if self.filled_slots_acw is None
                 else len(self.filled_slots_acw))
        return (f'RigidBunchBBSetup({len(self.enc_names)} encounters, '
                f'n_slots={self.n_slots}, B1={n_cw} B2={n_acw} bunches)')

    def set_filling(self, filling_scheme_cw, filling_scheme_acw,
                    bunch_intensity_particles_cw,
                    bunch_intensity_particles_acw):
        """Set the two occupancy patterns and corresponding bunch intensities.

        Each filling scheme is a slot-indexed occupancy array of length
        ``n_slots``. Each intensity is either a scalar, applied uniformly to
        all filled slots, or a slot-indexed array of the same length. Derived
        physical slot identifiers are exposed as ``filled_slots_cw`` and
        ``filled_slots_acw``.

        The installed elements have one entry per RF slot, so any filling
        change updates their slot-indexed data in place without reallocating
        the Xobjects."""
        normalized_cw = _normalize_filling(
            filling_scheme_cw, bunch_intensity_particles_cw,
            self.n_slots, 'cw')
        normalized_acw = _normalize_filling(
            filling_scheme_acw, bunch_intensity_particles_acw,
            self.n_slots, 'acw')
        (self.filling_scheme_cw, self.filled_slots_cw,
         self.bunch_intensity_particles_cw) = normalized_cw
        (self.filling_scheme_acw, self.filled_slots_acw,
         self.bunch_intensity_particles_acw) = normalized_acw

        # Skipped before both elements and geometry exist. Once configured,
        # filling changes reset the slot-indexed opposing state in place.
        if self.bb_cw and self.geom:
            self._configure_bb()

    # ------------------------------------------------------------------
    # Building (used by install_multibunch_beambeam)
    # ------------------------------------------------------------------
    def _representative_other_beam(self, line, mirror):
        """One inactive-kick representative per opposing RF slot.

        All representatives remain active so their count allocates full-slot
        storage, while zero weight preserves the bare-lattice cold start until
        a solution update loads the physical bunch populations.
        """
        import xtrack as xt
        other_line = self.cw_line if mirror else self.acw_line
        slots = np.arange(self.n_slots)
        return xt.Particles(
            _context=line._context,
            p0c=other_line.particle_ref.p0c[0],
            mass0=other_line.particle_ref.mass0,
            q0=other_line.particle_ref.q0,
            x=np.zeros(len(slots)),
            y=np.zeros(len(slots)),
            zeta=-np.asarray(slots) * self.bunch_spacing_zeta,
            weight=np.zeros(len(slots)))

    def _make_bb(self, line, mirror):
        """Build one rigid-bunch element with one entry per RF slot."""
        import xfields as xf
        beta0_other = _beta0(self.acw_line if not mirror else self.cw_line)
        q0_other = float((self.acw_line if not mirror
                          else self.cw_line).particle_ref.q0)
        own_zeta = -np.arange(self.n_slots) * self.bunch_spacing_zeta
        return xf.BeamBeamBiGaussianRigidBunch2D(
            other_particles=self._representative_other_beam(line, mirror),
            own_beam_zeta=own_zeta,
            zeta_offset=0.0,
            zeta_match_tol=0.1 * self.bunch_spacing_zeta,
            zeta_period=self.n_slots * self.bunch_spacing_zeta,
            other_beam_q0=q0_other, other_beam_beta0=beta0_other,
            coherent=True,
            sigma_x=1.0, sigma_y=1.0,
            other_beam_sigma_x=1.0, other_beam_sigma_y=1.0,
            _context=line._context)

    def _place_bb(self, line, mirror):
        """Place one full-slot beam-beam element per encounter DIRECTLY
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
        position_column = 's_from_ip_acw' if mirror else 's_from_ip_cw'
        for (base, ip, _), displacement in zip(
                self.enc_specs, self.encounter_table[position_column]):
            at = (s_ip[ip] + displacement + 1e-6) % length
            elname = self.bb_name(base, mirror)
            bb = self._make_bb(line, mirror)
            places.append(env.place(elname, bb, at=at))
            names.append((base, elname))
        line.insert(places)
        _bind_beambeam_scale(line, [elname for _, elname in names])
        return {base: line[elname] for base, elname in names}

    def _resolve_ip_offsets(self, tw_cw):
        """Head-on pairing offset (in slots) of each IP: from ``self.ips`` if a
        mapping, else from the ring geometry (first IP as the reference),
        ``round(2 * (s_ip - s_ref) / bunch_spacing_zeta)``."""
        if isinstance(self.ips, dict):
            return {ip: int(v) % self.n_slots for ip, v in self.ips.items()}
        ref = self.ip_names[0]
        s_ref = tw_cw['s', self.bb_name(f'bb_{ref}_ho', False)]
        return {ip: int(round(2 * (tw_cw['s', self.bb_name(f'bb_{ip}_ho',
                                                           False)] - s_ref)
                              / self.bunch_spacing_zeta)) % self.n_slots
                for ip in self.ip_names}

    def _compute_geometry(self, survey_separation=True):
        """Fill ``self.geom`` from the shared Xfields geometry description.

        The beam-beam elements are the observation points. They must already
        be placed and inactive, so the shared Twiss and covariance calculation
        sees the bare optics.
        """
        import xfields as xf

        geometry_table, twisses = xf.compute_beambeam_geometry(
            encounter_table=self.encounter_table,
            line_cw=self.cw_line, line_acw=self.acw_line,
            element_names_cw=self.bb_names_cw,
            element_names_acw=self.bb_names_acw,
            nemitt_x=self.nemitt_x, nemitt_y=self.nemitt_y,
            survey_separation=survey_separation)
        tw_cw = twisses['cw']
        tw_acw = twisses['acw']
        n_slots = self.n_slots
        self.ip_offsets = self._resolve_ip_offsets(tw_cw)
        self.encounter_table = xf.generate_beambeam_encounter_table(
            self.ip_names, self.num_long_range_encounters_per_side,
            bunch_spacing_zeta=self.bunch_spacing_zeta,
            delay_at_ips_slots=self.ip_offsets, n_slots=n_slots)

        geom = {}
        for j, (base, ip, sn) in enumerate(self.enc_specs):
            offset = int(self.encounter_table['delay_in_slots_cw'].iloc[j]) \
                % n_slots
            row = geometry_table.iloc[j]
            geom[base] = dict(
                ip=ip, offset=offset, signed_n=sn,
                betx_cw=row['betx_cw'], bety_cw=row['bety_cw'],
                betx_acw=row['betx_acw'], bety_acw=row['bety_acw'],
                sigma_x_cw=np.sqrt(row['Sigma_11_cw']),
                sigma_y_cw=np.sqrt(row['Sigma_33_cw']),
                sigma_x_acw=np.sqrt(row['Sigma_11_acw']),
                sigma_y_acw=np.sqrt(row['Sigma_33_acw']),
                sep_x=row['separation_x'], sep_y=row['separation_y'],
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
        design (static) optics the covariance-derived sizes are the same for all
        bunches, so the opposing per-bunch sizes (indexed by the OTHER beam) are
        filled with a single value broadcast over the opposing bunches."""
        for mirror, bb_dict in ((False, self.bb_cw), (True, self.bb_acw)):
            oth = 'cw' if mirror else 'acw'
            for base in self.enc_names:
                e = self.geom[base]
                bb = bb_dict[base]
                # Public bunch centres use zeta = -slot * spacing. For the CW
                # beam, an opposing slot ``own + offset`` is therefore at
                # ``zeta_own - offset * spacing``; ACW uses the inverse map.
                bb.zeta_offset = (e['offset'] if mirror else -e['offset']) \
                    * self.bunch_spacing_zeta
                bb.other_beam_sigma_x = np.full(
                    self.n_slots, e[f'sigma_x_{oth}'])
                bb.other_beam_sigma_y = np.full(
                    self.n_slots, e[f'sigma_y_{oth}'])
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
        every RF slot. Uses the covariance-derived bare-optics sizes cached in
        ``self.geom`` (uniform over slots)."""
        for mirror, bb_dict in ((False, self.bb_cw), (True, self.bb_acw)):
            own = 'acw' if mirror else 'cw'
            own_zeta = -np.arange(self.n_slots) * self.bunch_spacing_zeta
            for base in self.enc_names:
                e = self.geom[base]
                bb_dict[base].update_from_own_beam(
                    own_zeta,
                    sigma_x=e[f'sigma_x_{own}'],
                    sigma_y=e[f'sigma_y_{own}'])

    # ------------------------------------------------------------------
    # Sector-map reduction
    # ------------------------------------------------------------------
    def second_order_maps(self, keep_extra_cw=None, keep_extra_acw=None,
                          context=None):
        """Return a NEW :class:`RigidBunchBBSetup` on second-order-map copies of
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

        new = RigidBunchBBSetup(
            red_cw, red_acw, self.ips,
            self.num_long_range_encounters_per_side, self.harmonic_number,
            self.bunch_spacing_buckets, self.nemitt_x, self.nemitt_y,
            bb_suffix_cw=self.bb_suffix_cw, bb_suffix_acw=self.bb_suffix_acw)
        new.geom = self.geom
        new.meta = self.meta
        new.ip_offsets = self.ip_offsets
        new.filling_scheme_cw = self.filling_scheme_cw
        new.filling_scheme_acw = self.filling_scheme_acw
        new.filled_slots_cw = self.filled_slots_cw
        new.filled_slots_acw = self.filled_slots_acw
        new.bunch_intensity_particles_cw = self.bunch_intensity_particles_cw
        new.bunch_intensity_particles_acw = self.bunch_intensity_particles_acw
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

    def _sigma_vector(self, bb_dict, mirror):
        """Own-beam per-bunch sizes laid out to match :func:`_orbit_vector` (x
        then y, each the ``(n_bunches, n_enc)`` array raveled). :meth:`set_filling`
        keeps every element's own arrays in sync with all RF slots. The element
        stores increasing zeta, while Twiss follows increasing physical slot
        number (decreasing zeta), so the filled slots are selected and mapped
        back to public order before stacking."""
        zeta = self.bunch_zeta(mirror)

        def active(bb):
            n = int(bb.num_own_bunches)
            stored_zeta = np.asarray(bb.own_beam_zeta)[:n]
            indices = np.searchsorted(stored_zeta, zeta)
            return (np.asarray(bb.sigma_x)[:n][indices],
                    np.asarray(bb.sigma_y)[:n][indices])
        cols = [active(bb_dict[b]) for b in self.enc_names]
        sx = np.stack([c[0] for c in cols], axis=1)   # (n_bunches, n_enc)
        sy = np.stack([c[1] for c in cols], axis=1)
        return np.concatenate([sx.ravel(), sy.ravel()])

    def _update_opposing(self, bb_dict, mbtw_other, slots_other, slots_own,
                         bb_names_other, num_particles_other, sigmas_other=None,
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
        slots_other = np.asarray(slots_other, dtype=np.int64)
        slots_own = np.asarray(slots_own, dtype=np.int64)
        all_slots = np.arange(self.n_slots)
        zeta_all = -all_slots * self.bunch_spacing_zeta
        weight_all = np.zeros(self.n_slots)
        weight_all[slots_other] = num_particles_other
        ref = self.cw_line.particle_ref
        p = xt.Particles(
            p0c=ref.p0c[0], mass0=ref.mass0, q0=ref.q0,
            x=np.zeros(self.n_slots), y=np.zeros(self.n_slots),
            zeta=zeta_all, weight=weight_all)
        other = 'acw' if bb_dict is self.bb_cw else 'cw'
        own = 'cw' if bb_dict is self.bb_cw else 'acw'
        for j, base in enumerate(self.enc_names):
            x_all = np.zeros(self.n_slots)
            y_all = np.zeros(self.n_slots)
            x_all[slots_other] = xs[:, j] - self.geom[base]['sep_x']
            y_all[slots_other] = ys[:, j] - self.geom[base]['sep_y']
            p.x[:] = x_all
            p.y[:] = y_all
            kw = {}
            if sigmas_other is not None:
                sigma_x_other = np.full(
                    self.n_slots, self.geom[base][f'sigma_x_{other}'])
                sigma_y_other = np.full(
                    self.n_slots, self.geom[base][f'sigma_y_{other}'])
                sigma_x_other[slots_other] = sigmas_other[0][:, j]
                sigma_y_other[slots_other] = sigmas_other[1][:, j]
                kw = dict(other_beam_sigma_x=sigma_x_other,
                          other_beam_sigma_y=sigma_y_other)
            bb = bb_dict[base]
            bb.update_from_other_beam(p, **kw)
            if sigmas_own is not None:
                sigma_x_own = np.full(
                    self.n_slots, self.geom[base][f'sigma_x_{own}'])
                sigma_y_own = np.full(
                    self.n_slots, self.geom[base][f'sigma_y_{own}'])
                sigma_x_own[slots_own] = sigmas_own[0][:, j]
                sigma_y_own[slots_own] = sigmas_own[1][:, j]
                bb.update_from_own_beam(
                    zeta=zeta_all,
                    sigma_x=sigma_x_own,
                    sigma_y=sigma_y_own)

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
        self._update_opposing(
            self.bb_cw, mbtw_anticlockwise,
            self.filled_slots_acw, self.filled_slots_cw,
            self.bb_names_acw, self.bunch_intensity_particles_acw,
            sigmas_other=sizes_acw, sigmas_own=sizes_cw)
        self._update_opposing(
            self.bb_acw, mbtw_clockwise,
            self.filled_slots_cw, self.filled_slots_acw,
            self.bb_names_cw, self.bunch_intensity_particles_cw,
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
        if self.filled_slots_cw is None or self.filled_slots_acw is None:
            raise RuntimeError('bunch filling not set; call set_filling first')
        if twiss_mode is None:
            twiss_mode = 'fast' if dynamic_beta else 'fast_orbit'
        if dynamic_beta and twiss_mode == 'fast_orbit':
            twiss_mode = 'fast'

        cw, acw = self.cw_line, self.acw_line
        zeta_cw = self.bunch_zeta(mirror=False)
        zeta_acw = self.bunch_zeta(mirror=True)

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
            sig = np.concatenate([self._sigma_vector(self.bb_cw, mirror=False),
                                  self._sigma_vector(self.bb_acw, mirror=True)])
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


def _orbit_vector(mbtw, bb_names):
    """Flat (x then y) per-bunch orbit at all elements, for convergence."""
    x = mbtw['x', bb_names]
    y = mbtw['y', bb_names]
    return np.concatenate([np.asarray(x).ravel(), np.asarray(y).ravel()])


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------
def install_rigid_bunch_beambeam(
        env, clockwise_line, anticlockwise_line, ip_names,
        num_long_range_encounters_per_side, harmonic_number,
        bunch_spacing_buckets, delay_at_ips_slots=None,
        survey_separation=True, bb_suffix_cw='_cw', bb_suffix_acw='_acw'):
    """Install full-slot rigid-bunch elements and store their configuration."""
    if harmonic_number is None or bunch_spacing_buckets is None:
        raise ValueError(
            '`harmonic_number` and `bunch_spacing_buckets` are required '
            "when mode='rigid_bunch'.")
    if clockwise_line is None or anticlockwise_line is None:
        raise ValueError("mode='rigid_bunch' requires both beam lines.")

    def line_name(line, argument_name):
        if isinstance(line, str):
            if line not in env.lines:
                raise KeyError(f'Line `{line}` not found in environment.')
            return line
        matches = [name for name, candidate in env.lines.items()
                   if candidate is line]
        if len(matches) != 1:
            raise ValueError(
                f'`{argument_name}` must be a line in this environment or '
                'its name.')
        return matches[0]

    ip_names = list(ip_names)
    if delay_at_ips_slots is None:
        ips = ip_names
    elif isinstance(delay_at_ips_slots, dict):
        ips = {ip: int(delay_at_ips_slots[ip]) for ip in ip_names}
    else:
        delays = list(delay_at_ips_slots)
        if len(delays) != len(ip_names):
            raise ValueError('`delay_at_ips_slots` must have one entry per IP.')
        ips = {ip: int(delay) for ip, delay in zip(ip_names, delays)}

    if isinstance(num_long_range_encounters_per_side, dict):
        num_lr_values = [num_long_range_encounters_per_side[ip]
                         for ip in ip_names]
        num_lr_is_scalar = False
    elif np.ndim(num_long_range_encounters_per_side) == 0:
        num_lr_values = [num_long_range_encounters_per_side]
        num_lr_is_scalar = True
    else:
        num_lr_values = list(num_long_range_encounters_per_side)
        num_lr_is_scalar = False
        if len(num_lr_values) != len(ip_names):
            raise ValueError(
                '`num_long_range_encounters_per_side` must have one entry '
                'per IP.')
    normalized_num_lr = []
    for value in num_lr_values:
        normalized = int(value)
        if normalized != value or normalized < 0:
            raise ValueError(
                '`num_long_range_encounters_per_side` entries must be '
                'non-negative integers.')
        normalized_num_lr.append(normalized)
    if num_lr_is_scalar:
        num_lr = normalized_num_lr[0]
    else:
        num_lr = dict(zip(ip_names, normalized_num_lr))

    cw_name = line_name(clockwise_line, 'clockwise_line')
    acw_name = line_name(anticlockwise_line, 'anticlockwise_line')
    env._bb_config = {
        'mode': 'rigid_bunch',
        # Retain the conventional serialization shape alongside the
        # rigid-bunch installation description.
        'dataframes': {'clockwise': None, 'anticlockwise': None},
        'clockwise_line': cw_name,
        'anticlockwise_line': acw_name,
        'ip_names': ip_names,
        'ips': ips,
        'num_long_range_encounters_per_side': num_lr,
        'harmonic_number': int(harmonic_number),
        'bunch_spacing_buckets': int(bunch_spacing_buckets),
        'survey_separation': bool(survey_separation),
        'bb_suffix_cw': bb_suffix_cw,
        'bb_suffix_acw': bb_suffix_acw,
    }
    setup = RigidBunchBBSetup(
        env[cw_name], env[acw_name], ips, num_lr,
        harmonic_number, bunch_spacing_buckets,
        bb_suffix_cw=bb_suffix_cw, bb_suffix_acw=bb_suffix_acw)
    setup.bb_cw = setup._place_bb(setup.cw_line, mirror=False)
    setup.bb_acw = setup._place_bb(setup.acw_line, mirror=True)
    env._rigid_bunch_bb_setup = setup


def configure_rigid_bunch_beambeam(
        env, nemitt_x, nemitt_y, filling_scheme_cw, filling_scheme_acw,
        bunch_intensity_particles_cw, bunch_intensity_particles_acw):
    """Populate installed rigid-bunch elements and return their setup."""
    config = env._bb_config
    if config.get('mode') != 'rigid_bunch':
        raise RuntimeError(
            'Install beam-beam interactions with `mode="rigid_bunch"` first.')
    setup = getattr(env, '_rigid_bunch_bb_setup', None)
    if setup is None:
        cw = env[config['clockwise_line']]
        acw = env[config['anticlockwise_line']]
        setup = RigidBunchBBSetup(
            cw, acw, config['ips'],
            config['num_long_range_encounters_per_side'],
            config['harmonic_number'], config['bunch_spacing_buckets'],
            bb_suffix_cw=config['bb_suffix_cw'],
            bb_suffix_acw=config['bb_suffix_acw'])
        setup.bb_cw = {
            base: cw[name]
            for base, name in zip(setup.enc_names, setup.bb_names_cw)}
        setup.bb_acw = {
            base: acw[name]
            for base, name in zip(setup.enc_names, setup.bb_names_acw)}
    elif setup.geom:
        raise RuntimeError(
            'Rigid-bunch beam-beam interactions are already configured; use '
            '`setup.set_filling(...)` to change their filling.')

    setup.nemitt_x = nemitt_x
    setup.nemitt_y = nemitt_y
    setup.set_filling(
        filling_scheme_cw=filling_scheme_cw,
        filling_scheme_acw=filling_scheme_acw,
        bunch_intensity_particles_cw=bunch_intensity_particles_cw,
        bunch_intensity_particles_acw=bunch_intensity_particles_acw)
    setup._compute_geometry(
        survey_separation=config['survey_separation'])
    env._rigid_bunch_bb_setup = setup
    return setup


def install_multibunch_beambeam(env, clockwise_line, anticlockwise_line,
                                ips,
                                num_long_range_encounters_per_side,
                                harmonic_number, bunch_spacing_buckets,
                                nemitt_x, nemitt_y,
                                filling_scheme_cw, filling_scheme_acw,
                                bunch_intensity_particles_cw,
                                bunch_intensity_particles_acw,
                                survey_separation=True,
                                bb_suffix_cw='_cw', bb_suffix_acw='_acw'):
    """Compatibility entry point installing rigid-bunch beam-beam elements at
    N IPs of two counter-rotating rings and computing the encounter geometry.
    New code should use ``install_beambeam_interactions(mode='rigid_bunch')``
    followed by ``configure_beambeam_interactions(...)``. See
    :meth:`xtrack.environment.EnvXfields.install_multibunch_beambeam` for the
    full documentation. Returns a :class:`RigidBunchBBSetup`."""
    cw = _resolve_line(env, clockwise_line)
    acw = _resolve_line(env, anticlockwise_line)

    setup = RigidBunchBBSetup(
        cw, acw, ips, num_long_range_encounters_per_side,
        harmonic_number, bunch_spacing_buckets, nemitt_x, nemitt_y,
        bb_suffix_cw=bb_suffix_cw, bb_suffix_acw=bb_suffix_acw)
    setup.set_filling(
        filling_scheme_cw=filling_scheme_cw,
        filling_scheme_acw=filling_scheme_acw,
        bunch_intensity_particles_cw=bunch_intensity_particles_cw,
        bunch_intensity_particles_acw=bunch_intensity_particles_acw)
    setup.bb_cw = setup._place_bb(cw, mirror=False)
    setup.bb_acw = setup._place_bb(acw, mirror=True)
    setup._compute_geometry(survey_separation=survey_separation)
    return setup
