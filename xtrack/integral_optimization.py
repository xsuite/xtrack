import numpy as np
import xtrack as xt
import xdeps as xd

NORMAL_STRENGTHS_FROM_ATTR=['k0l', 'k1l', 'k2l', 'k3l', 'k4l', 'k5l']
SKEW_STRENGTHS_FROM_ATTR=['k0sl', 'k1sl', 'k2sl', 'k3sl', 'k4sl', 'k5sl']

class IntegralOptimization:
    def __init__(self, line, twiss, start, end, vary,
                 target_quantities, generated_knob_name,
                 scale_multipoles=None, feed_down=True, orbit=None):

        '''
        Correction based on RDT or other integrals dependent on twiss parameters,
        orbit, and strengths along a line. The correction generates a knob that
        when activated applies the correction.

        Parameters
        ----------
        line: xt.Line
            Line on which the correction is applied. It should contain the relevant
            sources and correctors for the integral correction.
        twiss: xt.Table
            Twiss table to be used to compute the integrand.
        start: str
            Name of the element at which the integral starts.
        end: str
            Name of the element at which the integral ends.
        vary: list xt.Vary
            List of knobs to be varied for the correction, together with their
            step and, optionally, limits.
        target_quantities: dict
            Dictionary with keys being the names of the target quantities to be
            corrected and values being either strings with the RDT to be corrected
            or callables that take twiss and line table rows and return a complex
            integrand to be minimized in absolute value.
        generated_knob_name: str
            Name of the knob that will be generated for the correction.
        scale_multipoles: array-like
            Array with the same length as the line table to be used to scale the
            multipoles when computing the integrand. This can be used to, for
            example, only consider certain elements as sources in the integral.
        feed_down: bool
            Whether to consider feed-down in the RDT computation (only relevant if
            target_quantities contains RDTs). Default is True.
        orbit: xt.Table
            Table containing orbit information to be used in the integrand computation
            if feed_down is True. It should have the same length and indexing as the
            twiss table.
        '''

        self.env = line.env
        self.twiss = twiss
        self.line = line
        self.start = start
        self.end = end
        self.vary = vary
        self.target_quantities = target_quantities
        self.generated_knob_name = generated_knob_name
        self.scale_multipoles = scale_multipoles
        self.feed_down = feed_down
        self.orbit = orbit

        self.knob_opt = None
        self.rdt_terms = {}

    def run(self):
        if self.line.tracker is None:
            self.line.build_tracker()

        # I do this instead of line.get_table to be faster
        tt0 = self.line.tracker._tracker_data_base._line_table
        tt = xt.Table(data={'name': tt0['name'], 'env_name': tt0['env_name'],
                            'parent_name': tt0['parent_name'], 's': tt0['s']})
        for kk in (NORMAL_STRENGTHS_FROM_ATTR + SKEW_STRENGTHS_FROM_ATTR
                   + ['shift_x', 'shift_y', 'rot_s_rad']):
            tt[kk] = np.concatenate([self.line.attr[kk], [0]])

        if self.scale_multipoles is not None:
            assert len(self.scale_multipoles) == len(tt)
            for kk in NORMAL_STRENGTHS_FROM_ATTR + SKEW_STRENGTHS_FROM_ATTR:
                tt[kk] *= self.scale_multipoles

        tt_range = tt.rows[self.start:self.end]

        ### The following could allow to use twiss and orbit tables from a line
        ### that does not have all the elements but contains the relevant sources
        ### and correctors. It is only partially tested and becomes problematic
        ### in the presence of repeated elements.
        #
        # # Identify elements controlled by correction knobs
        # elements_in_range = set(list(tt_range.env_name) + list(tt_range.parent_name))
        # correction_elements = []
        # for kk in self.correction_knobs:
        #     for tt in self.line.ref[kk]._find_dependant_targets():
        #         if isinstance(tt, xd.refs.ItemRef) and tt._key in elements_in_range:
        #             correction_elements.append(tt._key)
        #
        # mask_corr = tt_range.rows.mask[list(correction_elements)]
        # tt_integral = tt_range.rows[(tt_range[self.multipole] != 0) | (mask_corr)]
        #
        # tw_integral = self.twiss.rows[tt_integral.env_name]
        # orbit_integral = None
        # if self.orbit is not None:
        #     assert len(self.orbit) == len(self.twiss)
        #     orbit_integral = self.orbit.rows[tt_integral.env_name]

        tt_integral = tt_range
        tw_integral = self.twiss.rows[tt_integral.name]
        if self.orbit is not None:
            assert len(self.orbit) == len(self.twiss)
            orbit_integral = self.orbit.rows[tt_integral.name]
        else:
            orbit_integral = None

        for nntq, ttqq in self.target_quantities.items():

            if isinstance(ttqq, str):
                rdts = xt.rdt_first_order_perturbation(
                    rdt=[ttqq],
                    twiss=tw_integral,
                    strengths=tt_integral,
                    feed_down=self.feed_down,
                    orbit=orbit_integral
                )
                integrand = rdts[f"{ttqq}_integrand"]
            else:
                # I assume it's a callable
                integrand = ttqq(tw_integral, tt_integral)

            self.rdt_terms[nntq] = np.abs(np.sum(integrand))
            self.rdt_terms[nntq+'_integrand'] = integrand
        self.rdt_terms['s'] = tt_integral.s

        return self.rdt_terms

    def _make_optimizer(self):
        action_rdt_contrib = xt.Action(self.run)

        knob_opt = self.env.match_knob(
            knob_name=self.generated_knob_name,
            run=False,
            vary=self.vary,
            targets=[
                action_rdt_contrib.target(nttqq, 0.0)
                    for nttqq in self.target_quantities.keys()
            ])
        self.knob_opt = knob_opt

    def get_optimizer(self):
        if self.knob_opt is None:
            self._make_optimizer()
        return self.knob_opt

    def correct(self, n_steps=1):

        opt = self.get_optimizer()
        opt.step(n_steps)
        opt.generate_knob()

        return opt