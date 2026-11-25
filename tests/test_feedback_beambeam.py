# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2026.                 #
# ######################################### #

import numpy as np
import pytest
import xtrack as xt
from xobjects.test_helpers import (
    allow_kernel_compilation, for_all_test_contexts,
)


@for_all_test_contexts(excluding=('ContextPyopencl',))
@allow_kernel_compilation
def test_ip_feedback_changes_only_the_following_strong_strong_encounter(
    test_context, sandbox_cwd,
):
    """Check the next IP gap against quarter-turn transport."""
    xf = pytest.importorskip('xfields')
    beta_y, gap, correction = 0.1, 2e-6, 0.6
    count, turns = 256, 2
    cloud = np.random.default_rng(418).normal(size=(5, count))
    cloud -= cloud.mean(axis=1, keepdims=True)
    names = ('electron', 'positron')

    def track(active):
        manager = xt.PipelineManager()
        manager.add_element('ip')
        particles = {}
        for name, charge in zip(names, (-1, 1)):
            manager.add_particles(name, 0)
            particles[name] = xt.Particles(
                _context=test_context, mass0=xt.ELECTRON_MASS_EV, q0=charge,
                p0c=45.6e9, x=cloud[0] * 20e-6, px=cloud[4] * 10e-6,
                py=cloud[1] * 10e-6 - charge * gap / (2 * beta_y),
                y=cloud[2] * 2e-6, zeta=cloud[3] * 1e-3,
                weight=5e8 / count,
            )
            particles[name].init_pipeline(name)

        lines, before, after = {}, {}, {}
        for name, partner in zip(names, names[::-1]):
            update = xf.ConfigForUpdateBeamBeamBiGaussian3D(
                pipeline_manager=manager, element_name='ip',
                partner_particles_name=partner, update_every=1,
                slicer=xf.TempSlicer(
                    _context=test_context, n_slices=3, sigma_z=1e-3,
                    mode='shatilov',
                ),
                quasistrongstrong=False,
            )
            before[name], after[name] = [
                xt.BeamStatsMonitor(
                    _context=test_context, stop_at_turn=turns,
                    stats=['mean_y', 'mean_py'],
                )
                for _ in range(2)
            ]
            elements = {
                'corrector': xt.Multipole(ksl=[0.0]),
                'arc': xt.LineSegmentMap(length=1000.0, qy=0.25, bety=beta_y),
                'before': before[name],
                'ip': xf.BeamBeamBiGaussian3D(
                    _context=test_context, phi=0.0, alpha=0.0,
                    other_beam_q0=particles[partner].q0,
                    config_for_update=update,
                ),
                'after': after[name],
            }
            lines[name] = xt.Line(
                elements=elements, element_names=list(elements),
            )

        positron = lines['positron']
        positron['correction_rad'] = 0.0
        positron.ref['corrector'].ksl[0] = positron.ref['correction_rad']

        boundary = xt.feedback.SharedFeedbackBoundary(names)
        boundary.install(lines)
        for line in lines.values():
            line.build_tracker(
                _context=test_context, use_prebuilt_kernels=False,
            )

        def program(inputs):
            relative_angle = inputs['electron'] - inputs['positron']
            return {'outputs': {'command': correction * relative_angle}}

        controller = xt.feedback.FeedbackController(
            'ip_offset', program=program,
            acquisition=xt.feedback.Acquisition(sources={
                name: (after[name], 'mean_py') for name in names
            }),
            output=xt.feedback.KnobOutput(positron, 'correction_rad'),
            first_calculation_turn=1, update_turns=10,
        )
        if not active:
            controller.disable()
        system = xt.feedback.FeedbackSystem([controller], lines=lines)
        boundary.bind(xt.PipelineMultiTracker(branches=[
            xt.PipelineBranch(line=lines[name], particles=particles[name])
            for name in names
        ]), system=system)
        boundary.track(num_turns=turns)

        separation = before['electron'].mean_y - before['positron'].mean_y
        first_kicks = [
            after[name].mean_py[0] - before[name].mean_py[0] for name in names
        ]
        outgoing_angle = (
            after['electron'].mean_py[0] - after['positron'].mean_py[0]
        )
        trim = positron['correction_rad']
        return separation, first_kicks, outgoing_angle, trim

    off, off_kicks, off_angle, off_trim = track(False)
    on, on_kicks, on_angle, trim = track(True)
    assert off[0] == pytest.approx(gap, abs=1e-16)
    assert on[0] == pytest.approx(off[0], abs=1e-16)
    np.testing.assert_allclose(on_kicks, off_kicks, atol=1e-17, rtol=1e-12)
    assert on_kicks[0] * on_kicks[1] < 0  # The two beams deflect oppositely.
    assert on_angle == pytest.approx(off_angle, abs=1e-17)
    assert off_trim == 0.0
    assert trim == pytest.approx(correction * off_angle, abs=1e-17)

    # A quarter turn maps the outgoing relative angle to the next IP gap.
    next_gap = beta_y * off_angle
    assert abs(next_gap) > 1e-9
    assert off[1] == pytest.approx(next_gap, abs=1e-16)
    assert on[1] == pytest.approx((1 - correction) * next_gap, abs=1e-16)
