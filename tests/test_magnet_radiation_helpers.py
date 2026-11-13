import numpy as np

import xobjects as xo
import xtrack as xt
from xobjects.test_helpers import allow_kernel_compilation, for_all_test_contexts


@for_all_test_contexts
@allow_kernel_compilation
def test_magnet_radiation_direction_and_perpendicular_field(test_context):
    class RadiationFieldProbe(xt.BeamElement):
        _xofields = {}
        _depends_on = xt.Magnet._depends_on
        _internal_record_class = xt.Magnet._internal_record_class
        _extra_c_sources = [
            '#include "xtrack/beam_elements/elements_src/track_magnet_radiation.h"',
            '''
            GPUFUN
            void RadiationFieldProbe_track_local_particle(
                    RadiationFieldProbeData el, LocalParticle* part0){
                START_PER_PARTICLE_BLOCK(part0, part);
                    double const px = LocalParticle_get_px(part);
                    double const py = LocalParticle_get_py(part);
                    double const delta = LocalParticle_get_delta(part);
                    // Use x, y, zeta as field inputs, then direction outputs.
                    double const b_perp = compute_b_perp_mod(px, py, delta,
                        LocalParticle_get_x(part), LocalParticle_get_y(part),
                        LocalParticle_get_zeta(part));
                    double ix, iy, is;
                    direction_of_motion(px, py, delta, &ix, &iy, &is);
                    LocalParticle_set_x(part, ix);
                    LocalParticle_set_y(part, iy);
                    LocalParticle_set_zeta(part, is);
                    LocalParticle_set_s(part, b_perp);
                END_PER_PARTICLE_BLOCK;
            }
            ''',
        ]

    directions = np.array([
        [0.3, 0.0, np.sqrt(0.91)],
        [0.0, 0.4, np.sqrt(0.84)],
        [0.3, -0.4, np.sqrt(0.75)],
    ])
    # Parallel, perpendicular, and general fields for each direction.
    fields = np.concatenate([
        2 * directions,
        np.cross(directions, [1.0, 0.0, 0.0]),
        np.tile([1.0, 2.0, 3.0], (3, 1)),
    ])
    directions = np.tile(directions, (3, 1))
    delta = np.tile([-0.2, 0.0, 0.3], 3)
    particles = xt.Particles(
        _context=test_context, p0c=1e9, delta=delta,
        px=directions[:, 0] * (1 + delta),
        py=directions[:, 1] * (1 + delta),
        x=fields[:, 0], y=fields[:, 1], zeta=fields[:, 2])
    probe = RadiationFieldProbe(_context=test_context)
    probe.compile_kernels(only_if_needed=False)
    probe.track(particles)

    ctx2np = test_context.nparray_from_context_array
    actual_direction = np.column_stack([
        ctx2np(particles.x), ctx2np(particles.y), ctx2np(particles.zeta)])
    xo.assert_allclose(np.linalg.norm(actual_direction, axis=1), 1,
                       rtol=0, atol=1e-14)
    xo.assert_allclose(actual_direction, directions, rtol=0, atol=1e-14)
    # For a unit direction, |B_perp| = |direction x B|.
    expected_b_perp = np.linalg.norm(np.cross(directions, fields), axis=1)
    xo.assert_allclose(ctx2np(particles.s), expected_b_perp,
                       rtol=1e-14, atol=1e-14)
