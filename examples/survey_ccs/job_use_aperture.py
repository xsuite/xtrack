"""Generate M2 tracking apertures for several horizontal scraper openings.

Run with ``python job_use_aperture.py`` (or ``%run job_use_aperture.py``).
``lines_by_gap`` contains independent tracking lines, keyed by half-gap in metres.
The native aperture model is the source of geometry; LimitPolygon elements are
generated snapshots. Regenerate them after changing the settings or geometry.
Use ``line.to_json(path, include_aperture=True)`` to save the native model and
its expression bindings together with the line and variables.
"""

from pathlib import Path

import numpy as np
import xtrack as xt


def make_tracking_line(line, max_step=1.0):
    """Sample the current aperture and install limits in an independent line.

    ``max_step`` is the maximum longitudinal distance [m] between checks inside
    each pipe. Polygon accuracy is controlled by ``line.aperture.num_profile_points``.
    No limits are generated in gaps where the MAD-X sequence has no aperture.
    """
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError('max_step must be positive and finite.')
    aperture = line.aperture
    samples = []
    for pipe in aperture.get_pipe_table().rows:
        start, end = float(pipe.s_start), float(pipe.s_end)
        # Sample just inside the ends to avoid ambiguities at frame changes.
        margin = min(10 * aperture.s_tol, (end - start) / 4)
        count = max(2, int(np.ceil((end - start) / max_step)) + 1)
        samples.extend(np.linspace(start + margin, end - margin, count))

    limits = aperture.get_limit_elements_at_s(np.unique(samples))
    # A deep copy freezes both optics knobs and geometry for this scan point.
    tracking_line = line.copy()
    insertions = []
    for index, (s, limit) in enumerate(limits.items()):
        name = f'm2_aperture_{index:04d}'
        tracking_line.env.elements[name] = limit
        insertions.append(tracking_line.env.place(name, at=s))
    tracking_line.insert(insertions)
    return tracking_line


def main():
    here = Path(__file__).resolve().parent
    env = xt.load(here / 'm2_mtn3p8_v6_notilt.py')
    env.vars.load(here / 'M2_MTN3p8_v5_notilt.str')
    line = env['m2']
    line.particle_ref = xt.Particles(
        mass0=xt.PROTON_MASS_EV, p0c=env['beam_momentum'] * 1e9)

    knob = 'kxcmh.x0610727_scr3h_coll1h'
    scraper = 'xcmh.x0610727'
    lines_by_gap = {}
    print('half-gap [mm]    k1 [m^-2]       aperture x [mm]       limits')
    for gap in (0.010, 0.028, 0.050):
        # The same knob also changes the scraper's quadrupole strength.
        env[knob] = gap
        # Queries refresh only the profiles and bounds changed by this knob.
        tracking_line = make_tracking_line(line, max_step=1.0)
        lines_by_gap[gap] = tracking_line

        extents = line.aperture.get_extents_at_element(scraper, resolution=None)
        count = sum(isinstance(tracking_line.element_dict[name], xt.LimitPolygon)
                    for name in tracking_line.element_names)
        print(f'{gap * 1e3:13.1f}  {line[scraper].k1:12.6g}  '
              f'[{extents.min_x[0] * 1e3:7.2f}, '
              f'{extents.max_x[0] * 1e3:7.2f}]  {count:8d}')

    # For example: lines_by_gap[0.028].track(particles, num_turns=1)
    return lines_by_gap


if __name__ == '__main__':
    lines_by_gap = main()
