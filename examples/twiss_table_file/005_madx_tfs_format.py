"""MAD-X-style output with explicit convention conversion.

The format preset only selects and formats fields. This example additionally
prepares exit-point optics and energy-normalized dispersion/chromaticity for a
complete, forward, on-momentum, periodic 4D Twiss table with strengths=True.
Prepare before selecting rows: shifting an already filtered table is incorrect.
Strengths retain Xtrack's definitions. For example, an implicit design bend has
nonzero k0l in Xtrack but MAD-X can report zero K0L with the bend in ANGLE instead.
Exact strength/provenance reproduction requires the original element definitions.
"""

import numpy as np

import xtrack as xt


def prepare_madx_conventions(tw):
    """Example preparation for the complete 4D Twiss table described above."""
    if (tw.values_at != 'entry' or tw.reference_frame != 'proper'
            or tw.method != '4d' or not tw.periodic
            or tw.name[-1] != '_end_point'):
        raise ValueError('This example requires a complete forward periodic 4D entry Twiss table')
    if not np.allclose(tw.delta, 0, rtol=0, atol=1e-14):
        raise ValueError('This example requires an on-momentum Twiss table')

    spec = xt.format_spec.madx_twiss_v1
    columns = [field['source_name'] for field in spec['columns']]
    # Keep each element's name and strengths, dropping the synthetic endpoint.
    data = {name: tw[name][:-1].copy() for name in columns}

    # MAD-X's default TWISS values are at element exits. Xtrack's values at the
    # next element's entry give those exits, including zero-length elements.
    for name in ('s', 'x', 'y', 'px', 'py', 'betx', 'bety', 'alfx', 'alfy',
                 'mux', 'muy', 'dx', 'dy', 'dpx', 'dpy'):
        data[name] = tw[name][1:].copy()

    # At delta=0, d(ptau)/d(delta) = beta0. MAD-X uses the ptau normalization
    # for these derivatives; Xtrack uses delta.
    for name in ('dx', 'dy', 'dpx', 'dpy'):
        data[name] /= tw.beta0
    for field in spec['scalars']:
        if 'source_name' in field:
            name = field['source_name']
            data[name] = tw[name]
    data['dqx'] /= tw.beta0
    data['dqy'] /= tw.beta0
    return xt.Table(data, col_names=columns)


if __name__ == '__main__':
    line = xt.Line(elements=[
        xt.Drift(length=1), xt.Multipole(knl=[0, 0.1]),
        xt.Drift(length=1), xt.Multipole(knl=[0, -0.1]),
    ] * 4)
    line.particle_ref = xt.Particles(p0c=1e9, mass0=xt.PROTON_MASS_EV)
    tw = line.twiss4d(strengths=True)

    # Formatting only: values and row conventions remain those of Xtrack.
    tw.to_tfs('madx_layout.tfs', format_spec=xt.format_spec.madx_twiss_v1)

    # Formatting plus the explicit convention changes above.
    prepared = prepare_madx_conventions(tw)
    prepared.to_tfs('madx_conventions.tfs', format_spec=xt.format_spec.madx_twiss_v1)
