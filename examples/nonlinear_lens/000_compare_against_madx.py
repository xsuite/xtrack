import numpy as np
import xtrack as xt
import xpart as xp
import xobjects as xo
from cpymad.madx import Madx

test_context = xo.context_default

mad = Madx()

dr_len = 1e-11
mad.input(f"""
ss: sequence, l={dr_len};
    lens: nllens, at=0, cnll=2, knll=0.3;
    ! since in MAD-X we can't track a zero-length line, we put in
    ! this tiny drift here at the end of the sequence:
    dr: drift, at={dr_len / 2}, l={dr_len};
endsequence;
beam;
use, sequence=ss;
""")

line = xt.Line.from_madx_sequence(mad.sequence.ss)
line.build_tracker(_context=test_context)

p0 = xp.Particles(p0c=2e9, x=1e-3, px=2e-6, y=3e-3, py=4e-6, zeta=5e-3, ptau=6e-4)

for ii in range(len(p0.x)):
    mad.input(f"""
    beam, particle=proton, pc={p0.p0c[ii] / 1e9}, sequence=ss, radiate=FALSE;

    track, onepass, onetable;
    start, x={p0.x[ii]}, px={p0.px[ii]}, y={p0.y[ii]}, py={p0.py[ii]}, \
        t={p0.zeta[ii]/p0.beta0[ii]}, pt={p0.ptau[ii]};
    run,
        turns=1,
        track_harmon=1e-15;  ! since in this test we don't care about
            ! losing particles due to t difference, we set track_harmon to
            ! something very small, to make t_max large.
    endtrack;
    """)

    mad_results = mad.table.mytracksumm[-1]

    part = p0.copy(_context=test_context)
    line.track(part, _force_no_end_turn_actions=True)
    part.move(_context=xo.context_default)

    xt_tau = part.zeta/part.beta0
    assert np.allclose(part.x[ii], mad_results.x, atol=1e-10, rtol=0), 'x'
    assert np.allclose(part.px[ii], mad_results.px, atol=1e-11, rtol=0), 'px'
    assert np.allclose(part.y[ii], mad_results.y, atol=1e-10, rtol=0), 'y'
    assert np.allclose(part.py[ii], mad_results.py, atol=1e-11, rtol=0), 'py'
    assert np.allclose(xt_tau[ii], mad_results.t, atol=1e-9, rtol=0), 't'
    assert np.allclose(part.ptau[ii], mad_results.pt, atol=1e-11, rtol=0), 'pt'
    assert np.allclose(part.s[ii], mad_results.s, atol=1e-11, rtol=0), 's'

