import numpy as np
import xtrack as xt
import xobjects as xo
from cpymad.madx import Madx

test_context = xo.context_default

mad = Madx()

dr_len = 1e-11
mad.input(f"""
ss: sequence, l={dr_len};
    lens: nllens, at=0, cnll=0.15, knll=0.3;
    ! since in MAD-X we can't track a zero-length line, we put in
    ! this tiny drift here at the end of the sequence:
    dr: drift, at={dr_len / 2}, l={dr_len};
endsequence;
beam;
use, sequence=ss;
""")

line = xt.Line.from_madx_sequence(mad.sequence.ss)
line.config.XTRACK_USE_EXACT_DRIFTS = True # to be consistent with madx
line.build_tracker(_context=test_context)

num_p_test = 10
x_test = np.linspace(-1e-2, 2e-2, num_p_test)
y_test = np.linspace(-3e-2, 1e-2, num_p_test)
px_test = np.linspace(-2e-5, 4e-5, num_p_test)
py_test = np.linspace(-4e-5, 2e-5, num_p_test)

p0 = xt.Particles(p0c=2e9, x=x_test, px=px_test, y=y_test, py=py_test,
                  zeta=.1, ptau=1e-3)

part = p0.copy(_context=test_context)
line.track(part, _force_no_end_turn_actions=True)
part.move(_context=xo.context_default)

xt_tau = part.zeta/part.beta0
px = []
py = []
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

    px.append(mad_results.px)
    py.append(mad_results.py)


    assert np.allclose(part.x[ii], mad_results.x, atol=1e-14, rtol=0), 'x'
    assert np.allclose(part.px[ii], mad_results.px, atol=1e-14, rtol=0), 'px'
    assert np.allclose(part.y[ii], mad_results.y, atol=1e-14, rtol=0), 'y'
    assert np.allclose(part.py[ii], mad_results.py, atol=1e-14, rtol=0), 'py'
    assert np.allclose(xt_tau[ii], mad_results.t, atol=1e-14, rtol=0), 't'
    assert np.allclose(part.ptau[ii], mad_results.pt, atol=1e-14, rtol=0), 'pt'
    assert np.allclose(part.s[ii], mad_results.s, atol=1e-14, rtol=0), 's'

import matplotlib.pyplot as plt
plt.close('all')

plt.figure(1)
plt.plot(x_test, px, '.')
plt.plot(x_test, part.px, 'x')
plt.plot(x_test, py, '.')
plt.plot(x_test, part.py, 'x')
plt.show()
