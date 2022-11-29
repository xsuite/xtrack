import xtrack as xt
import xpart as xp
import xobjects as xo
import numpy as np

def test_twiss_4d_fodo_vs_beta_rel():

    for context in xo.context.get_test_contexts():

        print('Testing on context', context)

        ## Generate a simple line
        n = 6
        fodo = [
            xt.Multipole(length=0.2, knl=[0, +0.2], ksl=[0, 0]),
            xt.Drift(length=1.0),
            xt.Multipole(length=0.2, knl=[0, -0.2], ksl=[0, 0]),
            xt.Drift(length=1.0),
            xt.Multipole(length=1.0, knl=[2*np.pi/n], hxl=[2*np.pi/n]),
            xt.Drift(length=1.0),
        ]
        line = xt.Line(elements=n*fodo + [xt.Cavity(frequency=1e9, voltage=0, lag=180)])
        tracker = line.build_tracker(_context=context)

        ## Twiss
        p0c_list = [1e8,1e9,1e10,1e11,1e12]
        tw_4d_list = []
        for p0c in p0c_list:
            tracker.particle_ref=xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=p0c)
            tw = tracker.twiss(method="4d", at_s = np.linspace(0, line.get_length(), 500))
            tw_4d_list.append(tw)

        for tw in tw_4d_list:
            assert np.allclose(tw.betx, tw_4d_list[0].betx, atol=1e-12, rtol=0)
            assert np.allclose(tw.bety, tw_4d_list[0].bety, atol=1e-12, rtol=0)
            assert np.allclose(tw.alfx, tw_4d_list[0].alfx, atol=1e-12, rtol=0)
            assert np.allclose(tw.alfy, tw_4d_list[0].alfy, atol=1e-12, rtol=0)
            assert np.allclose(tw.dx, tw_4d_list[0].dx, atol=1e-8, rtol=0)
            assert np.allclose(tw.dy, tw_4d_list[0].dy, atol=1e-8, rtol=0)
            assert np.allclose(tw.dpx, tw_4d_list[0].dpx, atol=1e-8, rtol=0)
            assert np.allclose(tw.dpy, tw_4d_list[0].dpy, atol=1e-8, rtol=0)
            assert np.isclose(tw.qx, tw_4d_list[0].qx, atol=1e-7, rtol=0)
            assert np.isclose(tw.qy, tw_4d_list[0].qy, atol=1e-7, rtol=0)
            assert np.isclose(tw.dqx, tw_4d_list[0].dqx, atol=1e-4, rtol=0)
            assert np.isclose(tw.dqy, tw_4d_list[0].dqy, atol=1e-4, rtol=0)