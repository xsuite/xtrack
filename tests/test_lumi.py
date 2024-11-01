import numpy as np

import xobjects as xo
import xpart as xp
import xtrack as xt


def test_lumi_calculation():

    # Some columns are neede just not to upset the reverse
    twiss_b1 = xt.twiss.TwissTable(
        data=dict(
            name=np.array([   'ip3',   'ip5',    'ip5_exit', 'ip1',    'ip1_exit',    '_end_point']),
            betx=np.array([   0,       55.0e-2,  55.0e-2,    55.0e-2,  55.0e-2,       0]),
            bety=np.array([   0,       55.0e-2,  55.0e-2,    55.0e-2,  55.0e-2,       0]),
            px=np.array([     0,       285e-6/2, 285e-6/2,   0,        0,             0]),
            py=np.array([     0,       0,        0,          285e-6/2, 285e-6/2,      0]),
            alfx=np.array([   0,       0,        0,          0,        0,             0]),
            alfy=np.array([   0,       0,        0,          0,        0,             0]),
            dx=np.array([     0,       0,        0,          0,        0,             0]),
            dpx=np.array([    0,       0,        0,          0,        0,             0]),
            dy=np.array([     0,       0,        0,          0,        0,             0]),
            dpy=np.array([    0,       0,        0,          0,        0,             0]),
            x=np.array([      0,       0,        0,          0,        0,             0]),
            y=np.array([      0,       0,        0,          0,        0,             0]),
            s=np.array([      0,       0,        0,          0,        0,             0]),
            dx_zeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dy_zeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dpx_zeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dpy_zeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),

            zeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            delta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ptau=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            gamx=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            gamy=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            mux=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            muy=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            muzeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dzeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            phix=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            phiy=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            phizeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            W_matrix=np.zeros(shape=(6, 6, 6))
        ))

    twiss_b2 = xt.twiss.TwissTable(
        data=dict(
            s=np.array([0, 0, 0, 0, 0,0]),
            name=np.array(['ip3',   'ip1',    'ip1_exit', 'ip5',    'ip5_exit',    '_end_point']),
            betx=np.array([0,       55.0e-2,  55.0e-2,    55.0e-2,  55.0e-2,       0]),
            bety=np.array([0,       55.0e-2,  55.0e-2,    55.0e-2,  55.0e-2,       0]),
            px=np.array([  0,       0,        0,         -285e-6/2, -285e-6/2,     0]),
            py=np.array([  0,       285e-6/2, 285e-6/2,   0,       0,              0]),
            alfx=np.array([0,       0,        0,          0,        0,             0]),
            alfy=np.array([0,       0,        0,          0,        0,             0]),
            dx=np.array([  0,       0,        0,          0,        0,             0]),
            dpx=np.array([ 0,       0,        0,          0,        0,             0]),
            dy=np.array([  0,       0,        0,          0,        0,             0]),
            dpy=np.array([ 0,       0,        0,          0,        0,             0]),
            x=np.array([   0,       0,        0,          0,        0,             0]),
            y=np.array([   0,       0,        0,          0,        0,             0]),
            dx_zeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dy_zeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dpx_zeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dpy_zeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),

            # Just not to upset the reverse
            zeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            delta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ptau=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            gamx=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            gamy=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            mux=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            muy=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            muzeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            dzeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            phix=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            phiy=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            phizeta=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        W_matrix=np.zeros(shape=(6, 6, 6))
        ))


    twiss_b1._data['T_rev0'] = 8.892446333483924e-05
    twiss_b1._data['particle_on_co'] = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12)
    twiss_b1._data['values_at'] = 'entry'
    twiss_b1._data['reference_frame'] = 'proper'
    twiss_b1._data['only_markers'] = False

    twiss_b2._data['T_rev0'] = 8.892446333483924e-05
    twiss_b2._data['particle_on_co'] = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1, p0c=7e12)
    twiss_b2._data['values_at'] = 'entry'
    twiss_b2._data['reference_frame'] = 'proper'
    twiss_b2._data['only_markers'] = False

    n_colliding_bunches = 2808
    num_particles_per_bunch = 1.15e11
    nemitt_x = 3.75e-6
    nemitt_y = 3.75e-6
    sigma_z = 0.0755

    ll_ip1 = xt.lumi.luminosity_from_twiss(
        n_colliding_bunches=n_colliding_bunches,
        num_particles_per_bunch=num_particles_per_bunch,
        ip_name='ip1',
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        twiss_b1=twiss_b1,
        twiss_b2=twiss_b2,
        crab=False)

    xo.assert_allclose(ll_ip1, 1.0e+34, rtol=1e-2, atol=0)

    ll_ip5 = xt.lumi.luminosity_from_twiss(
        n_colliding_bunches=n_colliding_bunches,
        num_particles_per_bunch=num_particles_per_bunch,
        ip_name='ip5',
        nemitt_x=nemitt_x,
        nemitt_y=nemitt_y,
        sigma_z=sigma_z,
        twiss_b1=twiss_b1,
        twiss_b2=twiss_b2,
        crab=False)

    xo.assert_allclose(ll_ip5, 1.0e+34, rtol=1e-2, atol=0)