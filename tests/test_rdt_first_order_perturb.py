import xtrack as xt
import xtrack.feed_down as fd
from xtrack.rdt import tracking_from_rdt
import numpy as np
import xobjects as xo
import pytest
import pathlib

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

def test_feed_down_rotation_dipole():
    kn = np.array([1, 0])
    ks = np.array([0, 0])
    psi = np.pi / 2 # 90 degrees tilt

    kn_eff, kskew_eff = fd.feed_down(
        kn=kn[None, :],
        kskew=ks[None, :],
        shift_x=0.,
        shift_y=0.,
        psi=psi,
        x0=0.,
        y0=0.,
        max_output_order=None,
    )

    assert kn_eff.shape == (1, 2)
    assert kskew_eff.shape == (1, 2)

    xo.assert_allclose(kn_eff, np.array([[0, 0]]), atol=1e-15)
    xo.assert_allclose(kskew_eff, np.array([[-1, 0]]), atol=1e-15)

    m = xt.Multipole(knl=kn, ksl=ks, rot_s_rad=psi)
    m_fdown = xt.Multipole(knl=kn_eff[0, :], ksl=kskew_eff[0, :], rot_s_rad=0.0)

    p = xt.Particles()
    p_fdown = p.copy()
    m.track(p)
    m_fdown.track(p_fdown)

    xo.assert_allclose(p.px, p_fdown.px, atol=1e-15)
    xo.assert_allclose(p.py, p_fdown.py, atol=1e-15)

def test_feed_down_rotation_quadrupole():
    kn = np.array([0, 1])
    ks = np.array([0, 0])
    psi = np.pi / 4 # 45 degrees tilt

    kn_eff, kskew_eff = fd.feed_down(
        kn=kn[None, :],
        kskew=ks[None, :],
        shift_x=0.,
        shift_y=0.,
        psi=psi,
        x0=0.,
        y0=0.,
        max_output_order=None,
    )

    m = xt.Multipole(knl=kn, ksl=ks, rot_s_rad=psi)
    m_fdown = xt.Multipole(knl=kn_eff[0, :], ksl=kskew_eff[0, :], rot_s_rad=0.0)

    p = xt.Particles()
    p_fdown = p.copy()
    m.track(p)
    m_fdown.track(p_fdown)

    xo.assert_allclose(p.px, p_fdown.px, atol=1e-15)
    xo.assert_allclose(p.py, p_fdown.py, atol=1e-15)


def test_feed_down_rotation_higher_order():
    kn = np.array([0, 1, 2, 10])
    ks = np.array([0, 3, 2, 20])
    psi = np.pi / 4 # 45 degrees tilt
    shift_x = 0.001
    shift_y = -0.002

    kn_eff, kskew_eff = fd.feed_down(
        kn=kn[None, :],
        kskew=ks[None, :],
        shift_x=shift_x,
        shift_y=shift_y,
        psi=psi,
        x0=0.,
        y0=0.,
        max_output_order=None,
    )

    m = xt.Multipole(knl=kn, ksl=ks, rot_s_rad=psi, shift_x=shift_x, shift_y=shift_y)
    m_fdown = xt.Multipole(knl=kn_eff[0, :], ksl=kskew_eff[0, :])

    p = xt.Particles(x=4e-3, y=5e-3)
    p_fdown = p.copy()
    m.track(p)
    m_fdown.track(p_fdown)

    xo.assert_allclose(p.px, p_fdown.px, atol=1e-15)
    xo.assert_allclose(p.py, p_fdown.py, atol=1e-15)

PIMMS_RTD_CONFIGURATIONS = [
    'sextupole',
    'skew_sextupole',
    'skew_quadrupole',
    'tilted_quadrupole',
    'shifted_octupole_x',
    'shifted_octupole_y',
    'orbit_in_octupole_x',
    'orbit_in_octupole_y',
    'orbit_in_skew_octupole_x',
    'orbit_in_skew_octupole_y',
]

@pytest.mark.parametrize("configuration", PIMMS_RTD_CONFIGURATIONS)
def test_rdt_first_order_perturbation_against_tracking(configuration):

    env = xt.load(test_data_folder / 'pimms/PIMM.seq')
    line = env.pimms
    line.set_particle_ref('proton', kinetic_energy0=100e6)
    line.replace_all_repeated_elements()

    line.env.new('skew_quad', xt.Quadrupole, length=0.2)
    line.env.new('octup', xt.Octupole, length=0.2)

    line.insert('skew_quad', anchor='start', at='xrra@end')
    line.insert('octup', anchor='start', at='skew_quad@end')

    env['qf1k1'] =  3.15396e-01
    env['qd1k1'] = -5.24626e-01
    env['qf2k1'] =  5.22717e-01

    # Build knobs to control the orbit in the straight section with the non-linear
    # elements
    line.vars.default_to_zero = True
    env['qd.3'].knl[0] = 'kx1'
    env['qf1.3'].knl[0] = 'kx2'
    env['qf1.4'].knl[0] = 'kx3'
    env['qd.4'].knl[0] = 'kx4'
    env['qd.3'].ksl[0] = 'ky1'
    env['qf1.3'].ksl[0] = 'ky2'
    env['qf1.4'].ksl[0] = 'ky3'
    env['qd.4'].ksl[0] = 'ky4'
    line.vars.default_to_zero = False

    opt = line.match_knob(
        method='4d',
        betx=1, bety=1,
        knob_name='x_bump_mm',
        vary=xt.VaryList(['kx1', 'kx2', 'kx3', 'kx4'], step=1e-6),
        targets=[
            xt.TargetSet(x=1e-3, px=0, at='skew_quad'),
            xt.TargetSet(x=0.0, px=0, at='qd.5')
        ],
        run=False)
    opt.solve()
    opt.generate_knob()

    opt = line.match_knob(
        method='4d',
        betx=1, bety=1,
        knob_name='y_bump_mm',
        vary=xt.VaryList(['ky1', 'ky2', 'ky3', 'ky4'], step=1e-6),
        targets=[
            xt.TargetSet(y=-1e-3, py=0, at='skew_quad'),
            xt.TargetSet(y=0.0, py=0, at='qd.5')
        ],
        run=False)
    opt.solve()
    opt.generate_knob()

    tw0 = line.twiss4d()

    rtols = {}
    if configuration == 'sextupole':
        env['xrra'].k2 = 0.8
        rdts = ['f3000', 'f1200', 'f1020', 'f0111',
                # 'f0120' # this one in quite bad
                ]
    elif configuration == 'skew_sextupole':
        env['xrrb'].k2s = 0.8
        rdts = ['f0030', 'f0012', 'f2010', 'f0210', 'f1110']
    elif configuration == 'skew_quadrupole':
        env['skew_quad'].k1s = 0.02
        rdts = ['f1001', 'f1010', 'f0110']
    elif configuration == 'tilted_quadrupole':
        env['qd.4'].rot_s_rad = 0.002
        rdts = ['f1001', 'f1010', 'f0110']
    elif configuration == 'shifted_octupole_x':
        env['octup'].k3 = 80.
        env['octup'].shift_x = 0.005
        rdts = ['f3000', 'f1200', 'f1020', 'f0111',
                # 'f0120' # this one in quite bad
                ]
        rtols = {'f3000': 0.5, 'f1200': 0.2, 'f0111': 0.2}
    elif configuration == 'shifted_octupole_y':
        env['octup'].k3 = 80.
        env['octup'].shift_y = -0.005
        rdts = ['f0030', 'f0012', 'f2010', 'f0210', 'f1110']
    elif configuration == 'orbit_in_octupole_x':
        env['octup'].k3 = 80.
        line['x_bump_mm'] = -5.0 # mm
        rdts = ['f3000', 'f1200', 'f1020', 'f0111',
                # 'f0120' # this one in quite bad
                ]
        rtols = {'f1200': 0.2, 'f0111': 0.2}
    elif configuration == 'orbit_in_octupole_y':
        env['octup'].k3 = 80.
        line['y_bump_mm'] = 5.0 # mm
        rdts = ['f0030', 'f0012', 'f2010', 'f0210', 'f1110']
    elif configuration == 'orbit_in_skew_octupole_x':
        env['octup'].k3s = 80.
        line['x_bump_mm'] = 5.0 # mm
        rdts = ['f0030', 'f0012', 'f2010', 'f0210', 'f1110']
    elif configuration == 'orbit_in_skew_octupole_y':
        env['octup'].k3s = 80.
        line['y_bump_mm'] = -5.0 # mm
        rdts = ['f3000', 'f1200', 'f1020', 'f0111',
                # 'f0120' # this one in quite bad
                ]
        rtols = {'f1200': 0.2, 'f0111': 0.2}
    else:
        raise ValueError(f'Unknown configuration {configuration}')


    tw1 = line.twiss4d()

    # Compute strengths
    strengths = line.get_table(attr=True)

    # Generate 20 particles on the x axis
    # particles = line.build_particles(x=3e-3, px=5e-4, y=0, py=0, zeta=0, delta=0)
    particles = line.build_particles(x=3e-3, px=5e-4, y=2e-3, py=3e-5, zeta=0, delta=0)

    # Inspect the particles
    particles.get_table()

    # Track 1000 turns logging turn-by-turn data
    num_turns = 20_000
    line.track(particles, num_turns=num_turns, turn_by_turn_monitor=True,
            with_progress=100)
    rec = line.record_last_track

    nc  = tw0.get_normalized_coordinates(rec)

    # Compute RDTs via first-order perturbation theory
    rdt_vals = xt.rdt_first_order_perturbation(rdt=rdts, twiss=tw0,
                                            orbit=tw1,
                                            strengths=strengths)

    i_part_analyze = 0
    x_norm = nc.x_norm[i_part_analyze, :]
    px_norm = nc.px_norm[i_part_analyze, :]
    y_norm = nc.y_norm[i_part_analyze, :]
    py_norm = nc.py_norm[i_part_analyze, :]

    zx_norm = x_norm - 1j * px_norm
    zy_norm = y_norm - 1j * py_norm
    # tracking from RDTs
    Ix = 0.5 * (zx_norm[0].real**2 + zx_norm[0].imag**2)
    Iy = 0.5 * (zy_norm[0].real**2 + zy_norm[0].imag**2)
    psi_x0 = np.angle(zx_norm[0].real + 1j * zx_norm[0].imag)
    psi_y0 = np.angle(zy_norm[0].real + 1j * zy_norm[0].imag)

    def initial_conditions(Ix, Iy, psi_x0, psi_y0):

        hx_minus, hy_minus = tracking_from_rdt(
            rdts={rr: (rdt_vals[rr][0]) for rr in rdts},
            Ix=Ix,
            Iy=Iy,
            Qx=tw0.qx,
            Qy=tw0.qy,
            psi_x0=psi_x0,
            psi_y0=psi_y0,
            num_turns=1
        )

        return np.array([hx_minus[0].real,
                        hx_minus[0].imag,
                        hy_minus[0].real,
                        hy_minus[0].imag])

    opt = xt.match.opt_from_callable(
            lambda xx: initial_conditions(xx[0], xx[1], xx[2], xx[3]),
            x0=np.array([Ix, Iy, psi_x0, psi_y0]),
            steps=np.array([Ix*1e-4, Ix*1e-4, 1e-4, 1e-4]),
            tar=np.array([zx_norm[0].real, zx_norm[0].imag,
                            zy_norm[0].real, zy_norm[0].imag]),
            tols=[1e-10, 1e-10, 1e-10, 1e-10],
        )
    opt.step()
    res = opt.get_knob_values()
    Ix = res[0]
    Iy = res[1]
    psi_x0 = res[2]
    psi_y0 = res[3]


    hx_minus, hy_minus = tracking_from_rdt(
        rdts={rr: (rdt_vals[rr][0]) for rr in rdts},
        Ix=Ix,
        Iy=Iy,
        Qx=tw0.qx,
        Qy=tw0.qy,
        psi_x0=psi_x0,
        psi_y0=psi_y0,
        num_turns=num_turns
    )

    zx_spectrum = np.fft.fft(zx_norm)
    zy_spectrum = np.fft.fft(zy_norm)

    hx_spectrum = np.fft.fft(hx_minus)
    hy_spectrum = np.fft.fft(hy_minus)

    freqs = np.fft.fftfreq(num_turns)
    freqs[freqs < 0] += 1.0

    import nafflib
    f_hx, s_hx = nafflib.get_tunes_all(hx_minus, N=100)
    f_hy, s_hy = nafflib.get_tunes_all(hy_minus, N=100)
    f_x, s_x = nafflib.get_tunes_all(zx_norm, N=100)
    f_y, s_y = nafflib.get_tunes_all(zy_norm, N=100)

    f_x[f_x < 0] += 1.0
    f_hx[f_hx < 0] += 1.0
    f_y[f_y < 0] += 1.0
    f_hy[f_hy < 0] += 1.0

    f_h = {'x': f_hx, 'y': f_hy}
    s_h = {'x': s_hx, 'y': s_hy}
    f_z = {'x': f_x, 'y': f_y}
    s_z = {'x': s_x, 'y': s_y}

    dq_search = 0.001
    print()
    for rr in rdts:
        print(f'Spectral lines excited by {rr}:')
        for pp in ['x', 'y']:
            if rdt_vals[rr + f'_ampl_{pp}_expr'] == '0':
                print(f'  Expected {pp} freq: not excited')
            elif rdt_vals[rr + f"_freq_{pp}"] == 0:
                print(f'  Expected {pp} freq: zero frequency')
            else:
                print(f'  Expected {pp} freq: {rdt_vals[rr + f"_freq_{pp}_expr"]} = {rdt_vals[rr + f"_freq_{pp}"]:.6f}')
                mask_search_z = (np.abs(f_z[pp] - rdt_vals[rr + f'_freq_{pp}']) < dq_search)
                i_max_z = np.argmax(np.abs(s_z[pp][mask_search_z]))
                f_z_max = f_z[pp][mask_search_z][i_max_z]
                s_z_max = s_z[pp][mask_search_z][i_max_z]
                mask_search_h = (np.abs(f_h[pp] - rdt_vals[rr + f'_freq_{pp}']) < dq_search)
                i_max_h = np.argmax(np.abs(s_h[pp][mask_search_h]))
                f_h_max = f_h[pp][mask_search_h][i_max_h]
                s_h_max = s_h[pp][mask_search_h][i_max_h]
                print(f'    From tracking: freq={f_z_max:.6f}, amp={np.abs(s_z_max):.6e}, phase={np.angle(s_z_max, deg=True):.2f} deg')
                print(f'    From RDTs:     freq={f_h_max:.6f}, amp={np.abs(s_h_max):.6e}, phase={np.angle(s_h_max, deg=True):.2f} deg')
                xo.assert_allclose(f_z_max, f_h_max, atol=5e-4)
                xo.assert_allclose(s_z_max, s_h_max, rtol=rtols.get(rr, 0.1))
        print('')
