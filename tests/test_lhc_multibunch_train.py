# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

"""Cross-check of the (generic) multi-bunch beam-beam machinery against pytrain
(TRAIN): drives the machine-independent tools in ``xtrack.multibunch_beambeam``
(``env.xfields.install_multibunch_beambeam`` -> ``setup.second_order_maps`` ->
``setup_red.solve``) on the LHC sector-map model, for both the injection
(BBLR only) and the collision (6.8 TeV squeezed, head-on + BBLR) scenarios --
all 2460+2460 bunches -- and compares the per-bunch closed-orbit deviations and
tune shifts at IP1 with the stored pytrain references
(``test_data/lhc_2024/pytrain/pytrain_{injection,collision}.json``, regenerated
with the scripts in the same directory).

The tools install one ``BeamBeamBiGaussianMultibunch2D`` element per head-on and
long-range encounter at IP1/2/5/8 and find the per-bunch self-consistent closed
orbit by iterating the fast multi-bunch twiss. The geometry (per-encounter
bunch-pairing offset, convolved beam sizes, signed survey separation) is
computed on the full thick lattice, then the arcs are replaced by second-order
maps and the beam-beam elements installed on the reduced lines. This test
exercises the sector-map (building-block) workflow of the generic tools.

NOTE: slow (several minutes per scenario, dominated by building the
second-order maps of both beams).
"""

import json
import pathlib

import numpy as np
import pytest

import xobjects as xo
import xtrack as xt

test_data_folder = pathlib.Path(
    __file__).parent.joinpath('../test_data').absolute()
lhc_data = test_data_folder / 'lhc_2024'

HARMONIC_NUMBER = 35640
BUNCH_SPACING_BUCKETS = 10        # 25 ns
N_SLOTS = HARMONIC_NUMBER // BUNCH_SPACING_BUCKETS      # 3564
NPARASITIC = 45                   # long-range encounters per IP side

SCENARIOS = {
    'injection': dict(p0c=450e9, bunch_intensity=1.8e11, nemitt=1.5e-6,
                      optics='injection_optics.madx', n_iter=3),
    'collision': dict(p0c=6800e9, bunch_intensity=1.1e11, nemitt=2.3e-6,
                      optics='collision_optics_15cm_flat_2026.madx',
                      n_iter=4),
}

# pytrain is a thin (slicefactor 8) model solved with second-order beam-beam
# maps, xsuite an exact-kick model built from a thick lattice, so the two
# never agree to machine precision:
# - orbits: sub-nm to few-nm agreement in collision; at injection a few
#   pytrain bunches show um-level slicing artifacts (measured max 2.5e-5).
# - tune shifts: at injection the pytrain per-bunch tunes are not converging
#   well; in collision the second-order map truncation of the head-on kick
#   gives a ~2% shift (max 5.1e-4).
TOLERANCES = {
    #             orbit atol   tune atol   tune rms
    'injection': dict(orbit=5e-5, tune=1e-1, tune_rms=1e-2),
    'collision': dict(orbit=2e-7, tune=2e-3, tune_rms=1e-3),
}


def _wrap_frac_tune(v):
    """Tune difference on the fractional-tune circle, wrapped to
    (-0.5, 0.5]."""
    return (np.asarray(v) + 0.5) % 1.0 - 0.5


def _load_lines(p0c, optics):
    env = xt.load(str(lhc_data / 'lhc.seq'), format='madx',
                  reverse_lines=['lhcb2'])
    for ln in (env.lhcb1, env.lhcb2):
        ln.particle_ref = xt.Particles(mass0=xt.PROTON_MASS_EV, p0c=p0c)
    env.vars.load(str(lhc_data / optics))
    for ln in (env.lhcb1, env.lhcb2):
        ln.twiss_default['method'] = '4d'
        ln.cycle(name_first_element='ip3', inplace=True)  # no IP at s=0
    return env, env.lhcb1, env.lhcb2


def _run_xsuite_scenario(scenario):
    par = SCENARIOS[scenario]
    env, line_b1, line_b2 = _load_lines(par['p0c'], par['optics'])
    n_slots = HARMONIC_NUMBER // BUNCH_SPACING_BUCKETS

    with open(lhc_data / '25ns_2460b_2448_2092_2239_144bpi_20inj.json') as f:
        scheme = json.load(f)
    # per-slot bunch population (uniform intensity at the filled slots)
    filling_b1 = (np.array(scheme['schemebeam1']) > 0) * par['bunch_intensity']
    filling_b2 = (np.array(scheme['schemebeam2']) > 0) * par['bunch_intensity']

    # Install head-on + long-range beam-beam at IP1/2/5/8 (offsets derived from
    # the ring geometry since the IPs are given as a list of names) on the full
    # lattice, then make a second-order-map copy so the self-consistent solve is
    # fast.
    setup = env.xfields.install_multibunch_beambeam(
        clockwise_line='lhcb1', anticlockwise_line='lhcb2',
        ips=['ip1', 'ip2', 'ip5', 'ip8'],
        num_long_range_encounters_per_side=NPARASITIC,
        harmonic_number=HARMONIC_NUMBER,
        bunch_spacing_buckets=BUNCH_SPACING_BUCKETS,
        nemitt_x=par['nemitt'], nemitt_y=par['nemitt'],
        filling_clockwise=filling_b1, filling_anticlockwise=filling_b2,
        survey_separation=True)
    setup_red = setup.second_order_maps()

    # bare per-bunch tunes (second-order maps preserve the linear optics, so the
    # reduced-line tunes equal the full-lattice ones stored in setup.meta)
    bare = setup.meta

    mbtw_b1, mbtw_b2 = setup_red.solve(
        max_iterations=par['n_iter'], tol_sigma=0.0,
        twiss_mode='fast_orbit', show_progress=False)

    def extract(mbtw, slots, bare_qx, bare_qy, mirror):
        bb = setup_red.bb_name('bb_ip1_ho', mirror)
        x = mbtw['x', bb]
        if mirror:
            x = -x   # reversed beam-2 line -> physical frame
        y = mbtw['y', bb]
        return dict(slots=slots,
                    dx=x - x.mean(), dy=y - y.mean(),
                    dqx=_wrap_frac_tune(mbtw.qx - bare_qx),
                    dqy=_wrap_frac_tune(mbtw.qy - bare_qy))

    return (extract(mbtw_b1, setup_red.bunches_cw,
                    bare['qx_cw'], bare['qy_cw'], mirror=False),
            extract(mbtw_b2, setup_red.bunches_acw,
                    bare['qx_acw'], bare['qy_acw'], mirror=True))


@pytest.mark.parametrize('scenario', ['injection', 'collision'])
def test_lhc_multibunch_train(scenario):
    with open(test_data_folder / 'lhc_2024' / 'pytrain'
              / f'pytrain_{scenario}.json') as fid:
        ref = json.load(fid)
    tol = TOLERANCES[scenario]

    res_b1, res_b2 = _run_xsuite_scenario(scenario)

    for beam, res in (('b1', res_b1), ('b2', res_b2)):
        rr = ref[beam]
        assert np.array_equal(res['slots'], np.asarray(rr['slots']))

        # per-bunch closed-orbit deviation (from the bunch average) at IP1
        xo.assert_allclose(res['dx'], np.asarray(rr['dx']),
                           rtol=0, atol=tol['orbit'])
        xo.assert_allclose(res['dy'], np.asarray(rr['dy']),
                           rtol=0, atol=tol['orbit'])

        # per-bunch beam-beam tune shifts (fractional-tune circle), larger
        # tolerance: pytrain tunes carry thin-slicing / map-truncation errors
        for col in ('dqx', 'dqy'):
            dq_xs = res[col]
            dq_pt = _wrap_frac_tune(np.asarray(rr[col]))
            xo.assert_allclose(dq_xs, dq_pt, rtol=0, atol=tol['tune'])
            rms = float(np.sqrt(np.mean(
                _wrap_frac_tune(dq_xs - dq_pt) ** 2)))
            assert rms < tol['tune_rms'], (
                f'{scenario} {beam} {col}: RMS tune-shift difference '
                f'{rms:.2e} exceeds {tol["tune_rms"]:.0e}')
