import pathlib
import numpy as np
from scipy.constants import e as qe
from scipy.constants import c as clight
from scipy.constants import epsilon_0

import xpart as xp
import xtrack as xt
import xobjects as xo

test_data_folder = pathlib.Path(
        __file__).parent.joinpath('../test_data').absolute()

def test_radiation():

    for context in xo.context.get_test_contexts():
        print(f"Test {context.__class__}")

        theta_bend = 0.05
        L_bend = 5.

        dipole_ave = xt.Multipole(knl=[theta_bend], length=L_bend, hxl=theta_bend,
                                  radiation_flag=1, _context=context)
        dipole_rnd = xt.Multipole(knl=[theta_bend], length=L_bend, hxl=theta_bend,
                                  radiation_flag=2, _context=context)

        particles_ave = xp.Particles(
                _context=context,
                p0c=5e9, # 5 GeV
                x=np.zeros(1000000),
                px=1e-4,
                py=-1e-4,
                mass0=xp.ELECTRON_MASS_EV)
        particles_rnd = particles_ave.copy()

        dct_ave_before = particles_ave.to_dict()
        dct_rng_before = particles_rnd.to_dict()

        particles_ave._init_random_number_generator()
        particles_rnd._init_random_number_generator()

        dipole_ave.track(particles_ave)
        dipole_rnd.track(particles_rnd)

        dct_ave = particles_ave.to_dict()
        dct_rng = particles_rnd.to_dict()

        assert np.allclose(dct_ave['delta'], np.mean(dct_rng['delta']),
                          atol=0, rtol=5e-3)

        # assert np.allclose(dct_ave['px']*dct_ave['rpp'],
        #                    dct_ave_before['px']*dct_ave_before['rpp'],
        #                    atol=0, rtol=1e-10)

        # assert np.allclose(dct_rng['px']*dct_rng['rpp'],
        #                    dct_rng_before['px']*dct_rng_before['rpp'],
        #                    atol=0, rtol=1e-10)

        # assert np.allclose(dct_ave['py']*dct_ave['rpp'],
        #                    dct_ave_before['py']*dct_ave_before['rpp'],
        #                    atol=0, rtol=1e-10)

        # assert np.allclose(dct_rng['py']*dct_rng['rpp'],
        #                    dct_rng_before['py']*dct_rng_before['rpp'],
        #                    atol=0, rtol=1e-10)

        rho = L_bend/theta_bend
        mass0_kg = (dct_ave['mass0']*qe/clight**2)
        r0 = qe**2/(4*np.pi*epsilon_0*mass0_kg*clight**2)
        Ps = (2*r0*clight*mass0_kg*clight**2*
              dct_ave['beta0'][0]**4*dct_ave['gamma0'][0]**4)/(3*rho**2) # W

        Delta_E_eV = -Ps*(L_bend/clight) / qe
        Delta_trk = (dct_ave['ptau']-dct_ave_before['ptau'])*dct_ave['p0c']

        assert np.allclose(Delta_E_eV, Delta_trk, atol=0, rtol=1e-6)


def test_ring_with_radiation():

    from cpymad.madx import Madx

    # Import thick sequence
    mad = Madx()

    # CLIC-DR
    mad.call(str(test_data_folder.joinpath('clic_dr/sequence.madx')))
    mad.use('ring')

    # Twiss
    twthick = mad.twiss().dframe()

    # Emit
    mad.sequence.ring.beam.radiate = True
    mad.emit()
    mad_emit_table = mad.table.emit.dframe()
    mad_emit_summ = mad.table.emitsumm.dframe()

    # Makethin
    mad.input(f'''
    select, flag=MAKETHIN, SLICE=4, thick=false;
    select, flag=MAKETHIN, pattern=wig, slice=1;
    MAKETHIN, SEQUENCE=ring, MAKEDIPEDGE=true;
    use, sequence=RING;
    ''')
    mad.use('ring')
    mad.twiss()

    # Build xtrack line
    line = xt.Line.from_madx_sequence(mad.sequence['RING'])
    line.particle_ref = xp.Particles(
            mass0=xp.ELECTRON_MASS_EV,
            q0=-1,
            gamma0=mad.sequence.ring.beam.gamma)

    for context in xo.context.get_test_contexts():
        # Build tracker
        tracker = xt.Tracker(line=line, _context=context)
        tracker.matrix_stability_tol = 1e-2

        tracker.configure_radiation(mode='mean')

        # Twiss
        tw = tracker.twiss(eneloss_and_damping=True)

        # Checks
        met = mad_emit_table
        assert np.isclose(tw['eneloss_turn'], mad_emit_summ.u0[0]*1e9,
                        rtol=3e-3, atol=0)
        assert np.isclose(tw['damping_constants_s'][0],
            met[met.loc[:, 'parameter']=='damping_constant']['mode1'][0],
            rtol=3e-3, atol=0
            )
        assert np.isclose(tw['damping_constants_s'][1],
            met[met.loc[:, 'parameter']=='damping_constant']['mode2'][0],
            rtol=1e-3, atol=0
            )
        assert np.isclose(tw['damping_constants_s'][2],
            met[met.loc[:, 'parameter']=='damping_constant']['mode3'][0],
            rtol=3e-3, atol=0
            )

        assert np.isclose(tw['partition_numbers'][0],
            met[met.loc[:, 'parameter']=='damping_partion']['mode1'][0],
            rtol=3e-3, atol=0
            )
        assert np.isclose(tw['partition_numbers'][1],
            met[met.loc[:, 'parameter']=='damping_partion']['mode2'][0],
            rtol=1e-3, atol=0
            )
        assert np.isclose(tw['partition_numbers'][2],
            met[met.loc[:, 'parameter']=='damping_partion']['mode3'][0],
            rtol=3e-3, atol=0
            )

        tracker.configure_radiation(mode='mean')
        part_co = tracker.find_closed_orbit()
        par_for_emit = xp.build_particles(tracker=tracker, _context=context,
                                        x_norm=50*[0],
                                        zeta=part_co.zeta[0], delta=part_co.delta[0],
                                        )
        tracker.configure_radiation(mode='quantum')

        num_turns=1500
        tracker.track(par_for_emit, num_turns=num_turns, turn_by_turn_monitor=True)
        mon = tracker.record_last_track

        assert np.isclose(np.std(mon.zeta[:, 750:]),
            np.sqrt(met[met.loc[:, 'parameter']=='emittance']['mode3'][0]*tw['betz0']),
            rtol=0.2, atol=0
            )

        assert np.isclose(np.std(mon.x[:, 750:]),
            np.sqrt(met[met.loc[:, 'parameter']=='emittance']['mode1'][0]*tw['betx'][0]),
            rtol=0.2, atol=0
            )

        assert np.all(mon.y[:] < 1e-15)
